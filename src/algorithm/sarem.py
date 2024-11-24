import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot, cross_entropy, binary_cross_entropy
from copy import deepcopy
from typing import List, Optional
from src.model_utils import *
from src.MPE_methods.dedpul import dedpul
import logging
import wandb
from src.core_utils import *
from abstention.calibration import  VectorScaling
import os
import time

import src.algorithm.constrained_optimization as constrained_optimization
from src.algorithm.constrained_optimization.problem import ConstrainedMinimizationProblem
from src.algorithm.constrained_optimization.lagrangian_formulation import LagrangianFormulation
from src.algorithm.constrained_optimization.optim import *
from src.algorithm.constrained_optimization.constrained_optimizer import ConstrainedOptimizer
from src.algorithm.constrained_optimization.problem import CMPState
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, average_precision_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler
from src.plots.tsne_plot import *
from src.data_utils import *
from tqdm import tqdm

log = logging.getLogger("app")

class TrainSAREM(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_source_classes: int = 10,
        dataset: str = "CIFAR10",
        learning_rate: float = 0.1,
        dual_learning_rate: float = 2e-2,
        target_recall: float = 0.04,
        logit_multiplier: float = 2.,
        target_precision: float = 0.99,
        precision_confidence: float = 0.95,
        weight_decay: float = 1e-4,
        penalty_type: float = 'l2',
        max_epochs: int = 500,
        inner_epochs: int = 50,
        warmup_epochs: int = 0,
        warmup_patience: int = 0,
        epochs_for_each_alpha: int = 20,
        online_alpha_search: bool = False,
        pred_save_path: str = "./outputs/",
        work_dir: str = ".",
        hash: Optional[str] = None,
        pretrained: bool = False,
        seed: int = 0,
        separate: bool = False,
        pretrained_model_dir: Optional[str] = None,
        pretrained_model_path: str = None,
        device: str = "cuda",
        mode: str = "domain_disc",
        ood_class: int = 0,
        ood_class_ratio: float = 0.005,
        fraction_ood_class: float = 0.01,
        constrained_penalty: float = 3e-7,
        save_model_path: str = "/export/r36a/data/schaud35/shiftpu/models/",
        use_superclass: bool = False,
        data_dir: str = "/export/r36a/data/schaud35/shiftpu/models/",
        use_labels: bool = False,
        clip: float = 5.0,
        refit: bool = False,
    ):
        super().__init__()
        self.seed = seed
        self.num_classes = num_source_classes
        self.fraction_ood_class = fraction_ood_class
        self.use_superclass = use_superclass
        self._device = device
        self.clip = clip
        self.arch = arch
        self.data_dir = data_dir

        self.num_outputs = 2 # + self.num_classes
        self.dataset = dataset
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.constrained_penalty = constrained_penalty
        self.penalty_type = penalty_type
        self.mode = mode
        self.start = 0
        self.pretrained_model_dir = save_model_path
        self.pretrained_model_path = save_model_path + self.dataset + "_" + "SAREM_seed_"+str(seed)+"_num_source_cls_"+str(num_source_classes)+"_fraction_ood_class_"+str(fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"/supervised_pretrained_novelty_detector_constrained_opt.pth" # "/cis/home/schaud35/shiftpu/models/imagenet_CoNoC_seed_"+str(seed)+"_ood_ratio_"+str(ood_class_ratio)+"/"
        
        self.novelty_detector, self.novelty_optimizer = get_model(arch, data_dir, self.dataset, self.num_outputs, pretrained= self.pretrained,
                                                                 learning_rate=self.learning_rate, weight_decay=self.weight_decay, features=False,
                                                                 pretrained_model_dir=self.pretrained_model_dir, pretrained_model_path=self.pretrained_model_path, mode="classification")
        self.novelty_detector.to(self._device)
        self.propensity_estimator, self.propensity_optimizer = get_model(arch, data_dir, self.dataset, self.num_outputs, pretrained= self.pretrained,
                                                                 learning_rate=self.learning_rate, weight_decay=self.weight_decay, features=False,
                                                                 pretrained_model_dir=self.pretrained_model_dir, pretrained_model_path=self.pretrained_model_path, mode="propensity_estimation")
        self.propensity_estimator.to(self._device)
        # dummy optimizer only for lightning module checkpointing
        _, self.dummy_optimizer = get_model(arch, data_dir, self.dataset, self.num_outputs, pretrained= self.pretrained,
                                                                 learning_rate=self.learning_rate, weight_decay=self.weight_decay, features=False,
                                                                 pretrained_model_dir=self.pretrained_model_dir, pretrained_model_path=self.pretrained_model_path, mode="classification")
        self.novelty_lr_scheduler = lr_scheduler.LinearLR(self.novelty_optimizer, start_factor=1.0, end_factor=1.0, total_iters=15000)
        self.propensity_lr_scheduler = lr_scheduler.LinearLR(self.propensity_optimizer, start_factor=1.0, end_factor=1.0, total_iters=15000)
        self.target_precision = target_precision
        self.precision_confidence = precision_confidence
        # self.target_recall = 0.02 # target_recall
        
        self.max_epochs = max_epochs
        self.inner_epochs = inner_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_patience = warmup_patience

        self.validation_step_outputs = []
        self.validation_step_outputs_s = []
        self.validation_step_outputs_t = []
        self.validation_step_outputs_discard = []
        self.val_features_s = torch.tensor([], device=device)
        self.val_features_t = torch.tensor([], device=device)

        self.novelty_learning_rate = learning_rate
        self.propensity_learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.refit = refit

        
        self.expected_prior_nonnovel = torch.tensor([], device=self._device)
        self.expected_propensity = torch.tensor([], device=self._device)
        self.expected_posterior_nonnovel = torch.tensor([], device=self._device)
        
        # Some variables for the alpha line search
        self.epoch = 0
        self.online_alpha_search = online_alpha_search
        self.alpha_search_midpoint = None
        self.epochs_since_alpha_update = 0.
        self.epochs_for_each_alpha = epochs_for_each_alpha
        self.pure_bin_estimate = [0.]*2
        self.pure_MPE_threshold = [0.]*2
        self.best_valid_supervised_loss, self.epoch_at_best_valid_supervised_loss = 1000., 0
        self.best_bin_size = [0.]*2
        self.best_candidate_alpha = [0.]*2
        self.best_valid_loss = [1000.]*2
        self.best_source_loss = [1000.]*2
        self.auc_roc_at_selection = [0.]*2
        self.ap_at_selection = [0.]*2
        self.precision_at_selection = [0.]*2
        self.recall_at_selection = [0.]*2
        self.f1_at_selection = [0.]*2
        self.acc_at_selection = [0.]*2
        self.recall_target_at_selection = [0.]*2
        self.fpr_at_selection = [1.]*2
        self.num_allowed_fp = -1
        self.alpha_checkpoints = [0.01, 0.1, 0.3, 0.6, 0.9]
        self.constraint_satisified = False
        self.lower_bound_alpha = (target_recall, 0.)
        self.cur_alpha_estimate = (target_recall, 0.)
        self.upper_bound_alpha = (None, 0.)
        self.bin_size_sensitivity = 0.05 #when gap between bin sizes is larger than this, we'll consider then significantly different
        # once constraint is approximately satisifed, allow 5 epochs to train with it, and then reexamine alpha

        self.pred_save_path = f"{pred_save_path}/{dataset}/"

        self.logging_file = f"{self.pred_save_path}/SAREM_{arch}_{num_source_classes}_{seed}_log_update.txt"
        
        self.model_path = save_model_path + self.dataset + "_" + "SAREM_seed_"+str(seed)+"_num_source_cls_"+str(num_source_classes)+"_ood_class_"+str(ood_class)+"_fraction_ood_class_"+str(fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"_use_labels_"+str(use_labels)+"_use_superclass_"+str(use_superclass)+"/" # "/cis/home/schaud35/shiftpu/models/imagenet_CoNoC_seed_"+str(seed)+"_ood_ratio_"+str(ood_class_ratio)+"/"
        # self.model_path = "/cis/home/schaud35/shiftpu/models/CoNoC_seed_"+str(seed)+"_ood_ratio_"+str(ood_class_ratio)+"/"

        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)

        if not os.path.exists(save_model_path + self.dataset + "_" + "SAREM_seed_"+str(seed)+"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"/" ):
            os.makedirs(save_model_path + self.dataset + "_" + "SAREM_seed_"+str(seed)+"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"/" )


        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.warm_start = False if self.warmup_epochs == 0 else True
        self.reload_model = False

        self.automatic_optimization = False

    def forward(self, model, x):
        return model(x)

    def expectation_nonnovel(self, expectation_nonnovel, expectation_propensity, s):
        # probability of data points being in non-novel class
        # if s = 1 (src data), must be in non-novel class
        result = s + (1 - s) * (expectation_nonnovel * (1 - expectation_propensity)) / (1 - expectation_nonnovel * expectation_propensity)
        return result

    def loglikelihood_probs(self, nonnovel_probs, propensity_scores, s):
        prob_src = nonnovel_probs * propensity_scores
        prob_tgt_nonnovel = nonnovel_probs * (1 - propensity_scores)
        prob_tgt_novel = 1 - nonnovel_probs
        prob_nonnovel_given_tgt = prob_tgt_nonnovel / (prob_tgt_nonnovel + prob_tgt_novel)
        prob_novel_given_tgt = 1 - prob_nonnovel_given_tgt
        return (s * torch.log(prob_src) + (1 - s) * (prob_nonnovel_given_tgt * torch.log(prob_tgt_nonnovel) + prob_novel_given_tgt * torch.log(prob_tgt_novel))).mean()

    def on_train_start(self):
        
        torch.manual_seed(self.seed)
        # initialize with unlabeled=negative, but reweighting the examples so that the expected class prior is 0.5
        train_loader = self.trainer.datamodule.train_dataloader()
        val_loader = self.trainer.datamodule.val_dataloader()[3]
        s = torch.tensor([], device=self._device)
        self.all_labels = torch.tensor([], device=self._device)
        self.s = torch.tensor([],device=self._device)
        torch.manual_seed(self.seed)
        
        for idx, batch in enumerate(train_loader):
            x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
            self.all_labels = torch.cat([self.all_labels, labels])
            self.s = torch.cat([self.s, torch.ones_like(y_s, device=self._device), torch.zeros_like(y_t, device=self._device)], dim=0)
        torch.manual_seed(self.seed)
        for idx, batch in enumerate(val_loader):
            x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
            self.all_labels = torch.cat([self.all_labels, labels])
            self.s = torch.cat([self.s, torch.ones_like(y_s, device=self._device), torch.zeros_like(y_t, device=self._device)], dim=0)
        proportion_src = self.s.sum() / self.s.size(0)
        detector_class_weights = torch.tensor([1 - proportion_src, proportion_src]).to(self._device)
        # for novelty_detector/propensity_estimator, output = 0 is non-novel/propensity=1, output = 1 is novel/propensity=0
        self.novelty_detector = self._inner_fit(self.novelty_detector, train_loader, val_loader, self.novelty_optimizer, self.inner_epochs, mode='classification', class_weight=detector_class_weights)
        seen_samples=0
        torch.manual_seed(self.seed)
        for idx, batch in enumerate(train_loader):
            x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
            s = torch.cat([torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
            y = 1 - s
            self.assert_reprocible(labels, s, seen_samples)
            self.expected_prior_nonnovel = torch.cat([self.expected_prior_nonnovel, F.softmax(self.forward(self.novelty_detector, x), dim=1)[:,0].detach()], dim=0) # prob of being non-novel (positive in PU terms)
            seen_samples = seen_samples+len(labels)
        torch.manual_seed(self.seed)
        for idx, batch in enumerate(val_loader):
            x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
            s = torch.cat([torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
            y = 1 - s
            self.assert_reprocible(labels, s, seen_samples)
            self.expected_prior_nonnovel = torch.cat([self.expected_prior_nonnovel, F.softmax(self.forward(self.novelty_detector, x), dim=1)[:,0].detach()], dim=0)
            seen_samples = seen_samples+len(labels)

        propensity_sample_weights = self.s + (1 - self.s) * self.expected_prior_nonnovel
        self.propensity_estimator = self._inner_fit(self.propensity_estimator, train_loader, val_loader, self.propensity_optimizer, self.inner_epochs, mode='propensity_estimation', sample_weight=propensity_sample_weights)
        seen_samples=0
        torch.manual_seed(self.seed)
        for idx, batch in enumerate(train_loader):
            x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
            s = torch.cat([torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
            y = 1 - s
            self.assert_reprocible(labels, s, seen_samples)
            self.expected_propensity = torch.cat([self.expected_propensity, F.softmax(self.forward(self.propensity_estimator, x), dim=1)[:,0].detach()], dim=0)
            seen_samples = seen_samples+len(labels)
        torch.manual_seed(self.seed)
        for idx, batch in enumerate(val_loader):
            x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
            s = torch.cat([torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
            y = 1 - s
            self.assert_reprocible(labels, s, seen_samples)
            self.expected_propensity = torch.cat([self.expected_propensity, F.softmax(self.forward(self.propensity_estimator, x), dim=1)[:,0].detach()], dim=0)
            seen_samples = seen_samples+len(labels)

        self.expected_posterior_nonnovel = self.expectation_nonnovel(self.expected_prior_nonnovel, self.expected_propensity, self.s)

    def reset_expectation(self):
        self.expected_prior_nonnovel = torch.tensor([], device=self._device)
        self.expected_propensity = torch.tensor([], device=self._device)
        self.expected_posterior_nonnovel = torch.tensor([], device=self._device)
        return
    
    def on_train_epoch_start(self):
        
        self.epoch = self.epoch + 1
        train_loader = self.trainer.datamodule.train_dataloader()
        val_loader = self.trainer.datamodule.val_dataloader()[3]
        
        # for novelty_detector/propensity_estimator, output = 0 is non-novel/propensity=1, output = 1 is novel/propensity=0
        propensity_estimator, propensity_optimizer = get_model(self.arch, self.data_dir, self.dataset, self.num_outputs, pretrained=False,
                                                            learning_rate=self.propensity_learning_rate, weight_decay=self.weight_decay, features=False, mode="propensity_estimation")
        propensity_estimator.to(self._device)
        self.propensity_estimator = self._inner_fit(propensity_estimator, train_loader, val_loader, propensity_optimizer, self.inner_epochs, mode='propensity_estimation', sample_weight=self.expected_posterior_nonnovel, clip=self.clip)
        # classification_s = torch.cat([torch.ones_like(self.expected_posterior_nonnovel, dtype=torch.int64), torch.zeros_like(self.expected_posterior_nonnovel, dtype=torch.int64)], dim=0)
        # classification_weights = torch.cat([self.expected_posterior_nonnovel, 1 - self.expected_posterior_nonnovel], dim=0)
        
        novelty_detector, novelty_optimizer = get_model(self.arch, self.data_dir, self.dataset, self.num_outputs, pretrained=False,
                                                            learning_rate=self.novelty_learning_rate, weight_decay=self.weight_decay, features=False, mode="classification")
        novelty_detector.to(self._device)
        # target of 1st half of the data is 0 (non-novel)
        self.novelty_detector = self._inner_fit(novelty_detector, train_loader, val_loader, novelty_optimizer, self.inner_epochs, mode='classification', double_pass=True, sample_weight=self.expected_posterior_nonnovel, clip=self.clip)
        nll_total = -self.loglikelihood_probs(self.expected_prior_nonnovel, self.expected_propensity, self.s)
        print('epoch: ', self.epoch, 'before reset nll total:', nll_total)
        # expectation
        self.reset_expectation()
        nll_train, _, _, _ = self.evaluate_nll_probs(train_loader)
        nll_val, _, _, _ = self.evaluate_nll_probs(val_loader)
        print('epoch:', self.epoch,'after reset nll_train:', nll_train, 'nll_val:', nll_val, 'nll_total:', 5*nll_train/6 + nll_val/6)
        seen_samples = 0
        all_s = torch.tensor([], device=self._device)
        torch.manual_seed(self.seed)
        for idx, batch in enumerate(train_loader):
            x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
            s = torch.cat([torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
            all_s = torch.cat([all_s, s], dim=0)
            y = 1 - s
            self.assert_reprocible(labels, s, seen_samples)
            self.expected_prior_nonnovel = torch.cat([self.expected_prior_nonnovel, F.softmax(self.forward(self.novelty_detector, x), dim=1)[:,0].detach()], dim=0)
            self.expected_propensity = torch.cat([self.expected_propensity, F.softmax(self.forward(self.propensity_estimator, x), dim=1)[:,0].detach()], dim=0)
            self.expected_posterior_nonnovel = torch.cat([self.expected_posterior_nonnovel, self.expectation_nonnovel(self.expected_prior_nonnovel[-len(labels):], self.expected_propensity[-len(labels):], s)], dim=0)
            seen_samples = seen_samples+len(labels)
        torch.manual_seed(self.seed)
        nll_train = -self.loglikelihood_probs(self.expected_prior_nonnovel, self.expected_propensity, all_s)
        print('epoch: ', self.epoch, 'after recalibration nll_train:', nll_train)
        for idx, batch in enumerate(val_loader):
            x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
            s = torch.cat([torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
            all_s = torch.cat([all_s, s], dim=0)
            y = 1 - s
            self.assert_reprocible(labels, s, seen_samples)
            self.expected_prior_nonnovel = torch.cat([self.expected_prior_nonnovel, F.softmax(self.forward(self.novelty_detector, x), dim=1)[:,0].detach()], dim=0)
            self.expected_propensity = torch.cat([self.expected_propensity, F.softmax(self.forward(self.propensity_estimator, x), dim=1)[:,0].detach()], dim=0)
            self.expected_posterior_nonnovel = torch.cat([self.expected_posterior_nonnovel, self.expectation_nonnovel(self.expected_prior_nonnovel[-len(labels):], self.expected_propensity[-len(labels):], s)], dim=0)
            seen_samples = seen_samples+len(labels)
        
        nll_total = -self.loglikelihood_probs(self.expected_prior_nonnovel, self.expected_propensity, all_s)
        print('epoch: ', self.epoch, 'after recalibration nll_total:', nll_total)
        return
    
    def assert_reprocible(self, labels, s, seen_samples):
        assert len(labels)==len(s), "Length of labels, (" + str(len(labels)) + ") and s, (" + str(len(s)) + ") should be same"
        assert sum(labels==self.all_labels[seen_samples:seen_samples+len(labels)])==len(labels)
        assert sum(s==self.s[seen_samples:seen_samples+len(s)])==len(s)

    def _inner_fit(self, model, train_loader, val_loader, optimizer, inner_epochs, mode='classification', double_pass: bool=False, patience=5, class_weight=None, sample_weight=None, clip=5.0):
        import wandb
        best_val_loss = np.inf
        staleness = 0
        run = wandb.init()
        # define our custom x axis metric
        wandb.define_metric("inner_epoch")
        
        for e in tqdm(range(1, inner_epochs + 1)):
            model.train()
            seen_samples = 0
            torch.manual_seed(self.seed)
            for idx, batch in enumerate(train_loader):
                x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
                s = torch.cat([torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
                y = 1 - s
                self.assert_reprocible(labels, s, seen_samples)
                if self.use_superclass & (self.dataset in ["cifar100","newsgroups20","amazon_reviews"]) :
                    labels = labels//5 if self.dataset=="cifar100" else labels//5
    
                curr_sample_weight = sample_weight[seen_samples:seen_samples+len(labels)] if sample_weight is not None else None
                if double_pass and mode=='classification':
                    x = torch.cat([x, x], dim=0)
                    y = torch.cat([torch.ones_like(y), torch.zeros_like(y)], dim=0)
                    curr_sample_weight = torch.cat([curr_sample_weight, 1-curr_sample_weight], dim=0) if sample_weight is not None else None
                train_out = self.forward(model, x)
                train_loss = self.evaluate_nll_loss(train_out, y, class_weight, curr_sample_weight)
                optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                seen_samples = seen_samples + len(labels)
            train_nll_loss, logits, probs, train_labels = self.evaluate_nll_loss(class_weight=class_weight, sample_weight=sample_weight, loader=train_loader, model=model, double_pass=double_pass)  
            wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/train_nll_loss": train_nll_loss, "inner_epoch": e})
            # train_roc, train_ap, train_f1 = self.calculate_metrics(probs, train_labels)
            # wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/train_roc": train_roc, "inner_epoch": e})
            # wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/train_ap": train_ap, "inner_epoch": e})
            # wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/train_f1": train_f1, "inner_epoch": e})
            model.eval()
            with torch.no_grad():
                torch.manual_seed(self.seed)
                for idx, batch in enumerate(val_loader):
                    x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
                    s = torch.cat([torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
                    y = 1 - s
                    self.assert_reprocible(labels, s, seen_samples)
                    if self.use_superclass & (self.dataset in ["cifar100","newsgroups20","amazon_reviews"]) :
                        labels = labels//5 if self.dataset=="cifar100" else labels//5
                    
                    curr_sample_weight = sample_weight[seen_samples:seen_samples+len(labels)] if sample_weight is not None else None
                    if double_pass:
                        x = torch.cat([x, x], dim=0)
                        y = torch.cat([torch.ones_like(y), torch.zeros_like(y)], dim=0)
                        curr_sample_weight = torch.cat([curr_sample_weight, 1-curr_sample_weight], dim=0) if sample_weight is not None else None
                    val_out = self.forward(model, x)
                    
                    # result = evaluate_classification(y.detach().cpu().numpy(), softmax(val_out, dim=-1)[:,1].detach().cpu().numpy())
                    # print(x.shape, y.shape, {k:result[k] for k in ['roc_auc', 'average_precision', 'f1', 'accuarcy', 'precision']})
                    val_loss = self.evaluate_nll_loss(val_out, y, class_weight, curr_sample_weight)
                
                    seen_samples = seen_samples + len(labels)
            
            val_nll_loss, logits, probs, val_labels = self.evaluate_nll_loss(class_weight=class_weight, sample_weight=sample_weight, loader=val_loader, model=model, double_pass=double_pass)  
            if len(probs)>len(val_labels):
                val_labels = torch.cat([val_labels, val_labels],dim=0)
            wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/val_nll_loss": val_nll_loss, "inner_epoch": e})
            roc, ap, f1 = self.calculate_metrics(probs, val_labels)
            wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/val_roc_auc": roc, "inner_epoch": e})
            wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/val_average_precision": ap, "inner_epoch": e})
            wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/val_f1": f1, "inner_epoch": e})
            if val_nll_loss < best_val_loss:
                best_model = deepcopy(model)
                best_val_loss = val_nll_loss
                staleness = 0
            else:
                staleness += 1
            
            
            wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/best_val_nll_loss": best_val_loss, "inner_epoch": e})

            if staleness > patience:
                break
            
            # val_roc, val_ap, val_f1 = self.calculate_metrics(probs, val_labels)
            # wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/val_roc": val_roc, "inner_epoch": e})
            # wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/val_ap": val_ap, "inner_epoch": e})
            # wandb.log({"inner_"+str(self.epoch)+"/"+mode+"/val_f1": val_f1, "inner_epoch": e})
            
            if staleness > patience:
                break
        print("val nll loss:", val_nll_loss, "train nll loss:", train_nll_loss)
        return best_model
    
    def configure_optimizers(self):
        return self.dummy_optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["expected_prior_nonnovel"] = self.expected_prior_nonnovel
        checkpoint["expected_propensity"] = self.expected_propensity
        checkpoint["expected_posterior_nonnovel"] = self.expected_posterior_nonnovel

    def on_load_checkpoint(self, checkpoint):
        self.expected_prior_nonnovel = checkpoint["expected_prior_nonnovel"]
        self.expected_propensity = checkpoint["expected_propensity"]
        self.expected_posterior_nonnovel = checkpoint["expected_posterior_nonnovel"]

    def get_data(self, batch):
        if len(batch["source_full"][:3])>2:
            x_s, y_s, _ = batch["source_full"][:3]
            x_t, y_t, _ = batch["target_full"][:3]
        elif len(batch["source_full"])==2:
            x_s, y_s = batch["source_full"]
            x_t, y_t = batch["target_full"]
        
        x_s = x_s.to(self._device)
        y_s = y_s.to(self._device)
        x_t = x_t.to(self._device)
        y_t = y_t.to(self._device)
        
        if self.use_superclass & (self.dataset in ["cifar100", "newsgroupd20","amazon_reviews"]):
            y_s = y_s//5 if self.dataset=="cifar100" else y_s//5
            y_t = y_t//5 if self.dataset=="cifar100" else y_t//5

        if torch.is_tensor(x_s) and torch.is_tensor(x_t):
            x = torch.cat([x_s, x_t], dim=0)
        elif isinstance(x_s, list) and isinstance(x_t, list):
            x = x_s.copy()
            x.extend(x_t)
        elif isinstance(x_s, dict) and isinstance(x_t, dict):
            x = {}
            for k in x_s.keys():
                x[k] = torch.cat([x_s[k], x_t[k]], dim=0)
        else:
            raise Exception("Not valid data type of x_s", type(x_s),"or x_t",type(x_t))
        y = torch.cat([y_s, y_t], dim=0)
        
        return x, y, x_s, y_s, x_t, y_t
    
    def calculate_metrics(self, probs, labels):
        probs = probs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        y_oracle = np.zeros_like(labels)
        novel_inds = np.where(labels == self.num_classes)[0]
        y_oracle[novel_inds] = 1

        roc_auc = roc_auc_score(y_oracle, probs[:, 1])
        ap = average_precision_score(y_oracle, probs[:, 1])
        f1 = f1_score(y_oracle, np.argmax(probs, axis=1))
        return roc_auc, ap, f1
    
    def evaluate_nll_loss(self, logits=None, y=None, class_weight=None, sample_weight=None, loader=None, model=None, double_pass=False, mode="classification"):
        if ((loader is None) or (model is None)) and (logits is not None) and (y is not None):
            if sample_weight is not None:
                nll_loss = F.nll_loss(F.log_softmax(logits, dim=1), y, weight=class_weight, reduction="none")
                nll_loss = torch.mean(nll_loss * sample_weight)
            else:
                nll_loss = F.nll_loss(F.log_softmax(logits, dim=1), y, weight=class_weight, reduction='mean')
            return nll_loss
        else:
            model.eval()
            with torch.no_grad():
                all_logits = torch.tensor([], device=self._device)
                all_labels = torch.tensor([], device=self._device)
                all_y = torch.tensor([], device=self._device, dtype=torch.int64)
                all_sample_weight = torch.tensor([], device=self._device)
                seen_samples = 0
                torch.manual_seed(self.seed)
                for idx, batch in enumerate(loader):
                    x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
                    all_labels = torch.cat([all_labels, labels], dim=0)
                    s = torch.cat([torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
                    y = 1 - s
                    
                    curr_sample_weight = sample_weight[seen_samples:seen_samples+len(labels)] if sample_weight is not None else None
                    if double_pass:
                        x = torch.cat([x, x], dim=0)
                        y = torch.cat([torch.ones_like(1 - s), torch.zeros_like(1 - s)], dim=0)
                        curr_sample_weight = torch.cat([curr_sample_weight, 1-curr_sample_weight], dim=0) if sample_weight is not None else None
                    all_sample_weight = torch.cat([all_sample_weight, curr_sample_weight], dim=0) if sample_weight is not None else None
                    all_y = torch.cat([all_y, y], dim=0)
                    logits = self.forward(model, x)
                    all_logits = torch.cat([all_logits, logits], dim=0)
                    seen_samples = seen_samples + len(labels)
                if sample_weight is not None:
                    nll_loss = F.nll_loss(F.log_softmax(all_logits, dim=1), all_y, weight=class_weight, reduction="none")
                    nll_loss = torch.mean(nll_loss * all_sample_weight)
                else:
                    nll_loss = F.nll_loss(F.log_softmax(all_logits, dim=1), all_y, weight=class_weight, reduction='mean')
            return nll_loss, logits, softmax(logits, dim=-1), all_labels


    def evaluate_nll_probs(self, loader, novelty_model=None, propensity_model=None):
        if novelty_model is None:
            novelty_model = self.novelty_detector
        if propensity_model is None:
            propensity_model = self.propensity_estimator
        novelty_model.eval()
        propensity_model.eval()

        expected_prior_nonnovel = torch.tensor([], device=self._device)
        expected_propensity = torch.tensor([], device=self._device)
        expected_posterior_nonnovel = torch.tensor([], device=self._device)
        all_labels = torch.tensor([], device=self._device)
        logits = torch.tensor([], device=self._device)
        probs = torch.tensor([], device=self._device)
        s = torch.tensor([], device=self._device)

        with torch.no_grad():
            torch.manual_seed(self.seed)
            for idx, batch in enumerate(loader):
                x, labels, x_s, y_s, x_t, y_t = self.get_data(batch)
                all_labels = torch.cat([all_labels, labels], dim=0)
                s = torch.cat([s, torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
                expected_prior_nonnovel = torch.cat([expected_prior_nonnovel, F.softmax(self.forward(novelty_model, x), dim=1)[:,0].detach()],dim=0)
                expected_propensity = torch.cat([expected_propensity, F.softmax(self.forward(propensity_model, x), dim=1)[:,0].detach()],dim=0)
                expected_posterior_nonnovel = torch.cat([expected_posterior_nonnovel, self.expectation_nonnovel(expected_prior_nonnovel, expected_propensity, s)],dim=0)
                batch_logits =  self.forward(novelty_model, x)
                logits = torch.cat([logits, batch_logits], dim=0)
                probs = torch.cat([probs, softmax(batch_logits, dim=-1)], dim=0)

        nll = -self.loglikelihood_probs(expected_prior_nonnovel, expected_propensity, s)
        
        return nll, logits, probs, all_labels

    def training_step(self, batch, batch_idx: int):
        # nll_loss = self.process_batch(batch, "train")
        # self.log("train/nll_loss", nll_loss, on_step=True, on_epoch=True, prog_bar=False)
        
        # self.log("train/loss.supervised", supervised_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        return  # {"nll_loss": nll_loss.detach()}
    
    def on_train_epoch_end(self):
        
        train_loader = self.trainer.datamodule.train_dataloader()
        val_loader = self.trainer.datamodule.val_dataloader()[3]
        nll_train_loss_novelty, _, _, _ = self.evaluate_nll_loss(class_weight=None, sample_weight=self.expected_posterior_nonnovel, loader=train_loader, model=self.novelty_detector, double_pass=True)
        nll_train_loss_propensity, _, _, _ = self.evaluate_nll_loss(class_weight=None, sample_weight=self.expected_posterior_nonnovel, loader=train_loader, model=self.propensity_estimator, double_pass=False)
        self.log("train/nll_loss_novelty", nll_train_loss_novelty, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/nll_loss_propensity", nll_train_loss_propensity, on_step=False, on_epoch=True, prog_bar=False)
        nll_train, logits, probs, train_labels = self.evaluate_nll_probs(train_loader)
        self.log("train/nll_probs", nll_train, on_step=False, on_epoch=True, prog_bar=False)
        # nll_loss_train, _, _, _ = self.evaluate_nll_loss(logits=logits, y=1-self.s, sample_weight=self.expected_posterior_nonnovel)
        # self.log("train/nll_loss", nll_loss_train, on_step=False, on_epoch=True, prog_bar=False)
        nll_total = -self.loglikelihood_probs(self.expected_prior_nonnovel, self.expected_propensity, self.s)
        self.log("total/nll_probs", nll_total, on_step=False, on_epoch=True, prog_bar=False)
        train_labels = train_labels.detach().cpu().numpy()

        y_oracle = np.zeros_like(train_labels)
        novel_inds = np.where(train_labels == self.num_classes)[0]
        y_oracle[novel_inds] = 1

        roc_auc = roc_auc_score(y_oracle, probs[:, 1].detach().cpu().numpy())
        ap = average_precision_score(y_oracle, probs[:, 1].detach().cpu().numpy())
        f1 = f1_score(y_oracle, np.argmax(probs.detach().cpu().numpy(), axis=1))
        self.log("train/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)
        self.log("train/performance.AP", ap, on_step=False, on_epoch=True)
        self.log("train/performance.F1", f1, on_step=False, on_epoch=True)
        

        total_norm = [0,0]
        for p in self.novelty_detector.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm[0] += param_norm.item() ** 2
        for p in self.propensity_estimator.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm[1] += param_norm.item() ** 2
        total_norm = [i ** (1. / 2) for i in total_norm]
        log.info('novelty & propensity grad norms after training {}'.format(total_norm))
        return
    def on_train_end(self):
        train_loader = self.trainer.datamodule.train_dataloader()
        val_loader = self.trainer.datamodule.val_dataloader()[3]
        if self.refit:
            for l in self.all_labels.unique():
                print("Average propensity in class ", str(l), self.expected_propensity[self.all_labels == l].mean())
            
            novelty_detector, novelty_optimizer = get_model(self.arch, self.data_dir, self.dataset, self.num_outputs, pretrained=False,
                                                                learning_rate=self.novelty_learning_rate, weight_decay=self.weight_decay, features=False, mode="classification")
            novelty_detector.to(self._device)
            weights_nonnovel = self.s / (self.expected_propensity + 1e-12)
            weights_novel = (1 - self.s) + self.s * (1 - 1 / (self.expected_propensity + 1e-12))

            sample_weights = weights_nonnovel
            self.novelty_detector = self._inner_fit(novelty_detector, train_loader, val_loader, novelty_optimizer, self.inner_epochs, mode='classification', double_pass=True, sample_weight=sample_weights)
            
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # if dataloader_idx == 3:
        #     nll, logits, probs, y, labels = self.process_batch(batch, "pred_source")
        #     outputs = {"probs": probs, "y": y, "logits": logits, "nll":nll, "labels":labels}
        #     self.validation_step_outputs.append(outputs)
        #     return outputs
        return

    def on_validation_epoch_end(self):
        

        val_loader = self.trainer.datamodule.val_dataloader()[3]
        nll, logits, probs, val_labels  = self.evaluate_nll_probs(val_loader)
        val_labels = val_labels.detach().cpu().numpy()
        
        # results_train = evaluate_all(y_train,s_train,e_train, f_model.predict_proba(x_train),e_model.predict_proba(x_train))
        
        # supervised_loss = cross_entropy(torch.tensor(disc_class_logits_s), torch.tensor(y_s))
        # self.log("pred/supervised_loss", supervised_loss)
        # import pdb; pdb.set_trace()
        
        # disc_ce_loss = cross_entropy(torch.cat((self.val_features_s, self.val_features_t),dim=0).cpu().detach(), torch.tensor(y))

        y_oracle = np.zeros_like(val_labels)
        novel_inds = np.where(val_labels == self.num_classes)[0]
        y_oracle[novel_inds] = 1
        novel_ce_loss = cross_entropy(logits, torch.tensor(y_oracle, dtype=torch.int64, device=self._device))
        results_test = evaluate_classification(y_oracle, probs[:,1].detach().cpu().numpy())

        

        roc_auc = roc_auc_score(y_oracle, probs[:, 1].detach().cpu().numpy())
        ap = average_precision_score(y_oracle, probs[:, 1].detach().cpu().numpy())
        f1 = f1_score(y_oracle, np.argmax(probs.detach().cpu().numpy(), axis=1))
        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)
        self.log("val/performance.AP", ap, on_step=False, on_epoch=True)
        self.log("val/performance.F1", f1, on_step=False, on_epoch=True)
        self.log("val/nll_probs", nll, on_step=False, on_epoch=True)
        self.validation_step_outputs = []

        # features = torch.cat((self.val_features_s, self.val_features_t),dim=0).cpu().detach().numpy()
        # y_t_plot = np.ones_like(y_t)
        # y_t_plot[novel_inds] = 2
        # gt = np.concatenate((y_s_oracle, y_t_plot), axis=0)
        
        
        # results_tsne_2d = compute_tsne(features, n_components=2, n_iter=5000)
        # results_tsne_3d = compute_tsne(features, n_components=3, n_iter=5000)
        # results_pca_2d = compute_PCA(features, n_components=2)
        # results_pca_3d = compute_PCA(features, n_components=3)
        # plt_2d_scatterplot(results_tsne_2d, gt, num_classes=3, save_plt_path='./tsne_2d.png')
        # plt_3d_scatterplot(results_tsne_3d, gt, reduction_algo='tsne', save_plt_path='./tsne_3d.png')
        # plt_2d_scatterplot(results_pca_2d, gt, num_classes=3, save_plt_path='./pca_2d.png')
        # plt_3d_scatterplot(results_pca_3d, gt, reduction_algo='pca', save_plt_path='./pca_3d.png')
        # import pdb; pdb.set_trace()
         
        total_norm = [0,0]
        for p in self.novelty_detector.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm[0] += param_norm.item() ** 2
        for p in self.propensity_estimator.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm[1] += param_norm.item() ** 2
        total_norm = [i ** (1. / 2) for i in total_norm]
        
    def dataselector(self, unlabeled_data, budget_per_AL_cycle, batch_size=200):
        self.novelty_detector.load_state_dict(self.novelty_detector.state_dict(), self.model_path + "novelty_detector_model.pth")
        unlabeled_dataloder = DataLoader( unlabeled_data, batch_size=batch_size, shuffle=False, \
            num_workers=8,  pin_memory=True)
        all_probs = torch.tensor([], device=self._device)
        for batch in tqdm(unlabeled_dataloder):
            all_probs = torch.cat((all_probs, softmax(self.novelty_detector(batch[0]), dim=-1)),dim=0)
        
        
        return
        

    def configure_optimizers(self):
        return self.dummy_optimizer