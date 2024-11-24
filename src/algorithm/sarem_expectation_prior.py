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
        inner_epochs: int = 100,
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
                                                                 pretrained_model_dir=self.pretrained_model_dir, pretrained_model_path=self.pretrained_model_path)
        self.novelty_detector.to(self._device)
        self.propensity_estimator, self.propensity_optimizer = get_model(arch, data_dir, self.dataset, self.num_outputs, pretrained= self.pretrained,
                                                                 learning_rate=self.learning_rate, weight_decay=self.weight_decay, features=False,
                                                                 pretrained_model_dir=self.pretrained_model_dir, pretrained_model_path=self.pretrained_model_path)
        self.propensity_estimator.to(self._device)
        # dummy optimizer only for lightning module checkpointing
        _, self.dummy_optimizer = get_model(arch, data_dir, self.dataset, self.num_outputs, pretrained= self.pretrained,
                                                                 learning_rate=self.learning_rate, weight_decay=self.weight_decay, features=False,
                                                                 pretrained_model_dir=self.pretrained_model_dir, pretrained_model_path=self.pretrained_model_path)
        self.novelty_lr_scheduler = lr_scheduler.LinearLR(self.novelty_optimizer, start_factor=1.0, end_factor=1.0, total_iters=15000)
        self.propensity_lr_scheduler = lr_scheduler.LinearLR(self.propensity_optimizer, start_factor=1.0, end_factor=1.0, total_iters=15000)
        self.target_precision = target_precision
        self.precision_confidence = precision_confidence
        # self.target_recall = 0.02 # target_recall
        
        

        self.max_epochs = max_epochs
        self.inner_epochs = inner_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_patience = warmup_patience

        self.validation_step_outputs_s = []
        self.validation_step_outputs_t = []
        self.validation_step_outputs_discard = []
        self.val_features_s = torch.tensor([], device=device)
        self.val_features_t = torch.tensor([], device=device)

        self.novelty_learning_rate = learning_rate
        self.propensity_learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.refit = refit

        # Some variables for the alpha line search
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
        # initialize with unlabeled=negative, but reweighting the examples so that the expected class prior is 0.5
        datamodule = self.trainer.datamodule
        loader = datamodule.train_dataloader()
        data = next(iter(loader))
        data = {k: [data[k][i].to(self._device) for i in range(len(data[k]))] for k in data.keys()}
        
        # get first 80% of the data for training, rest for validation
        train_sz_s, train_sz_t = int(data['source_full'][1].shape[0]*0.8), int(data['target_full'][1].shape[0]*0.8)
        train_mask = torch.cat([torch.ones_like(data['source_full'][1][:train_sz_s]), torch.zeros_like(data['source_full'][1][train_sz_s:]), torch.ones_like(data['target_full'][1][:train_sz_t]), torch.zeros_like(data['target_full'][1][train_sz_t:])], dim=0)
        
        x = torch.cat([data['source_full'][0], data['target_full'][0]], dim=0)
        s = torch.cat([torch.ones_like(data['source_full'][1]), torch.zeros_like(data['target_full'][1])], dim=0)
        proportion_src = s.sum() / s.size(0)
        detector_class_weights = torch.tensor([1 - proportion_src, proportion_src]).to(self._device)
        
       
        # for novelty_detector/propensity_estimator, output = 0 is non-novel/propensity=1, output = 1 is novel/propensity=0
        self.novelty_detector = self._inner_fit(self.novelty_detector, x, 1 - s, train_mask, self.novelty_optimizer, self.inner_epochs, class_weight=detector_class_weights)
        detector_expectation = F.softmax(self.forward(self.novelty_detector, x), dim=1)[:,0].detach() # prob of being non-novel (positive in PU terms)

        # propensity_estimator, propensity_optimizer = get_model_optimizer(self.model_type,
        #                                                                  self.arch_param,
        #                                                                  self.learning_rate,
        #                                                                  self.weight_decay)
        # propensity_estimator.to(self._device)
        propensity_sample_weights = s + (1 - s) * detector_expectation
        self.propensity_estimator = self._inner_fit(self.propensity_estimator, x, 1 - s, train_mask, self.propensity_optimizer, self.inner_epochs, sample_weight=propensity_sample_weights)
        
        self.expected_prior_nonnovel = detector_expectation
        self.expected_propensity = F.softmax(self.forward(self.propensity_estimator, x), dim=1)[:,0].detach()
        self.expected_posterior_nonnovel = self.expectation_nonnovel(self.expected_prior_nonnovel, self.expected_propensity, s)

    def _inner_fit(self, model, data, y, train_mask, optimizer, inner_epochs, patience=20, class_weight=None, sample_weight=None, clip=5.0):
        best_val_loss = np.inf
        staleness = 0
        
        for e in tqdm(range(1, inner_epochs + 1)):
            train_out = self.forward(model, data[train_mask==1])
            if sample_weight is None:
                train_loss = F.nll_loss(F.log_softmax(train_out, dim=1), y[train_mask==1], weight=class_weight)
            else:
                train_loss = F.nll_loss(F.log_softmax(train_out, dim=1), y[train_mask==1], weight=class_weight, reduction="none")
                train_loss = torch.mean(train_loss * sample_weight[train_mask==1])
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            val_out = self.forward(model, data[train_mask==0])
            if sample_weight is None:
                val_loss = F.nll_loss(F.log_softmax(val_out, dim=1), y[train_mask==0], weight=class_weight)
            else:
                val_loss = F.nll_loss(F.log_softmax(val_out, dim=1), y[train_mask==0], weight=class_weight, reduction="none")
                val_loss = torch.mean(val_loss * sample_weight[train_mask==0])
            if val_loss < best_val_loss:
                best_model = deepcopy(model)
                best_val_loss = val_loss
                staleness = 0
            else:
                staleness += 1

            if staleness > patience:
                break
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

    def process_batch(self, batch, stage="train"):
        # import pdb; pdb.set_trace()
        if self.current_epoch>=self.warmup_epochs:
            self.warm_start=False
        if stage == "train":
            # import pdb; pdb.set_trace()
            if len(batch["source_full"][:3])>2:
                x_s, y_s, _ = batch["source_full"][:3]
                x_t, y_t, _ = batch["target_full"][:3]
            elif len(batch["source_full"])==2:
                x_s, y_s = batch["source_full"]
                x_t, y_t = batch["target_full"]
            
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
            if len(y_s)!=len(y_t):
                return
            # get first 80% of the data for training, rest for validation
            train_mask = torch.cat([torch.ones_like(y_s[:int(len(y_s)*0.8)]), torch.zeros_like(y_s[int(len(y_s)*0.8):]), torch.ones_like(y_t[:int(len(y_t)*0.8)]), torch.zeros_like(y_t[int(len(y_t)*0.8):])], dim=0)
            s = torch.cat([torch.ones_like(y_s), torch.zeros_like(y_t)], dim=0)
            y = 1 - s
            labels = torch.cat([y_s, y_t], dim=0)

            # y_s_oracle = torch.zeros_like(y_s, device=self._device)
            # novel_inds = torch.where(y_t == self.num_classes)[0]
            # y_t_oracle = torch.zeros_like(y_t, device=self._device)
            # y_t_oracle[novel_inds] = 1
            # y = torch.cat([y_s_oracle, y_t_oracle], dim=0)

            propensity_estimator, propensity_optimizer = get_model(self.arch, self.data_dir, self.dataset, self.num_outputs, pretrained=False,
                                                                learning_rate=self.propensity_learning_rate, weight_decay=self.weight_decay, features=False)
            propensity_estimator.to(self._device)
            self.propensity_estimator = self._inner_fit(propensity_estimator, x, 1 - s, train_mask, propensity_optimizer, self.inner_epochs, sample_weight=self.expected_posterior_nonnovel, clip=self.clip)
            classification_s = torch.cat([torch.ones_like(self.expected_posterior_nonnovel, dtype=torch.int64), torch.zeros_like(self.expected_posterior_nonnovel, dtype=torch.int64)], dim=0)
            classification_weights = torch.cat([self.expected_posterior_nonnovel, 1 - self.expected_posterior_nonnovel], dim=0)
            
            novelty_detector, novelty_optimizer = get_model(self.arch, self.data_dir, self.dataset, self.num_outputs, pretrained=False,
                                                                learning_rate=self.novelty_learning_rate, weight_decay=self.weight_decay, features=False)
            novelty_detector.to(self._device)
            # target of 1st half of the data is 0 (non-novel)
            self.novelty_detector = self._inner_fit(novelty_detector, torch.cat([x,x],dim=0), 1 - classification_s, torch.cat([train_mask, train_mask],dim=0), novelty_optimizer, self.inner_epochs, sample_weight=classification_weights, clip=self.clip)
            # expectation
            self.expected_prior_nonnovel = F.softmax(self.forward(self.novelty_detector, x), dim=1)[:,0].detach()
            self.expected_propensity = F.softmax(self.forward(self.propensity_estimator, x), dim=1)[:,0].detach()
            self.expected_posterior_nonnovel = self.expectation_nonnovel(self.expected_prior_nonnovel, self.expected_propensity, s)

            ll = self.loglikelihood_probs(self.expected_prior_nonnovel, self.expected_propensity, s)
            dummy_optimizer = self.optimizers()
            dummy_optimizer.zero_grad(); dummy_optimizer.step() # a dummy call for checkpointing
            return -ll            
            # return self.cmp.state.loss, self.cmp.get_penalty(self.novelty_detector), self.cmp.state.ineq_defect, lagrangian, torch.tensor(0.), self.cmp.state.misc['supervised_loss'], self.cmp.state.misc['cross_ent']

        elif stage == "pred_source":
            # import pdb; pdb.set_trace()
            if len(batch)>2:
                x_s, y_s, _ = batch[:3]
            elif len(batch)==2:
                x_s, y_s = batch

            if self.use_superclass & (self.dataset in ["cifar100","newsgroups20","amazon_reviews"]) :
                y_s = y_s//5 if self.dataset=="cifar100" else y_s//5

            ll = self.loglikelihood_probs(self.expected_prior_nonnovel, self.expected_propensity, s)
            novelty_logits_s = self.forward(self.novelty_detector, x_s)
            # self.val_features_s = torch.cat((self.val_features_s,logits), dim=0) 
            novelty_probs_s = softmax(novelty_logits_s, dim=-1)
#             disc_probs_s = probs
#
#             logits_s = self.source_model(x_s)
#             probs_s = softmax(logits_s, dim=1)
            return -ll, novelty_logits_s, novelty_probs_s, y_s

        elif stage == "pred_target":
            # import pdb; pdb.set_trace()
            if len(batch)>2:
                x_t, y_t, _ = batch[:3]
            elif len(batch)==2:
                x_t, y_t = batch

            if self.use_superclass & (self.dataset in ["cifar100","newsgroups20"]) :
                y_t = y_t//5 if self.dataset=="cifar100" else y_t//5

            ll = self.loglikelihood_probs(self.expected_prior_nonnovel, self.expected_propensity, s)
            novelty_logits_t = self.novelty_detector(x_t)
            # self.val_features_t = torch.cat((self.val_features_t,logits), dim=0) 
            novelty_probs_t = softmax(novelty_logits_t, dim=-1)
#             disc_probs_t = probs
#
#             logits_t = self.source_model(x_t)
#             probs_t = softmax(logits_t, dim=1)
            return -ll, novelty_logits_t, novelty_probs_t, y_t

        else:
            raise ValueError("Invalid stage %s" % stage)

    def training_step(self, batch, batch_idx: int):
        nll_loss = self.process_batch(batch, "train")
        self.log("train/nll_loss", nll_loss, on_step=True, on_epoch=True, prog_bar=False)
        
        # self.log("train/loss.supervised", supervised_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        return  {"nll_loss": nll_loss.detach()}

    def on_training_epoch_end(self, outputs):
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
        if self.refit:
            datamodule = self.trainer.datamodule
            loader = datamodule.train_dataloader()
            data = next(iter(loader))
            data = {k: [data[k][i].to(self._device) for i in range(len(data[k]))] for k in data.keys()}
            
            labels = torch.cat([data['source_full'],data['target_full']], dim=0)
            if self.use_superclass & (self.dataset in ["cifar100","newsgroups20","amazon_reviews"]) :
                labels = labels//5 if self.dataset=="cifar100" else labels//5
            for l in labels.unique():
                print("Average propensity in class ", str(l), self.expected_propensity[labels == l].mean())
            
            # get first 80% of the data for training, rest for validation
            train_sz = int(data['source_full'][1].shape[0]*0.8)
            train_mask = torch.cat([torch.ones_like(data['source_full'][1][:train_sz]), torch.zeros_like(data['source_full'][1][train_sz:]), torch.ones_like(data['target_full'][1][:train_sz]), torch.zeros_like(data['target_full'][1][train_sz:])], dim=0)
            
            x = torch.cat([data['source_full'][0], data['target_full'][0]], dim=0)
            s = torch.cat([torch.ones_like(data['source_full'][1]), torch.zeros_like(data['target_full'][1])], dim=0)
            
            novelty_detector, novelty_optimizer = get_model(self.arch, self.data_dir, self.dataset, self.num_outputs, pretrained=False,
                                                                learning_rate=self.novelty_learning_rate, weight_decay=self.weight_decay, features=False)
            novelty_detector.to(self._device)
            weights_nonnovel = s / (self.expected_propensity + 1e-12)
            weights_novel = (1 - s) + s * (1 - 1 / (self.expected_propensity + 1e-12))

            y = torch.cat([torch.zeros_like(s), torch.ones_like(s)], dim=0)
            sample_weights = torch.cat([weights_nonnovel, weights_novel], dim=0)
            self.novelty_detector = self._inner_fit(novelty_detector, torch.cat([x,x],dim=0), y, torch.cat([train_mask, train_mask],dim=0), novelty_optimizer, self.inner_epochs, sample_weight=sample_weights)
            
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        
        if dataloader_idx == 0:
            nll, logits, probs, y = self.process_batch(batch, "pred_source")
            outputs = {"probs_s": probs, "y_s": y, "logits_s": logits, "nll_s":nll}
            self.validation_step_outputs_s.append(outputs)
            return outputs

        elif dataloader_idx == 1:
            nll, logits, probs, y = self.process_batch(batch, "pred_source")
            outputs = {"probs_t": probs, "y_t": y, "logits_t": logits, "nll_t":nll}
            self.validation_step_outputs_s.append(outputs)
            return outputs


    def on_validation_epoch_end(self):
        
        outputs = (self.validation_step_outputs_s, self.validation_step_outputs_t)
    
        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        logits_s = torch.cat([x["logits_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        # supervised_loss = cross_entropy(torch.tensor(disc_class_logits_s), torch.tensor(y_s))
        # self.log("pred/supervised_loss", supervised_loss)
        # import pdb; pdb.set_trace()

        probs = np.concatenate((probs_s, probs_t), axis=0)
        y = np.concatenate((np.zeros_like(y_s), np.ones_like(y_t)), axis=0)
        
        # disc_ce_loss = cross_entropy(torch.cat((self.val_features_s, self.val_features_t),dim=0).cpu().detach(), torch.tensor(y))

        y_s_oracle = np.zeros_like(y_s)
        novel_inds = np.where(y_t == self.num_classes)[0]
        y_t_oracle = np.zeros_like(y_t)
        y_t_oracle[novel_inds] = 1
        y_oracle = torch.cat((torch.tensor(y_s_oracle), torch.tensor(y_t_oracle)),dim=0)
        novel_ce_loss = cross_entropy(torch.cat((self.val_features_s, self.val_features_t),dim=0).cpu().detach(), true_labels)
        
        roc_auc = roc_auc_score(y_oracle, probs[:, 1])
        ap = average_precision_score(y_oracle, probs[:, 1])
        f1 = f1_score(y_oracle, np.argmax(probs, axis=1))
        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)
        self.log("val/performance.AP", ap, on_step=False, on_epoch=True)
        self.log("val/performance.F1", f1, on_step=False, on_epoch=True)
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