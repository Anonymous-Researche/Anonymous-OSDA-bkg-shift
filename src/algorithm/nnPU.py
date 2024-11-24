import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot, cross_entropy, binary_cross_entropy

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
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, average_precision_score
import torch.optim.lr_scheduler as lr_scheduler
from src.plots.tsne_plot import *
from src.data_utils import *
from tqdm import tqdm
from src.mpe_utils import *
from copy import deepcopy
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, average_precision_score, f1_score


log = logging.getLogger("app")

class PULoss(nn.Module):
    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.nnPU = nnPU
        print(f"nnPU?: {self.nnPU}")
        self.positive = 1 # source
        self.unlabeled = -1 # target
        self.min_count = torch.tensor(1.)

    def forward(self, inp, y, warmup=True):
        assert (inp.shape == y.shape)
        positive, unlabeled = y == self.positive, y == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        self.min_count.type_as(inp)
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count,
                                                                                            torch.sum(unlabeled))

        y_positive = self.loss_func(positive * inp) * positive
        y_positive_inv = self.loss_func(-positive * inp) * positive
        y_unlabeled = self.loss_func(-unlabeled * inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled
        # if not warmup:
        #     import pdb; pdb.set_trace()
        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk
        else:
            return positive_risk + negative_risk
        
class TrainnnPU(pl.LightningModule):
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
        nnPU: bool = True,
    ):
        super().__init__()
        self.num_classes = num_source_classes
        self.fraction_ood_class = fraction_ood_class
        self._device = device
        self.use_superclass = use_superclass
        self.clip=clip
        self.use_labels = use_labels
        self.constraint_penalty = constrained_penalty
        self.nnPU = nnPU

        self.target_recalls = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45] # [0.02, 0.05, 0.1, 0.15, 0.2] # [0.02, 0.05, 0.15, 0.25]
        self.num_outputs = 2 + num_source_classes # 2*len(self.target_recalls)
        self.dataset = dataset
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.constrained_penalty = constrained_penalty
        self.penalty_type = penalty_type
        self.mode = mode
        self.start = 0
        self.pretrained_model_dir = pretrained_model_dir
        self.pretrained_model_path = pretrained_model_path + self.dataset + "_vanillaPU_seed_" + str(seed) +"_num_source_cls_"+str(num_source_classes)+"_fraction_ood_class_"+str(fraction_ood_class)+ "_ood_ratio_" + str(ood_class_ratio) +"/"+ "discriminator_model.pth"
        
        self.novelty_detector, self.primal_optimizer = get_model(arch, data_dir, self.dataset, self.num_outputs, pretrained= self.pretrained,
                                                                 learning_rate=self.learning_rate, weight_decay=self.weight_decay, features=False,
                                                                 pretrained_model_dir=self.pretrained_model_dir, pretrained_model_path=self.pretrained_model_path)
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

        self.validation_step_outputs_s = []
        self.validation_step_outputs_t = []
        self.validation_step_outputs_discard = []
        self.validation_step_outputs = []
        self.val_features_s = torch.tensor([], device=device)
        self.val_features_t = torch.tensor([], device=device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.best_warmup_loss = 1000.
        self.online_alpha_search = online_alpha_search
        self.alpha_search_midpoint = None
        self.epochs_since_alpha_update = 0.
        self.epochs_for_each_alpha = epochs_for_each_alpha
        self.pure_bin_estimate = [0.]*len(self.target_recalls)
        self.pure_MPE_threshold = [0.]*len(self.target_recalls)
        self.best_bin_size = [0.]*len(self.target_recalls)
        self.best_candidate_alpha = [0.]*len(self.target_recalls)
        self.best_valid_loss = [1000.]*len(self.target_recalls)
        self.best_source_loss = [1000.]*len(self.target_recalls)
        self.auc_roc_at_selection = [0.]*len(self.target_recalls)
        self.ap_at_selection = [0.]*len(self.target_recalls)
        self.oscr_at_selection = [0.]*len(self.target_recalls)
        self.oscpr_at_selection = [0.]*len(self.target_recalls)
        self.precision_at_selection = [0.]*len(self.target_recalls)
        self.recall_at_selection = [0.]*len(self.target_recalls)
        self.acc_at_selection = [0.]*len(self.target_recalls)
        self.recall_target_at_selection = [0.]*len(self.target_recalls)
        self.fpr_at_selection = [1.]*len(self.target_recalls)
        self.num_allowed_fp = -1
        self.alpha_checkpoints = [0.01, 0.1, 0.3, 0.6, 0.9]
        self.constraint_satisified = False
        self.lower_bound_alpha = (target_recall, 0.)
        self.cur_alpha_estimate = (target_recall, 0.)
        self.upper_bound_alpha = (None, 0.)
        self.bin_size_sensitivity = 0.05 #when gap between bin sizes is larger than this, we'll consider then significantly different
        # once constraint is approximately satisifed, allow 5 epochs to train with it, and then reexamine alpha
        self.pu_loss = None # assigned after mixture prior estimation (warm start epochs)

        self.pred_save_path = f"{pred_save_path}/{dataset}/"

        self.logging_file = f"{self.pred_save_path}/PAtR_{arch}_{num_source_classes}_{seed}_log_update.txt"
        
        self.model_path = save_model_path + self.dataset + "_" + "CoNoC_seed_"+str(seed)+"_num_source_cls_"+str(num_source_classes)+"_fraction_ood_class_"+str(fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"/" # "/cis/home/schaud35/shiftpu/models/imagenet_CoNoC_seed_"+str(seed)+"_ood_ratio_"+str(ood_class_ratio)+"/"
        # self.model_path = "/cis/home/schaud35/shiftpu/models/CoNoC_seed_"+str(seed)+"_ood_ratio_"+str(ood_class_ratio)+"/"

        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)

        if not os.path.exists(save_model_path + self.dataset + "_" + "CoNoC_seed_"+str(seed)+"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"/" ):
            os.makedirs(save_model_path + self.dataset + "_" + "CoNoC_seed_"+str(seed)+"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"/" )


        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.warm_start = False if self.warmup_epochs == 0 else True
        self.reload_model = False

        self.automatic_optimization = False
    
    def forward(self, x):
        return self.novelty_detector(x)

    def get_penalty(self, model):
        penalty_lambda = self.constrained_penalty
        if self.penalty_type == 'l2':
            penalty_term = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            penalty_term = sum(torch.abs(p).sum() for p in model.parameters())
        return penalty_lambda*penalty_term
    def get_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def process_batch(self, batch, stage="train"):
        
        if stage == "train":
            # import pdb; pdb.set_trace()
            if len(batch["source_full"][:3])>2:
                x_s, y_s, _ = batch["source_full"][:3]
                x_t, y_t, idx_t = batch["target_full"][:3]
            elif len(batch["source_full"])==2:
                x_s, y_s = batch["source_full"]
                x_t, y_t = batch["target_full"]
            
            if self.use_superclass & (self.dataset in ["cifar100", "newsgroupd20"]):
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
            y = torch.cat([torch.zeros_like(y_s), torch.ones_like(y_t)], dim=0)
            labels = torch.cat([y_s, y_t], dim=0)
            
            y_s_oracle = torch.zeros_like(y_s, device=self._device)
            novel_inds = torch.where(y_t == self.num_classes)[0]
            y_t_oracle = torch.zeros_like(y_t, device=self._device)
            y_t_oracle[novel_inds] = 1
            y_oracle = torch.cat([y_s_oracle, y_t_oracle], dim=0)
            
            if self.warm_start: # train plain domain discriminator during warm start for MPE estimate
                logits_detector = self.forward(x)
                supervised_loss = cross_entropy(logits_detector[y==0][:,:self.num_classes], labels[y==0]) if self.use_labels else 0.
                ce_loss = cross_entropy(logits_detector[:,self.num_classes:], y)
                loss = ce_loss + supervised_loss
                grad_norm = self.get_grad_norm(self.novelty_detector)
                # loss = cross_entropy(logits_detector, y)
                # self.primal_optimizer.zero_grad()
                # self.manual_backward(torch.sum(loss_sum))
                # self.primal_optimizer.step()
                optimizer = self.optimizers()
                optimizer.zero_grad()
                self.manual_backward(loss)
                optimizer.step()

                return loss, self.get_penalty(self.novelty_detector), ce_loss, supervised_loss, grad_norm

            else: # then use MPE estimate for uPU or nnPU risk estimates
                # transform logits and targets into PULoss format (src label: 1, tgt label: -1)
                logits_detector = self.forward(x)
                supervised_loss = cross_entropy(logits_detector[y==0][:,:self.num_classes], labels[y==0]) if self.use_labels else 0.
                pu_outputs = logits_detector[:,0+self.num_classes]
                pu_targets = torch.zeros_like(y, device=self._device, dtype=y.dtype)
                pu_targets[y == 0] = 1
                pu_targets[y == 1] = -1
                pu_loss = self.pu_loss(pu_outputs, pu_targets, self.warm_start)
                loss = pu_loss + supervised_loss
                grad_norm = self.get_grad_norm(self.novelty_detector)
                optimizer = self.optimizers()
                optimizer.zero_grad()
                self.manual_backward(loss)
                optimizer.step()
                return loss, self.get_penalty(self.novelty_detector), pu_loss, supervised_loss, grad_norm

        elif stage=="val":
            if len(batch["source_full"][:3])>2:
                x_s, y_s, _ = batch["source_full"][:3]
                x_t, y_t, idx_t = batch["target_full"][:3]
            elif len(batch["source_full"])==2:
                x_s, y_s = batch["source_full"]
                x_t, y_t = batch["target_full"]

            if self.use_superclass & (self.dataset in ["cifar100", "newsgroupd20"]):
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
            y = torch.cat([torch.zeros_like(y_s), torch.ones_like(y_t)], dim=0)
            labels = torch.cat([y_s, y_t], dim=0)

            y_s_oracle = torch.zeros_like(y_s, device=self._device)
            novel_inds = torch.where(y_t == self.num_classes)[0]
            y_t_oracle = torch.zeros_like(y_t, device=self._device)
            y_t_oracle[novel_inds] = 1
            y_oracle = torch.cat([y_s_oracle, y_t_oracle], dim=0)
            logits_detector = self.novelty_detector(x)
            probs = softmax(logits_detector[:,self.num_classes:], dim=-1)
            if self.warm_start:
                pos_probs = p_probs(self.novelty_detector, self._device, x_s)
                unlabeled_probs, unlabeled_targets = u_probs(self.novelty_detector, self._device, x_t, y_t)
                loss = cross_entropy(logits_detector[:,self.num_classes:], y)
                supervised_loss = cross_entropy(logits_detector[y==0][:,:self.num_classes], labels[y==0])

            else:
                logits_detector = self.forward(x)
                pu_outputs = logits_detector[:,0+self.num_classes]
                pu_targets = torch.zeros_like(y, device=self._device, dtype=y.dtype)
                pu_targets[pu_targets == 0] = 1
                pu_targets[pu_targets == 1] = -1
                loss = self.pu_loss(pu_outputs, pu_targets)
                supervised_loss = cross_entropy(logits_detector[y==0][:,:self.num_classes], labels[y==0])
                pos_probs, unlabeled_probs, unlabeled_targets = 0., 0., 0.
            
            return loss, logits_detector, probs, y, y_oracle, torch.tensor(pos_probs), torch.tensor(unlabeled_probs), torch.tensor(unlabeled_targets), supervised_loss, labels

        else:
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss, penalty, pu_loss, supervised_loss, grad_norm = self.process_batch(batch, "train")
        self.log("train/loss.grad_norm", grad_norm, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss.pu_loss", pu_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss.penalty", penalty, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss.CE", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss.supervised_loss", supervised_loss, on_step=False, on_epoch=True, prog_bar=False)
        return  {"loss": loss.detach()} #{"source_loss": loss1.detach(), "discriminator_loss": loss2.detach()}

    def on_training_epoch_end(self):
        total_norm = 0
        for p in self.novelty_detector.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        log.info('gradient norm after training {}'.format(total_norm))

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx==3:
            loss, logits, probs, y, y_oracle, pos_probs, unlabeled_probs, unlabeled_targets, supervised_loss, labels = self.process_batch(batch, "val")
            outputs =  {"loss": loss, "logits": logits, "probs": probs, "pos_probs": pos_probs, "unlabeled_probs": unlabeled_probs, "unlabeled_targets": unlabeled_targets, "y": y, "y_oracle": y_oracle, "supervised_loss": supervised_loss, "labels": labels} 
            self.validation_step_outputs.append(outputs)
            if self.warm_start:
                self.log("val/loss.warm_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("val/loss.", 1e3 - self.current_epoch, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(probs)) # dummy val loss just to avoid earlystopping
            else:
                self.log("val/loss.post_warmup_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/loss.supervised_loss", supervised_loss, on_step=False, on_epoch=True, prog_bar=False)
            return outputs

    def on_validation_epoch_end(self):
        # if not self.warm_start:
            # import pdb; pdb.set_trace()
        outputs = self.validation_step_outputs
        logits = torch.cat([x["logits"] for x in outputs], dim=0).detach().cpu().numpy()
        probs = torch.cat([x["probs"] for x in outputs], dim=0).detach().cpu().numpy()
        y = torch.cat([x["y"] for x in outputs], dim=0).detach().cpu().numpy()
        y_oracle = torch.cat([x["y_oracle"] for x in outputs], dim=0).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs], dim=0).detach().cpu().numpy()

        if self.warm_start:
            pos_probs = torch.cat([x["pos_probs"] for x in outputs], dim=0).detach().cpu().numpy()
            unlabeled_probs = torch.cat([x["unlabeled_probs"] for x in outputs], dim=0).detach().cpu().numpy()
            unlabeled_targets = torch.cat([x["unlabeled_targets"] for x in outputs], dim=0).detach().cpu().numpy()
            mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets) # unlabeled_targets isn't used for calculating the mpe estimate
            self.prior = mpe_estimate
        
        true_prior = 1 - y_oracle.sum().item()/len(y_oracle)

        roc_auc = roc_auc_score(y_oracle[y==1], probs[:, 1][y==1])
        ap = average_precision_score(y_oracle[y==1], probs[:, 1][y==1])
        f1 = f1_score(y_oracle[y==1], np.argmax(probs[y==1], axis=1))
        oscr = self.compute_oscr(probs[:,1][y==1][y_oracle[y==1]==0], probs[:,1][y==1][y_oracle[y==1]==1], np.argmax(logits[:,:self.num_classes][y==1], axis=1)[y_oracle[y==1]==0], labels[y==1][y_oracle[y==1]==0])
        oscpr = self.compute_oscpr(probs[:,1][y==1][y_oracle[y==1]==0], probs[:,1][y==1][y_oracle[y==1]==1], np.argmax(logits[:,:self.num_classes][y==1], axis=1)[y_oracle[y==1]==0], labels[y==1][y_oracle[y==1]==0])
        
        self.log("val/performance.OSCR", oscr, on_step=False, on_epoch=True)
        self.log("val/performance.OSCPR", oscpr, on_step=False, on_epoch=True)
        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True)
        self.log("val/performance.AP", ap, on_step=False, on_epoch=True)
        self.log("val/performance.F1", f1, on_step=False, on_epoch=True)
        self.log("val/estimated_prior", self.prior, on_step=False, on_epoch=True)
        self.log("val/true_prior", true_prior, on_step=False, on_epoch=True)
        
        if self.warm_start: # checkpoint best warmup model
            
            loss = cross_entropy(torch.tensor(logits[:,self.num_classes:]), torch.tensor(y))
            if loss < self.best_warmup_loss:
                self.best_warmup_loss = loss
                self.best_warmup_model = deepcopy(self.novelty_detector)

            if self.current_epoch < self.warmup_epochs:
                self.warm_start = True # keep it true
            else:
                print(f"End warm up at epoch: {self.current_epoch}")
                self.warm_start = False
                self.pu_loss = PULoss(prior=self.prior, nnPU=self.nnPU)
                self.novelty_detector = deepcopy(self.best_warmup_model)
                del self.best_warmup_model

        self.validation_step_outputs = []

    def compute_oscr(self, x1, x2, pred, labels):

        """
        Compute Open Set Classification Rate based on implementation in 
        LMC: Large Model Collaboration with Cross-assessment for Training-Free Open-Set Object Recognition
        :param x1: open set score for each known class sample (B_k,)
        :param x2: open set score for each unknown class sample (B_u,)
        :param pred: predicted class for each known class sample (B_k,)
        :param labels: correct class for each known class sample (B_k,)
        :return: Open Set Classification Rate
        """
        # if self.current_epoch==200:
        #     import pdb; pdb.set_trace()
        x1, x2 = -x1, -x2

        # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
        # pred = np.argmax(pred_k, axis=1)

        correct = (pred == labels)
        m_x1 = np.zeros(len(x1))
        m_x1[pred == labels] = 1
        k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
        u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
        predict = np.concatenate((x1, x2), axis=0)
        n = len(predict)

        # Cutoffs are of prediction values

        CCR = [0 for x in range(n + 2)]
        FPR = [0 for x in range(n + 2)]

        idx = predict.argsort()

        s_k_target = k_target[idx]
        s_u_target = u_target[idx]

        for k in range(n - 1):
            CC = s_k_target[k + 1:].sum()
            FP = s_u_target[k:].sum()

            # True	Positive Rate
            CCR[k] = float(CC) / float(len(x1))
            # False Positive Rate
            FPR[k] = float(FP) / float(len(x2))

        CCR[n] = 0.0
        FPR[n] = 0.0
        CCR[n + 1] = 1.0
        FPR[n + 1] = 1.0

        # Positions of ROC curve (FPR, TPR)
        ROC = sorted(zip(FPR, CCR), reverse=True)

        OSCR = 0

        # Compute AUROC Using Trapezoidal Rule
        for j in range(n + 1):
            h = ROC[j][0] - ROC[j + 1][0]
            w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

            OSCR = OSCR + h * w

        # if self.current_epoch==200:
        #     import pdb; pdb.set_trace()
        return OSCR

    def compute_oscpr(self, x1, x2, pred, labels):

        """
        Compute Open Set Classification Rate based on implementation in 
        LMC: Large Model Collaboration with Cross-assessment for Training-Free Open-Set Object Recognition
        :param x1: open set score for each known class sample (B_k,)
        :param x2: open set score for each unknown class sample (B_u,)
        :param pred: predicted class for each known class sample (B_k,)
        :param labels: correct class for each known class sample (B_k,)
        :return: Open Set Classification Rate
        """
        # if self.current_epoch==200:
        #     import pdb; pdb.set_trace()
        x1, x2 = -x1, -x2

        # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
        # pred = np.argmax(pred_k, axis=1)

        correct = (pred == labels)
        m_x1 = np.zeros(len(x1))
        m_x1[pred == labels] = 1
        k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
        u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
        predict = np.concatenate((x1, x2), axis=0)
        n = len(predict)

        # Cutoffs are of prediction values

        recall = [0 for x in range(n + 2)]
        precision = [0 for x in range(n + 2)]

        idx = predict.argsort()

        s_k_target = k_target[idx]
        s_u_target = u_target[idx]

        for k in range(n - 1):
            CC = s_k_target[k + 1:].sum()
            FP = s_u_target[k:].sum()
            FN = s_k_target[:k + 1].sum()

            # True	Positive Rate
            recall[k] = float(CC) / float(len(x1))
            # False Positive Rate
            precision[k] = float(CC) / (float(CC)+float(FN))

        recall[n] = 0.0
        precision[n] = 0.0
        recall[n + 1] = 1.0
        precision[n + 1] = 1.0

        # Positions of ROC curve (FPR, TPR)
        ROC = sorted(zip(recall, precision), reverse=True)

        OSCPR = 0

        # Compute AUROC Using Trapezoidal Rule
        for j in range(n + 1):
            h = ROC[j][0] - ROC[j + 1][0]
            w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

            OSCPR = OSCPR + h * w

        # if self.current_epoch==200:
        #     import pdb; pdb.set_trace()
        return OSCPR
    
    def dataselector(self, unlabeled_data, budget_per_AL_cycle, batch_size=200):
        self.novelty_detector.load_state_dict(self.novelty_detector.state_dict(), self.model_path + "novelty_detector_model.pth")
        unlabeled_dataloder = DataLoader( unlabeled_data, batch_size=batch_size, shuffle=False, \
            num_workers=8,  pin_memory=True)
        all_probs = torch.tensor([], device=self._device)
        for batch in tqdm(unlabeled_dataloder):
            all_probs = torch.cat((all_probs, softmax(self.novelty_detector(batch[0]), dim=-1)),dim=0)
        
        
        return
        

    def configure_optimizers(self):

        return [self.primal_optimizer]