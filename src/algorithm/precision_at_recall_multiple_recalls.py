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

log = logging.getLogger("app")

class RecallConstrainedClassification(ConstrainedMinimizationProblem):
    def __init__(self, target_recall=0.1, wd=0., penalty_type='l2', logit_multiplier=2., device='cuda', mode='domain_disc'):
        # self.criterion = torch.nn.BCELoss(reduction='mean')
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.target_recall = target_recall
        self.wd = wd
        self.penalty_type = penalty_type
        self.logit_multiplier = logit_multiplier
        self.device = device
        if mode=='constrained_opt':
            super().__init__(is_constrained=True)
        else:
            super().__init__(is_constrained=False)

    def get_penalty(self, model):
        penalty_lambda = self.wd
        if self.penalty_type == 'l2':
            penalty_term = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            penalty_term = sum(torch.abs(p).sum() for p in model.parameters())
        return penalty_lambda*penalty_term

    def closure(self, model, inputs, targets):
        # import pdb; pdb.set_trace()
        pred_logits = model.forward(inputs)
        pred_logits = pred_logits.reshape(pred_logits.shape[0],-1,2)
        with torch.no_grad():
            predictions = torch.argmax(pred_logits, dim=-1)
        
        penalty = self.get_penalty(model)
        cross_ent_ls, recall_ls, recall_proxy_ls, recall_loss_ls, preds_temp_ls, cross_ent_target_ls = torch.tensor([], requires_grad=True, device=self.device), torch.tensor([], requires_grad=True, device=self.device), torch.tensor([], requires_grad=True, device=self.device), torch.tensor([], requires_grad=True, device=self.device), torch.tensor([], requires_grad=True, device=self.device), torch.tensor([], requires_grad=True, device=self.device)
        
        for i in range(pred_logits.shape[1]):
            cross_ent = self.criterion(pred_logits[targets==0][:,i,:], targets[targets==0])
            cross_ent_target = self.criterion(pred_logits[targets==1][:,i,:], targets[targets==1])
            # cross_ent = self.criterion(pred_logits[:,i,:], targets)
            recall, recall_proxy, recall_loss, preds_temp, positives_temp = recall_from_logits(self.logit_multiplier*pred_logits[:,i,:],
                                                                            targets)
        
            
            cross_ent_ls = torch.cat((cross_ent_ls, torch.unsqueeze(cross_ent,0)))
            cross_ent_target_ls = torch.cat((cross_ent_target_ls, torch.unsqueeze(cross_ent_target,0)))
            recall_ls = torch.cat((recall_ls, torch.unsqueeze(recall,0)))
            recall_proxy_ls = torch.cat((recall_proxy_ls, torch.unsqueeze(recall_proxy,0)))
            preds_temp_ls = torch.cat((preds_temp_ls, torch.unsqueeze(preds_temp,0)))
            recall_loss_ls = torch.cat((recall_loss_ls, torch.unsqueeze(recall_loss,0)))
            # positives_temp_ls = torch.cat((positives_temp_ls, torch.unsqueeze(positives_temp,0)))                                                                 
        # cross_ent = self.criterion(pred_logits[targets==0][:,i:i+1])
        
        # cross_ent_ls = torch.sum(cross_ent_ls)
        loss = cross_ent_ls + penalty # 0.1*cross_ent + penalty
        loss_target = cross_ent_target + penalty
        
        
        ineq_defect = torch.tensor(self.target_recall, device=self.device) - recall_ls
        # import pdb; pdb.set_trace()
        proxy_ineq_defect = torch.tensor(self.target_recall, device=self.device) - recall_proxy_ls
        
        # loss = torch.sum(loss)/len(self.target_recall)
        # ineq_defect = torch.sum(ineq_defect)/len(self.target_recall)
        # proxy_ineq_defect = torch.sum(proxy_ineq_defect)/len(self.target_recall)
        
        total_norm = 0
        for n,p in model.named_parameters():
            if (n[-13:] == 'linear.weight' or n[-9:]=="fc.weight" or n[-9:]=="f4.weight" or n[-9:]=="f5.weight") and p.grad is not None:
                print('===========\ngradient:{}\n----------\n{}'.format(n,p.grad.data.norm(2)))
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print(total_norm, loss, loss_target, cross_ent_ls, cross_ent_target_ls, recall_ls, ineq_defect, proxy_ineq_defect)
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        return CMPState(loss=loss, ineq_defect=ineq_defect, proxy_ineq_defect=proxy_ineq_defect, recall_loss=recall_loss_ls,
                        eq_defect=None, misc={'cross_ent': torch.tensor(cross_ent_ls), 'cross_ent_target': torch.tensor(cross_ent_target_ls), 'recall_proxy': torch.tensor(recall_proxy_ls)})

class FPRConstrainedTiltedERM(ConstrainedMinimizationProblem):
    def __init__(self, target_fpr=0.01, wd=0., penalty_type='l2', logit_multiplier=2., device='cuda', mode='domain_disc'):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.target_fpr = target_fpr
        self.wd = wd
        self.penalty_type = penalty_type
        self.logit_multiplier = logit_multiplier
        self.device = device
        if mode.startswith('constrained'):
            super().__init__(is_constrained=True)
        else:
            super().__init__(is_constrained=False)

    def tilt_loss(self, loss: torch.Tensor, tilt=100):
        """
        As defined in Li et al. Tilted ERM paper
        """
        return (1/tilt)*torch.log(torch.mean((torch.exp(tilt*loss))))

    def get_penalty(self, model):
        penalty_lambda = self.wd
        if self.penalty_type == 'l2':
            penalty_term = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            penalty_term = sum(torch.abs(p).sum() for p in model.parameters())
        return penalty_lambda*penalty_term

    def closure(self, model, inputs, targets):
        pred_logits = model.forward(inputs)
        pred_logits = pred_logits.reshape(pred_logits.shape[0],-1,2)
        with torch.no_grad():
            predictions = torch.argmax(pred_logits, dim=-1)
        
        penalty = self.get_penalty(model)
        loss_source_ls, loss_target_ls, loss_target_tilt_ls = torch.tensor([], requires_grad=True, device=self.device), torch.tensor([], requires_grad=True, device=self.device), torch.tensor([], requires_grad=True, device=self.device)
        for i in range(pred_logits.shape[1]):
            loss_source = self.criterion(pred_logits[targets==0][:,i,:], targets[targets==0])
            loss_target = self.criterion(pred_logits[targets==1][:,i,:], targets[targets==1]) + penalty
            loss_target_tilt = self.tilt_loss(loss_target, tilt=-10)

            loss_source_ls = torch.cat((loss_source_ls, torch.unsqueeze(loss_source,0)))
            loss_target_ls = torch.cat((loss_target_ls, torch.unsqueeze(loss_target,0)))
            loss_target_tilt_ls = torch.cat((loss_target_tilt_ls, torch.unsqueeze(loss_target_tilt,0)))
        loss_source_ls = loss_source_ls - self.target_fpr
        loss_target_tilt_ls = loss_target_tilt_ls 
        total_grad_norm, total_param_norm = 0, 0
        for n,p in model.named_parameters():
            if (n[-13:] == 'linear.weight' or n[-9:]=="fc.weight") and p.grad is not None:
                print('===========\ngradient:{}\n----------\n{}'.format(n,p.grad.data.norm(2)))
        
        for p in model.parameters():
            if p.grad is not None:
                grad_norm = p.grad.data.norm(2)
                param_norm = p.data.norm(2)
                total_grad_norm += grad_norm.item() ** 2
                total_param_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1. / 2)
        total_param_norm = total_param_norm ** (1. / 2)
        print(total_grad_norm, total_param_norm)
            # loss = sum(y==0)/(sum(y==0) + sum(y==1))*loss_source + sum(y==1)/(sum(y==0) + sum(y==1))*loss_target
            # loss_ls = torch.cat((loss_ls, torch.unsqueeze(loss,0)))
        
        return CMPState(loss=loss_target_tilt_ls, ineq_defect=loss_source_ls, proxy_ineq_defect=loss_source_ls, 
                        misc={'cross_ent': loss_source_ls + loss_target_ls, 'loss_target':loss_target_ls})

class TrainPAtR(pl.LightningModule):
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
        ood_class_ratio: float = 0.005,
        fraction_ood_class: float = 0.01,
        constrained_penalty: float = 3e-7,
        save_model_path: str = "/cis/home/schaud35/shiftpu/models/",
    ):
        super().__init__()
        self.num_classes = num_source_classes
        self.fraction_ood_class = fraction_ood_class
        self._device = device

        self.target_recalls = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45] # [0.02, 0.05, 0.1, 0.15, 0.2] # [0.02, 0.05, 0.15, 0.25]
        self.num_outputs = 2*len(self.target_recalls)
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
        self.novelty_detector, self.primal_optimizer = get_model(arch, self.dataset, self.num_outputs, pretrained= self.pretrained,
                                                                 learning_rate=self.learning_rate, weight_decay=self.weight_decay, features=False,
                                                                 pretrained_model_dir=self.pretrained_model_dir, pretrained_model_path=self.pretrained_model_path)
        # self.primal_optimizer = [torch.optim.AdamW(self.novelty_detector.parameters(), lr=learning_rate, weight_decay=weight_decay), torch.optim.AdamW(self.novelty_detector.parameters(), lr=learning_rate, weight_decay=weight_decay)]
        self.primal_lr_scheduler = lr_scheduler.LinearLR(self.primal_optimizer, start_factor=1.0, end_factor=1.0, total_iters=15000)
        self.target_precision = target_precision
        self.precision_confidence = precision_confidence
        # self.target_recall = 0.02 # target_recall
        if self.mode=='constrained_opt':
            self.dual_optimizer = constrained_optimization.optim.partial_optimizer(torch.optim.Adam, lr=dual_learning_rate, weight_decay=self.weight_decay)
            self.cmp = RecallConstrainedClassification(target_recall=self.target_recalls, wd=self.constrained_penalty,
                                                   penalty_type=self.penalty_type, logit_multiplier=logit_multiplier, device=self._device, mode=self.mode)
            self.formulation = LagrangianFormulation(self.cmp, ineq_init = torch.tensor([1. for i in range(len(self.target_recalls))])) ## start from here tomorrow!!
        
            
            # self.primal_lr_scheduler = [lr_scheduler.LinearLR(self.primal_optimizer[0], start_factor=1.0, end_factor=0.001, total_iters=6200), lr_scheduler.LinearLR(self.primal_optimizer[1], start_factor=1.0, end_factor=1.0, total_iters=6200)]
            self.dual_lr_scheduler = constrained_optimization.optim.partial_scheduler(lr_scheduler.LinearLR, start_factor=1.0, end_factor=1.0, total_iters=15000) if self.mode=='constrained_opt' else None
        
            self.coop = ConstrainedOptimizer(
                formulation=self.formulation,
                primal_optimizer=self.primal_optimizer,
                primal_scheduler=self.primal_lr_scheduler,
                dual_optimizer=self.dual_optimizer,
                dual_scheduler=self.dual_lr_scheduler,
            )
        elif self.mode == 'constrained_tilted_erm':
            self.dual_optimizer = constrained_optimization.optim.partial_optimizer(torch.optim.Adam, lr=dual_learning_rate, weight_decay=self.weight_decay)
            self.cmp = FPRConstrainedTiltedERM(target_fpr=1-self.target_precision, wd=self.constrained_penalty,
                                                   penalty_type=self.penalty_type, logit_multiplier=logit_multiplier, device=self._device, mode=self.mode)
            self.formulation = LagrangianFormulation(self.cmp, ineq_init = torch.tensor([1. for i in range(len(self.target_recalls))])) 
            self.dual_lr_scheduler = constrained_optimization.optim.partial_scheduler(lr_scheduler.LinearLR, start_factor=1.0, end_factor=1.0, total_iters=15000) if self.mode.startswith('constrained') else None
            self.coop = ConstrainedOptimizer(
                formulation=self.formulation,
                primal_optimizer=self.primal_optimizer,
                primal_scheduler=self.primal_lr_scheduler,
                dual_optimizer=self.dual_optimizer,
                dual_scheduler=self.dual_lr_scheduler,
            )
        else:
            self.dual_optimizer = None
            self.dual_lr_scheduler = None
            # self.dual_optimizer =  constrained_optimization.optim.partial_optimizer(torch.optim.SGD, lr=dual_learning_rate)

        

        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

        self.validation_step_outputs_s = []
        self.validation_step_outputs_t = []
        self.validation_step_outputs_discard = []
        self.val_features_s = torch.tensor([], device=device)
        self.val_features_t = torch.tensor([], device=device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Some variables for the alpha line search
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

    def update_alpha_search_params(self):
        import pdb; pdb.set_trace()
        ## handle case where we still didn't find an upper bound for our search.
        self.epochs_since_alpha_update = 0
        log.info('begining update when cur estimate is {}'.format(str(self.cur_alpha_estimate)))
        log.info('upper is {}'.format(str(self.upper_bound_alpha)))
        log.info('lower is {}'.format(str(self.lower_bound_alpha)))
        if self.upper_bound_alpha is None:
            if self.cur_alpha_estimate[1] >= self.lower_bound_alpha[1] + self.bin_size_sensitivity:
                if self.cur_alpha_estimate[0] == self.alpha_checkpoints[-1]:
                    log.info('upper bound on constraint value is set to max possible')
                    self.upper_bound_alpha = self.cur_alpha_estimate
                    self.cur_alpha_estimate = ((self.upper_bound_alpha[0] + self.lower_bound_alpha[0]) / 2., 0.)
                else:
                    self.cur_alpha_estimate = (np.min(self.alpha_checkpoints[self.alpha_checkpoints > self.cur_alpha_estimate[0]]), 0.)
            elif self.cur_alpha_estimate[1] <= self.lower_bound_alpha[1] + self.bin_size_sensitivity:
                log.info('upper bound on constraint value is set')
                self.upper_bound_alpha = self.cur_alpha_estimate
                self.cur_alpha_estimate = ((self.upper_bound_alpha[0] + self.lower_bound_alpha[0]) / 2., 0.)
            return
        ## If we got here then there is an upper bound and we need to update according to standard binary search
        if self.lower_bound_alpha[1] > self.cur_alpha_estimate[1] > self.upper_bound_alpha[1]:
            log.info('setting a new upper bound for search at {}'.format(self.cur_alpha_estimate[0]))
            self.upper_bound_alpha = self.cur_alpha_estimate
            self.cur_alpha_estimate = ((self.upper_bound_alpha[0] + self.lower_bound_alpha[0]) / 2., 0.)
            self.alpha_search_midpoint = None
            return
        if self.lower_bound_alpha[1] < self.cur_alpha_estimate[1] < self.upper_bound_alpha[1]:
            log.info('setting a new lower bound for search at {}'.format(self.cur_alpha_estimate[0]))
            self.lower_bound_alpha = self.cur_alpha_estimate
            self.cur_alpha_estimate = ((self.upper_bound_alpha[0] + self.lower_bound_alpha[0]) / 2., 0.)
            self.alpha_search_midpoint = None
            return
        ## In case current search point is a peak between both endpoints, we store this as a midpoint and set search
        ## between lower bound and this one
        if self.lower_bound_alpha[1] < self.cur_alpha_estimate[1] > self.upper_bound_alpha[1]:
            if self.alpha_search_midpoint is None:
                log.info('')
                self.alpha_search_midpoint = self.cur_alpha_estimate
                self.cur_alpha_estimate = ((self.alpha_search_midpoint[0] + self.lower_bound_alpha[0]) / 2., 0.)
                return
            else:
                if self.cur_alpha_estimate[1] < self.alpha_search_midpoint[1]:
                    log.info('setting a new lower bound for search at {}'.format(self.cur_alpha_estimate[0]))
                    self.lower_bound_alpha = self.cur_alpha_estimate
                else:
                    log.info('setting a new upper bound for search at {}'.format(self.cur_alpha_estimate[0]))
                    self.upper_bound_alpha = self.alpha_search_midpoint
                self.cur_alpha_estimate = ((self.upper_bound_alpha[0] + self.lower_bound_alpha[0]) / 2., 0.)
                self.alpha_search_midpoint = None
        ## In case current search point is a valley between both endpoints, it's weird and we just set new search
        ## between lower value and this one
        if self.lower_bound_alpha[1] > self.cur_alpha_estimate[1] < self.upper_bound_alpha[1]:
             self.cur_alpha_estimate = ((self.upper_bound_alpha[0] + self.lower_bound_alpha[0]) / 2., 0.)
             return


    def reset_constrained_problem(self, target_recall, reset_model_weights = False):
        self.target_recalls = target_recall
        self.cmp = RecallConstrainedClassification(target_recall=target_recall, wd=self.weight_decay,
                                                   penalty_type=self.penalty_type)
        cur_ineq_weight = self.formulation.ineq_multipliers.weight.data
        self.formulation = LagrangianFormulation(self.cmp, ineq_init = cur_ineq_weight)
        self.coop = ConstrainedOptimizer(
            formulation=self.formulation,
            primal_optimizer=self.primal_optimizer,
            dual_optimizer=self.dual_optimizer,
        )
        if reset_model_weights:
            self.novelty_detector, self.primal_optimizer = (
                get_model(arch, self.dataset, self.num_outputs, pretrained=self.pretrained,
                          learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                          pretrained_model_dir=self.pretrained_model_dir))

    def tilt_loss(self, loss: torch.Tensor, tilt=100):
        """
        As defined in Li et al. Tilted ERM paper
        """
        return (1/tilt)*torch.log(torch.mean((torch.exp(tilt*loss))))
    
    def forward(self, x):
        return self.novelty_detector(x)

    def process_batch(self, batch, stage="train"):
        
        if self.current_epoch>=self.warmup_epochs:
            self.warm_start=False
        if stage == "train":
            # import pdb; pdb.set_trace()
            if len(batch["source_full"][:3])>2:
                x_s, y_s, _ = batch["source_full"][:3]
                x_t, y_t, idx_t = batch["target_full"][:3]
            elif len(batch["source_full"])==2:
                x_s, y_s = batch["source_full"]
                x_t, y_t = batch["target_full"]

            x = torch.cat([x_s, x_t], dim=0)
            y = torch.cat([torch.zeros_like(y_s), torch.ones_like(y_t)], dim=0)

            # y_s_oracle = torch.zeros_like(y_s, device=self._device)
            # novel_inds = torch.where(y_t == self.num_classes)[0]
            # y_t_oracle = torch.zeros_like(y_t, device=self._device)
            # y_t_oracle[novel_inds] = 1
            # y = torch.cat([y_s_oracle, y_t_oracle], dim=0)

            if self.warm_start:
                logits_detector = self.novelty_detector(x)
                logits_detector = logits_detector.reshape(logits_detector.shape[0],-1,2)
                loss_sum = torch.tensor([], requires_grad=True, device=self._device)
                loss_ls = torch.tensor([], requires_grad=True, device=self.device)
                for i in range(logits_detector.shape[1]):
                    loss = cross_entropy(logits_detector[:,i,:], y)
                    loss_ls = torch.cat((loss_ls, torch.unsqueeze(loss,0)))
                    loss_sum = torch.cat((loss_sum, torch.unsqueeze(loss,0)))
                # loss = cross_entropy(logits_detector, y)
                self.primal_optimizer.zero_grad()
                self.manual_backward(torch.sum(loss_sum))
                self.primal_optimizer.step()

                return loss_ls, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

            elif self.mode=="tilted_erm":
                # import pdb; pdb.set_trace()
                logits_detector = self.novelty_detector(x)
                logits_detector = logits_detector.reshape(logits_detector.shape[0],-1,2)
                loss_ls, loss_source_ls, loss_target_ls = torch.tensor([], requires_grad=True, device=self.device), torch.tensor([], requires_grad=True, device=self.device), torch.tensor([], requires_grad=True, device=self.device)
                for i in range(logits_detector.shape[1]):
                    loss_source = cross_entropy(logits_detector[:,i,:], y[y==0])
                    loss_target = cross_entropy(logits_detector[:,i,:], y[y==1])
                    loss_target = self.tilt_loss(loss_target, tilt=-2)
                    loss = loss_source + loss_target
                    loss_ls = torch.cat((loss_ls, torch.unsqueeze(loss,0)))
                    loss_source_ls = torch.cat((loss_source_ls, torch.unsqueeze(loss_source,0)))
                    loss_target_ls = torch.cat((loss_target_ls, torch.unsqueeze(loss_target,0)))

                # loss = cross_entropy(logits_detector, y)
                self.primal_optimizer.zero_grad()
                self.manual_backward(torch.sum(loss_ls))
                self.primal_optimizer.step()
                return loss_ls, loss_source, torch.tensor(0.), loss_target, torch.tensor(0.)
            
            elif self.mode.endswith('tilted_erm'):
                lagrangian = self.formulation.composite_objective(
                  self.cmp.closure, self.novelty_detector, x, y
                )
                self.formulation.custom_backward(lagrangian)
                torch.nn.utils.clip_grad_norm_(self.novelty_detector.parameters(), 5.0)
                self.coop.step(self.cmp.closure, self.novelty_detector, x, y)
                # print(self.coop.primal_optimizer.param_groups[0]['lr'], self.coop.dual_optimizer.param_groups[0]['lr'])
                return self.cmp.state.loss, self.cmp.get_penalty(self.novelty_detector), self.cmp.state.ineq_defect, lagrangian, self.cmp.state.misc['loss_target']

            else:
                lagrangian = self.formulation.composite_objective(
                  self.cmp.closure, self.novelty_detector, x, y
                )
                
                self.formulation.custom_backward(lagrangian)
                # torch.nn.utils.clip_grad_norm_(self.novelty_detector.parameters(), 5.0)

                self.coop.step(self.cmp.closure, self.novelty_detector, x, y)
#                 print(self.cmp.state)
#                 print(self.formulation.cmp.is_constrained)
#                 print(self.formulation.weighted_violation(self.cmp.state, "ineq"))
#                 print('lag val after {}'.format(self.formulation.composite_objective(
#                   self.cmp.closure, self.novelty_detector, x, y
#                 )))
                print(self.coop.primal_optimizer.param_groups[0]['lr'], self.coop.dual_optimizer.param_groups[0]['lr'])
                # print(self.primal_optimizer[0].param_groups[0]['lr'], self.priimal_optimizer[1].param_groups[0]['lr'])
                # import pdb; pdb.set_trace()
                return self.cmp.state.loss, self.cmp.get_penalty(self.novelty_detector), self.cmp.state.ineq_defect, lagrangian, torch.tensor(0.)

            if self.trainer.is_last_batch:
                update_optimizer(self.current_epoch, self.primal_optimizer, self.dataset, self.learning_rate)

            return loss2, self.cmp.get_penalty(self.novelty_detector), self.cmp.state.ineq_defect, torch.tensor(0.)

        elif stage == "pred_source":
            # import pdb; pdb.set_trace()
            if len(batch)>2:
                x_s, y_s, _ = batch[:3]
            elif len(batch)==2:
                x_s, y_s = batch

            logits = self.novelty_detector(x_s)
            self.val_features_s = torch.cat((self.val_features_s,logits), dim=0) 
            logits = logits.reshape(logits.shape[0], -1, 2)
            probs_s = softmax(logits, dim=-1)
#             disc_probs_s = probs
#
#             logits_s = self.source_model(x_s)
#             probs_s = softmax(logits_s, dim=1)
            return probs_s, y_s

        elif stage == "pred_target":
            # import pdb; pdb.set_trace()
            if len(batch)>2:
                x_t, y_t, _ = batch[:3]
            elif len(batch)==2:
                x_t, y_t = batch
            
            logits = self.novelty_detector(x_t)
            self.val_features_t = torch.cat((self.val_features_t,logits), dim=0) 
            logits = logits.reshape(logits.shape[0], -1, 2)
            probs_t = softmax(logits, dim=-1)
#             disc_probs_t = probs
#
#             logits_t = self.source_model(x_t)
#             probs_t = softmax(logits_t, dim=1)
            return probs_t, y_t

        elif stage == "discard":
            # import pdb; pdb.set_trace()
            if len(batch)>2:
                x_t, _, idx_t  = batch[:3]
            elif len(batch)==2:
                x_t, _ = batch
                idx_t = None
            logits = self.novelty_detector(x_t)
            logits = logits.reshape(logits.shape[0], -1, 2)
            probs = softmax(logits, dim = -1)[:,:,1]

            return probs, idx_t

        else:
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss, penalty, ineq_defect, lagrangian_value, target_fpr = self.process_batch(batch, "train")
        # self.log("train/loss", {"cross_ent": torch.sum(loss), "constraint_penalty": torch.sum(penalty), "lagrangian": lagrangian_value},
        #          on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss.constraint_penalty", penalty, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss.lagrangian", lagrangian_value, on_step=True, on_epoch=True, prog_bar=False)
        for i in range(len(self.target_recalls)):
            self.log("train/loss.cross_ent_"+str(self.target_recalls[i]), loss[i], on_step=True, on_epoch=True, prog_bar=False)
        
        if not self.warm_start:
            if self.mode.startswith('constrained'):
                # self.log("train/constraints", {"inequality_violation": torch.sum(ineq_defect),
                #                             "multiplier_value": torch.sum(self.formulation.ineq_multipliers.weight.detach().cpu())}, #, "recall_proxy": recall_proxy
                #         on_step=True, on_epoch=True, prog_bar=False)
                for i in range(len(self.target_recalls)):
                    self.log("train/constraints.inequality_violation_"+str(self.target_recalls[i]), ineq_defect[i], on_step=True, on_epoch=True, prog_bar=False)
                    self.log("train/constraints.multiplier_value_"+str(self.target_recalls[i]), self.formulation.ineq_multipliers.weight.detach().cpu()[i], on_step=True, on_epoch=True, prog_bar=False)
                    if self.mode.endswith('tilted_erm'):
                        self.log("train/loss.target_fpr_"+str(self.target_recalls[i]), target_fpr[i], on_step=True, on_epoch=True, prog_bar=False)
        return  {"lagrangian_loss": lagrangian_value.detach()} #{"source_loss": loss1.detach(), "discriminator_loss": loss2.detach()}

    def on_training_epoch_end(self, outputs):
        total_norm = 0
        for p in self.novelty_detector.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        log.info('gradient norm after training {}'.format(total_norm))
        if self.current_epoch > self.warmup_epochs:
            self.warm_start = False
            if self.online_alpha_search:
                ## see if it's time to update the alpha search
                if self.epochs_since_alpha_update >= self.epochs_for_each_alpha:
                    self.update_alpha_search_params()
                    self.reset_constrained_problem(self.cur_alpha_estimate[0])
                    self.epochs_since_alpha_update = 0
                else:
                    self.epochs_since_alpha_update += 1
#             else:
#                 if self.reload_model:
#                     self.novelty_detector.load_state_dict(torch.load(self.model_path + "novelty_detection_model.pth"))
#                     self.warm_start = False
#                     self.reload_model = False

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        
        if dataloader_idx == 0:
            probs_s, y_s = self.process_batch(batch, "pred_source")
            outputs = {"probs_s": probs_s, "y_s": y_s}#, "disc_probs_s": disc_probs_s }
            self.validation_step_outputs_s.append(outputs)
            return outputs

        elif dataloader_idx == 1:
            probs_t, y_t = self.process_batch(batch, "pred_target")
            outputs = {"probs_t": probs_t, "y_t": y_t}#, "disc_probs_t": disc_probs_t}
            self.validation_step_outputs_t.append(outputs)
            return outputs

        elif dataloader_idx == 2:
            probs, idx = self.process_batch(batch, "discard")
            outputs = {"probs": probs, "idx": idx}
            self.validation_step_outputs_discard.append(outputs)
            return outputs


    def on_validation_epoch_end(self):
        # import pdb; pdb.set_trace()
        outputs = (self.validation_step_outputs_s, self.validation_step_outputs_t, self.validation_step_outputs_discard)
    
        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        probs_s = probs_s.reshape(probs_s.shape[0], -1, 2)
        probs_t = probs_t.reshape(probs_t.shape[0], -1, 2)

        probs = np.concatenate((probs_s, probs_t), axis=0)
        y = np.concatenate((np.zeros_like(y_s), np.ones_like(y_t)), axis=0)
        
        disc_ce_loss = cross_entropy(torch.cat((self.val_features_s, self.val_features_t),dim=0).cpu().detach(), torch.tensor(y))
        
        

        if self.warm_start:
            loss_sum = torch.tensor([], requires_grad=True)
            for i in range(probs.shape[1]):
                loss = cross_entropy(torch.tensor(probs[:,i,:]), torch.tensor(y))
                if loss < self.best_valid_loss[i]:
                    torch.save(self.novelty_detector.state_dict(), self.model_path + "novelty_detector_model_target_recall_"+str(self.target_recalls[i])+"_"+self.mode+".pth")

#         pred_idx_s = np.argmax(probs_s, axis=1)
#         pred_idx_t = np.argmax(probs_t, axis=1)

        y_s_oracle = np.zeros_like(y_s)
        novel_inds = np.where(y_t == self.num_classes)[0]
        y_t_oracle = np.zeros_like(y_t)
        y_t_oracle[novel_inds] = 1
        true_labels = torch.cat((torch.tensor(y_s_oracle), torch.tensor(y_t_oracle)),dim=0)
        novel_ce_loss = cross_entropy(torch.cat((self.val_features_s, self.val_features_t),dim=0).cpu().detach(), true_labels)
        
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

        

        true_label_dist = get_label_dist(y_t, self.num_classes + 1)

#         pred_prob_s, pred_idx_s = np.max(probs_s, axis=1), np.argmax(probs_s, axis=1)
#         pred_prob_t, pred_idx_t  = np.max(probs_t, axis=1), np.argmax(probs_t, axis=1)
        cur_auc_true_ls = []
        recall_target_ls, fpr_ls = [], []
        for i in range(len(self.target_recalls)):
            ### IMPORTANT: notice that we put probs_t for source_probs and not prob_s.
            # This is because unlike the original use of BBE which looks for the top positive, we are looking for the top
            # negative bin.
        
            MP_estimate_BBE = 1 - BBE_estimate_binary(source_probs = probs_s[:, i, 0], target_probs = probs_t[:, i, 0])
            MP_estimate_EN = 1 - estimator_CM_EN(probs_s[:, i, 0], probs_t[:, i, 0])
            MP_estimate_dedpul = 1.0 - dedpul(np.max(probs_s, axis=1), np.max(probs_t, axis=1))
            if self.num_allowed_fp < 0.:
                self.num_allowed_fp = number_of_allowed_false_pos(len(y_s), target_p=self.target_precision,
                                                              confidence=self.precision_confidence)
            self.pure_bin_estimate[i], self.pure_MPE_threshold[i] = pure_MPE_estimator(probs_s[:, i, 1], probs_t[:, i, 1],
                                                                num_allowed_false_pos=self.num_allowed_fp)
            
            ## get the threshold required for achieving target recall and probabilities adjusted by that bias
            # logits_t = inverse_softmax(probs_t)
            # bias_for_required_recall = np.sort(logits_t[:, 1] - logits_t[:, 0])[::-1][int(self.target_recall * probs_t.shape[0])]
            # biased_logits_s = inverse_softmax(probs_s)
            # biased_logits_s[:, 1] -= 0.5*bias_for_required_recall
            # biased_logits_s[:, 0] += 0.5*bias_for_required_recall
            # biased_probs_s = softmax(torch.Tensor(biased_logits_s), dim=1).detach().cpu().numpy()

    #         log.info('num num_allowed_false_pos: {}'.format(self.num_allowed_fp))
    #         log.info('source bottom probs: {}'.format(np.sort(probs_s[:, 1])[:70]))
    #         log.info('source top probs: {}'.format(np.sort(probs_s[:, 1])[-70:]))
    #         log.info('targ top probs: {}'.format(np.sort(probs_t[:, 1])[-70:]))


#             self.log("pred_"+str(self.target_recalls[i])+"/MPE_estimate_ood" , {"pure_bin": pure_bin_estimate,
#                                             "BBE": MP_estimate_BBE,
#                                             "CM-EN": MP_estimate_EN,
# #                                             "dedpul": MP_estimate_dedpul,
#                                             "true": true_label_dist[self.num_classes]})
            self.log("pred_"+str(self.target_recalls[i])+"/MPE_estimate_ood.pure_bin", self.pure_bin_estimate[i])
            self.log("pred_"+str(self.target_recalls[i])+"/MPE_estimate_ood.BBE", MP_estimate_BBE)
            self.log("pred_"+str(self.target_recalls[i])+"/MPE_estimate_ood.CM-EN", MP_estimate_EN)
            self.log("pred_"+str(self.target_recalls[i])+"/MPE_estimate_ood.dedpul", MP_estimate_dedpul)
            self.log("pred_"+str(self.target_recalls[i])+"/MPE_estimate_ood.true", true_label_dist[self.num_classes])


            dataset_labels = np.concatenate([np.zeros_like(y_s), np.ones_like(y_t)])
            true_labels = np.concatenate([y_s_oracle, y_t_oracle])
            predictions = np.concatenate([probs_s[:,i,:], probs_t[:,i,:]])

            pred_idx_s = np.argmax(probs_s[:,i,:], axis=1)

            pred_idx_t = np.argmax(probs_t[:,i,:], axis=1)

            acc_pure_bin_threshold = np.mean(pred_idx_t == y_t_oracle)

            seen_inds = np.setdiff1d(np.arange(len(novel_inds)), novel_inds)
            recall_bin_threshold = np.sum((pred_idx_t[novel_inds]==1)) / len(novel_inds)
            
            prec_bin_threshold = np.sum(pred_idx_t[novel_inds]==1) / np.sum(pred_idx_t==1)
            # import pdb; pdb.set_trace()
            val_source_loss = log_loss(np.zeros_like(y_s), probs_s[:, i, 1], eps=1e-5, labels=[0, 1])
            # if val_source_loss > 10000000:
                # import pdb; pdb.set_trace()
            biased_val_source_loss = accuracy_score(np.zeros_like(y_s), pred_idx_s) #log_loss(np.zeros_like(y_s), biased_probs_s[:, 1], labels=[0, 1])
            recall_target = np.mean(np.argmax(probs_t[:,i,:], axis=1) == 1)
            recall_target_ls.append(recall_target)
            fpr_ls.append(1-biased_val_source_loss)
#           cur_auc_true = roc_auc_score(true_labels, predictions[:, 1])
            
            cur_auc_true = roc_auc_score(y_t_oracle, probs_t[:, i, 1])
            cur_auc_true_ls.append(cur_auc_true)
            cur_ap_true = average_precision_score(y_t_oracle, probs_t[:, i, 1])
            if not self.warm_start:
                if self.online_alpha_search:
                    if self.pure_bin_estimate >= self.cur_alpha_estimate[1]:
                        self.cur_alpha_estimate = (self.cur_alpha_estimate[0], pure_bin_estimate)
                        self.auc_roc_at_selection[i] = cur_auc_true
                        self.ap_at_selection[i] = cur_ap_true
    #               if biased_val_source_loss < self.best_source_loss and recall_target >= self.target_recall:
                if biased_val_source_loss > self.target_precision and recall_target >= self.best_candidate_alpha[i]:
                    self.best_source_loss[i] = biased_val_source_loss
                    self.best_bin_size[i] = recall_target #pure_bin_estimate
                    self.auc_roc_at_selection[i] = cur_auc_true
                    self.ap_at_selection[i] = cur_ap_true
                    self.recall_at_selection[i] = recall_bin_threshold
                    self.precision_at_selection[i] = prec_bin_threshold
                    self.acc_at_selection[i] = acc_pure_bin_threshold
                    self.best_candidate_alpha[i] = recall_target #self.cur_alpha_estimate[0]
                    self.fpr_at_selection[i] = 1 - biased_val_source_loss
                    self.recall_target_at_selection[i] = recall_target
                    wandb.log({"ROC_s_vs_t_true" : wandb.plot.roc_curve(y_t_oracle, probs_t[:,i,:],
                                                                    classes_to_plot=[1])})
                    wandb.log({"ROC_s_vs_t" : wandb.plot.roc_curve(dataset_labels, predictions,
                                                                classes_to_plot=[1])})
                    torch.save(self.novelty_detector.state_dict(), self.model_path + "novelty_detector_model.pth")
            
    #         self.log("pred_"+str(self.target_recalls[i])+"/performance", {"curr AU-ROC": cur_auc_true,
    #                                     "curr ave-precision": cur_ap_true,
    # #                                       "curr acc": acc_pure_bin_threshold,
    #                                     "val loss source": val_source_loss,
    #                                     "val loss source biased": biased_val_source_loss,
    #                                     "recall target": recall_target,
    #                                     "selected AU-ROC": self.auc_roc_at_selection,
    #                                     "selected ave-precision": self.ap_at_selection,
    #                                     "selected recall": self.recall_at_selection,
    #                                     "selected fpr": self.precision_at_selection,
    #                                     "selected acc": self.acc_at_selection,
    #                                     "selected alpha:": self.best_bin_size})

            self.log("pred_"+str(self.target_recalls[i])+"/performance.curr AU-ROC", cur_auc_true)
            self.log("pred_"+str(self.target_recalls[i])+"/performance.curr ave-precision", cur_ap_true)
            self.log("pred_"+str(self.target_recalls[i])+"/performance.val loss source", val_source_loss)
            self.log("pred_"+str(self.target_recalls[i])+"/performance.val loss source biased", biased_val_source_loss)
            self.log("pred_"+str(self.target_recalls[i])+"/performance.recall target", recall_target)
            self.log("pred_"+str(self.target_recalls[i])+"/performance.selected AU-ROC", self.auc_roc_at_selection[i])
            self.log("pred_"+str(self.target_recalls[i])+"/performance.selected ave-precision", self.ap_at_selection[i])
            self.log("pred_"+str(self.target_recalls[i])+"/performance.selected recall (MPE)", self.recall_at_selection[i])
            self.log("pred_"+str(self.target_recalls[i])+"/performance.selected recall target", self.recall_target_at_selection[i])
            self.log("pred_"+str(self.target_recalls[i])+"/performance.selected fpr", self.fpr_at_selection[i])# self.precision_at_selection)
            self.log("pred_"+str(self.target_recalls[i])+"/performance.selected acc", self.acc_at_selection[i])
            self.log("pred_"+str(self.target_recalls[i])+"/performance.selected alpha", self.best_bin_size[i])
            self.log("pred_"+str(self.target_recalls[i])+"/performance.disc cross entropy", disc_ce_loss)
            self.log("pred_"+str(self.target_recalls[i])+"/performance.novel cross entropy", novel_ce_loss)
        
        total_norm = 0
        for p in self.novelty_detector.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        log.info('recall {}'.format(recall_target_ls))
        log.info('fpr {}'.format(fpr_ls))
        if self.cmp.state.ineq_defect is not None:
            log.info('current inequality defect {}'.format(torch.sum(self.cmp.state.ineq_defect))) 
        log.info('current pure bin est {}'.format(self.pure_bin_estimate))
        log.info('current auc {}'.format(cur_auc_true_ls))
        log.info('gradient norm {}'.format(total_norm))
        

#         wandb.log({"ROC_s_vs_t_true" : wandb.plot.roc_curve(true_labels, predictions,
#                                                             classes_to_plot=[1])})
#         if self.current_epoch % 10 == 0:
#             wandb.log({"ROC_s_vs_t_true" : wandb.plot.roc_curve(y_t_oracle, probs_t,
#                                                                 classes_to_plot=[1])})
#             wandb.log({"ROC_s_vs_t" : wandb.plot.roc_curve(dataset_labels, predictions,
#                                                            classes_to_plot=[1])})

        if self.online_alpha_search:
            alpha_upper_bound = 1. if self.upper_bound_alpha[0] is None else self.upper_bound_alpha[0]
            # self.log("train/alpha_search", {"cur_search_candidate": self.cur_alpha_estimate[0],
            #                                 "cur_lower_bound": self.lower_bound_alpha[0],
            #                                 "cur_upper_bound": alpha_upper_bound}
            #         )
            self.log("train/alpha_search.curr_search_candidate", self.cur_alpha_estimate[0])
            self.log("train/alpha_search.cur_lower_bound", self.lower_bound_alpha[0])
            self.log("train/alpha_search.cur_upper_bound", alpha_upper_bound)
#         train_probs = torch.cat([x["probs"] for x in outputs[2]]).detach().cpu().numpy()
#         train_idx = torch.cat([x["idx"] for x in outputs[2]]).detach().cpu().numpy()
        ## LOOKS LIKE THERES A BUG HERE!!
#         self.keep_samples = keep_samples_discriminator(train_probs, train_idx, self.pure_bin_estimate)

        log_everything(self.logging_file, epoch=self.current_epoch,\
#             val_acc=np.array(),\ ##Continue from here!!!
            auc=cur_auc_true, val_acc=acc_pure_bin_threshold, mpe = np.array([self.pure_bin_estimate, MP_estimate_BBE, \
                                                                              MP_estimate_EN]) ,\
            true_mp = true_label_dist[-1],
            selected_mpe = self.best_bin_size, selected_auc = self.auc_roc_at_selection,
            selected_acc = self.acc_at_selection, selected_recall = self.recall_at_selection,
            selected_prec = self.precision_at_selection)

        
        self.validation_step_outputs_s = []
        self.validation_step_outputs_t = []
        self.validation_step_outputs_discard = []
        self.val_features_s = torch.tensor([], device=self._device)
        self.val_features_t = torch.tensor([], device=self._device)
#         torch.save(self.novelty_detector.state_dict(), self.model_path + "novelty_detection_model.pth")

#
#
#         pred_prob_s, pred_idx_s = np.max(probs_s, axis=1), np.argmax(probs_s, axis=1)
#         pred_prob_t, pred_idx_t  = np.max(probs_t, axis=1), np.argmax(probs_t, axis=1)
#
#
#         seen_idx = np.where(y_t < self.num_classes)[0]
#         ood_idx = np.where(y_t == self.num_classes)[0]
#
#         estimate_source_label_dist = self.MP_estimate[:self.num_classes]/np.sum(self.MP_estimate[:self.num_classes])
#
#         resample_idx = resample_probs(disc_probs_s, y_s, estimate_source_label_dist)
#
#         resample_disc_probs_s = disc_probs_s[resample_idx]
#
#         MPE_estimate_disc = BBE_estimate_binary(source_probs= resample_disc_probs_s,\
#             target_probs= disc_probs_t)
#
#         self.estimate_ood_alpha = 1.0 - MPE_estimate_disc
#
#         self.log(f"pred/MPE_ood", { "source_classifier" : self.MP_estimate[self.num_classes], \
#             "discriminator": 1.0 - MPE_estimate_disc,\
#             "true": true_label_dist[self.num_classes]} )
#
#         self.MP_estimate[:self.num_classes] = (self.MP_estimate[:self.num_classes]/np.sum(self.MP_estimate[:self.num_classes]))*MPE_estimate_disc
#
#         self.MP_estimate[self.num_classes] = 1.0 - MPE_estimate_disc
#
#         # for i in range(self.num_classes):
#         #     self.log(f"pred/MPE_class_{i}", { "estimate" : self.MP_estimate[i], "true": true_label_dist[i] } )
#
#
#         target_seen_acc = np.mean(pred_idx_t[seen_idx] == y_t[seen_idx])
#         source_seen_acc = np.mean(pred_idx_s== y_s)
#
#
#         self.log("pred/target_seen_acc", target_seen_acc)
#         self.log("pred/source_seen_acc", source_seen_acc)
#
#         ### OOD precision and recall
#
#         pred_idx = (disc_probs_t < 0.5)
#
#         ood_recall = np.sum((pred_idx[ood_idx] ==1)) / len(ood_idx)
#         ood_precision = np.sum((pred_idx[ood_idx]==1)) / np.sum(pred_idx ==1)
#
#         self.log("pred/ood_recall", ood_recall)
#         self.log("pred/ood_precision", ood_precision)
#
#         ### Domain discrimimation accuracy
#
#         acc_source_domain_disc = np.mean(disc_probs_s > 0.5)
#         acc_target_domain_disc = np.mean(disc_probs_t <= 0.5)
#
#         domain_disc_valid_acc = 2*(1.0 - self.estimate_ood_alpha)*acc_source_domain_disc + acc_target_domain_disc - (1.0 - self.estimate_ood_alpha)
#
#         domain_disc_accuracy = (acc_source_domain_disc + acc_target_domain_disc)/2
#         if self.current_epoch >=4 and domain_disc_accuracy >= self.best_domain_acc and self.reload_model:
#             self.best_domain_acc = domain_disc_accuracy
#             torch.save(self.discriminator_model.state_dict(), self.model_path + "discriminator_model.pth")
#
#
#         self.log("pred/domain_disc_acc", domain_disc_accuracy)
#         self.log("pred/domain_disc_valid_est", domain_disc_valid_acc)
#
#         ### Overall accruacy
#
#         ood_pred_idx = np.where(disc_probs_t < 0.5)[0]
#         seen_pred_idx = np.where(disc_probs_t >= 0.5)[0]
#
#         calibrator = VectorScaling()(inverse_softmax(probs_s), idx2onehot(y_s, self.num_classes))
#         calib_pred_prob_t = calibrator(inverse_softmax(probs_t))
#
#         label_shift_corrected_prob_t = label_shift_correction(calib_pred_prob_t, estimate_source_label_dist)
#
#         label_shift_corrected_pred_t = np.argmax(label_shift_corrected_prob_t, axis=1)
#
#         label_shift_preds = np.concatenate([label_shift_corrected_pred_t[seen_pred_idx], [self.num_classes] * len(ood_pred_idx)])
#         label_shift_y = np.concatenate([y_t[seen_pred_idx], y_t[ood_pred_idx]])
#
#         label_shift_corrected_acc = np.mean(label_shift_preds == label_shift_y)
#
#         # target_seen_acc_label_shift = np.mean(label_shift_preds[:len(seen_pred_idx)] == label_shift_y[:len(seen_pred_idx)])
#
#         target_seen_acc_label_shift =  np.mean(label_shift_corrected_pred_t[seen_idx] == y_t[seen_idx])
#
#         self.log("pred/label_shift_corrected_acc", label_shift_corrected_acc)
#
#         orig_preds = np.concatenate([pred_idx_t[seen_pred_idx], [self.num_classes] * len(ood_pred_idx)])
#
#         orig_acc = np.mean(orig_preds == label_shift_y)
#
#         combined_probs_t = np.zeros((probs_t.shape[0], probs_t.shape[1]+1))
#
#         combined_probs_t[:, :-1] = probs_t*(np.expand_dims(disc_probs_t, axis=1))
#         combined_probs_t[:, -1] = (1.0 - disc_probs_t)
#
#         combined_pred_t = np.argmax(combined_probs_t, axis=1)
#
#         combined_acc = np.mean(combined_pred_t == y_t)
#
#         self.log("pred/orig_acc", orig_acc)
#         self.log("pred/combined_orig_acc", combined_acc)
#
#         torch.save(self.source_model.state_dict(), self.model_path + "source_model.pth")
#         ### Update keep samples from outputs[1]
#
#         train_probs = torch.cat([x["probs"] for x in outputs[2]]).detach().cpu().numpy()
#         train_idx = torch.cat([x["idx"] for x in outputs[2]]).detach().cpu().numpy()
#
#         self.keep_samples = keep_samples_discriminator(train_probs, train_idx, self.estimate_ood_alpha) self.pure_bin_estimate
#
#         log_everything(self.logging_file, epoch=self.current_epoch,\
#             target_label_shift_acc=label_shift_corrected_acc, target_orig_acc= orig_acc,\
#             target_seen_label_acc= target_seen_acc_label_shift, target_seen_acc=target_seen_acc, source_acc =source_seen_acc,\
#             precision=ood_precision, recall=ood_recall, domain_disc_acc= domain_disc_accuracy, domain_disc_valid_acc= domain_disc_valid_acc, \
#             target_marginal_estimate = self.MP_estimate, target_marginal = true_label_dist)

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