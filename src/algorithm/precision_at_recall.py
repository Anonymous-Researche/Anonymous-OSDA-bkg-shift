import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot, cross_entropy

from typing import List, Optional
from src.model_utils import *
from src.MPE_methods.dedpul import dedpul
import logging
import wandb
from src.core_utils import *
from abstention.calibration import  VectorScaling
import os

import src.algorithm.constrained_optimization as constrained_optimization
from src.algorithm.constrained_optimization.problem import ConstrainedMinimizationProblem
from src.algorithm.constrained_optimization.lagrangian_formulation import LagrangianFormulation
from src.algorithm.constrained_optimization.optim import *
from src.algorithm.constrained_optimization.constrained_optimizer import ConstrainedOptimizer
from src.algorithm.constrained_optimization.problem import CMPState
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, average_precision_score

log = logging.getLogger("app")

class RecallConstrainedClassification(ConstrainedMinimizationProblem):
    def __init__(self, target_recall=0.1, wd=0., penalty_type='l2', logit_multiplier=2.):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.target_recall = target_recall
        self.wd = wd
        self.penalty_type = penalty_type
        self.logit_multiplier = logit_multiplier
        super().__init__(is_constrained=True)

    def get_penalty(self, model):
        penalty_lambda = self.wd
        if self.penalty_type == 'l2':
            penalty_term = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            penalty_term = sum(torch.abs(p).sum() for p in model.parameters())
        return penalty_lambda*penalty_term

    def closure(self, model, inputs, targets):
        pred_logits = model.forward(inputs)
        with torch.no_grad():
            predictions = torch.argmax(pred_logits, dim=1)
        penalty = self.get_penalty(model)
        cross_ent = self.criterion(pred_logits[targets==0], targets[targets==0])
        loss = cross_ent + penalty # 0.1*cross_ent + penalty
        recall, recall_proxy, preds_temp, positives_temp = recall_from_logits(self.logit_multiplier*pred_logits,
                                                                              targets)

        ineq_defect = self.target_recall - recall
        proxy_ineq_defect = self.target_recall - recall_proxy

        return CMPState(loss=loss, ineq_defect=ineq_defect, proxy_ineq_defect=proxy_ineq_defect,
                        eq_defect=None, misc={'cross_ent': cross_ent, 'recall_proxy': recall_proxy})

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
        pretrained_model_dir: Optional[str] = None
    ):
        super().__init__()
        self.num_classes = num_source_classes

        self.num_outputs = 2
        self.dataset = dataset
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pretrained_model_dir = pretrained_model_dir
        self.penalty_type = penalty_type

        self.novelty_detector, self.primal_optimizer = get_model(arch, self.dataset, self.num_outputs, pretrained= self.pretrained,
                                                                 learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                                                                 pretrained_model_dir=self.pretrained_model_dir)
        self.target_precision = target_precision
        self.precision_confidence = precision_confidence
        self.target_recall = target_recall
        self.dual_optimizer = constrained_optimization.optim.partial_optimizer(torch.optim.SGD, lr=dual_learning_rate)

        self.cmp = RecallConstrainedClassification(target_recall=self.target_recall, wd=self.weight_decay,
                                                   penalty_type=self.penalty_type, logit_multiplier=logit_multiplier)
        self.formulation = LagrangianFormulation(self.cmp, ineq_init = torch.tensor(1)) ## start from here tomorrow!!
        self.coop = ConstrainedOptimizer(
            formulation=self.formulation,
            primal_optimizer=self.primal_optimizer,
            dual_optimizer=self.dual_optimizer,
        )

        self.max_epochs = max_epochs

        self.warmup_epochs = warmup_epochs

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Some variables for the alpha line search
        self.online_alpha_search = online_alpha_search
        self.alpha_search_midpoint = None
        self.epochs_since_alpha_update = 0.
        self.epochs_for_each_alpha = epochs_for_each_alpha
        self.pure_bin_estimate = 0.
        self.best_bin_size = 0.
        self.best_candidate_alpha = 0.
        self.best_source_loss = 1000.
        self.auc_roc_at_selection = 0.
        self.ap_at_selection = 0.
        self.precision_at_selection = 0.
        self.recall_at_selection = 0.
        self.recall_target_at_selection = 0.
        self.fpr_at_selection = 1.
        self.acc_at_selection = 0.
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

        self.model_path = "./models/"

        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.warm_start = False if self.warmup_epochs == 0 else True
        self.reload_model = False

        self.automatic_optimization = False

    def update_alpha_search_params(self):
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
        self.target_recall = target_recall
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

    def forward(self, x):
        return self.novelty_detector(x)

    def process_batch(self, batch, stage="train"):

        if stage == "train":
            x_s, y_s, _ = batch["source_full"][:3]
            x_t, y_t, idx_t = batch["target_full"][:3]

            x = torch.cat([x_s, x_t], dim=0)
            y = torch.cat([torch.zeros_like(y_s), torch.ones_like(y_t)], dim=0)

            if self.warm_start:
                logits_detector = self.novelty_detector(x)
                loss = cross_entropy(logits_detector, y)
                self.primal_optimizer.zero_grad()
                self.manual_backward(loss)
                self.primal_optimizer.step()

                return loss, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
            else:
                lagrangian = self.formulation.composite_objective(
                  self.cmp.closure, self.novelty_detector, x, y
                )
                self.formulation.custom_backward(lagrangian)
                self.coop.step(self.cmp.closure, self.novelty_detector, x, y)
#                 print(self.cmp.state)
#                 print(self.formulation.cmp.is_constrained)
#                 print(self.formulation.weighted_violation(self.cmp.state, "ineq"))
#                 print('lag val after {}'.format(self.formulation.composite_objective(
#                   self.cmp.closure, self.novelty_detector, x, y
#                 )))
                return self.cmp.state.misc['cross_ent'], self.cmp.get_penalty(self.novelty_detector), self.cmp.state.ineq_defect, lagrangian

            if self.trainer.is_last_batch:
                update_optimizer(self.current_epoch, self.primal_optimizer, self.dataset, self.learning_rate)

            return loss2, self.cmp.get_penalty(self.novelty_detector), self.cmp.state.ineq_defect, torch.tensor(0.)

        elif stage == "pred_source":
            x_s, y_s, _ = batch[:3]

            logits = self.novelty_detector(x_s)
            probs_s = softmax(logits, dim=1)
#             disc_probs_s = probs
#
#             logits_s = self.source_model(x_s)
#             probs_s = softmax(logits_s, dim=1)
            return probs_s, y_s

        elif stage == "pred_target":

            x_t, y_t, _ = batch[:3]
            logits = self.novelty_detector(x_t)
            probs_t = softmax(logits, dim=1)
#             disc_probs_t = probs
#
#             logits_t = self.source_model(x_t)
#             probs_t = softmax(logits_t, dim=1)
            return probs_t, y_t

        elif stage == "discard":

            x_t, _, idx_t  = batch[:3]
            logits = self.novelty_detector(x_t)
            probs = softmax(logits, dim = 1)[:,1]

            return probs, idx_t

        else:
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss, penalty, ineq_defect, lagrangian_value = self.process_batch(batch, "train")

        self.log("train/loss", {"cross_ent": loss, "constraint_penalty": penalty, "lagrangian": lagrangian_value},
                 on_step=True, on_epoch=True, prog_bar=False)
        if not self.warm_start:
            self.log("train/constraints", {"inequality_violation": ineq_defect,
                                           "multiplier_value": self.formulation.ineq_multipliers.weight.detach().cpu()}, #, "recall_proxy": recall_proxy
                     on_step=True, on_epoch=True, prog_bar=False)

        return  {"lagrangian_loss": lagrangian_value.detach()} #{"source_loss": loss1.detach(), "discriminator_loss": loss2.detach()}

    def training_epoch_end(self, outputs):
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
            return {"probs_s": probs_s, "y_s": y_s}#, "disc_probs_s": disc_probs_s }

        elif dataloader_idx == 1:
            probs_t, y_t = self.process_batch(batch, "pred_target")
            return {"probs_t": probs_t, "y_t": y_t}#, "disc_probs_t": disc_probs_t}

        elif dataloader_idx == 2:
            probs, idx = self.process_batch(batch, "discard")
            return {"probs": probs, "idx": idx}


    def validation_epoch_end(self, outputs):

        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

#         pred_idx_s = np.argmax(probs_s, axis=1)
#         pred_idx_t = np.argmax(probs_t, axis=1)

        y_s_oracle = np.zeros_like(y_s)
        novel_inds = np.where(y_t == self.num_classes)[0]
        y_t_oracle = np.zeros_like(y_t)
        y_t_oracle[novel_inds] = 1

        true_label_dist = get_label_dist(y_t, self.num_classes + 1)

#         pred_prob_s, pred_idx_s = np.max(probs_s, axis=1), np.argmax(probs_s, axis=1)
#         pred_prob_t, pred_idx_t  = np.max(probs_t, axis=1), np.argmax(probs_t, axis=1)

        ### IMPORTANT: notice that we put probs_t for source_probs and not prob_s.
        # This is because unlike the original use of BBE which looks for the top positive, we are looking for the top
        # negative bin.
        MP_estimate_BBE = 1 - BBE_estimate_binary(source_probs = probs_s[:, 0], target_probs = probs_t[:, 0])
        MP_estimate_EN = 1 - estimator_CM_EN(probs_s[:, 0], probs_t[:, 0])
#         MP_estimate_dedpul = 1.0 - dedpul(np.max(probs_s, axis=1), np.max(probs_t, axis=1))
        if self.num_allowed_fp < 0.:
            self.num_allowed_fp = number_of_allowed_false_pos(len(y_s), target_p=self.target_precision,
                                                              confidence=self.precision_confidence)
        pure_bin_estimate, pure_MPE_threshold = pure_MPE_estimator(probs_s[:, 1], probs_t[:, 1],
                                                                   num_allowed_false_pos=self.num_allowed_fp)

        ## get the threshold required for achieving target recall and probabilities adjusted by that bias
        logits_t = inverse_softmax(probs_t)
        bias_for_required_recall = np.sort(logits_t[:, 1] - logits_t[:, 0])[::-1][int(self.target_recall * probs_t.shape[0])]
        biased_logits_s = inverse_softmax(probs_s)
        biased_logits_s[:, 1] -= 0.5*bias_for_required_recall
        biased_logits_s[:, 0] += 0.5*bias_for_required_recall
        biased_probs_s = softmax(torch.Tensor(biased_logits_s), dim=1).detach().cpu().numpy()

#         log.info('num num_allowed_false_pos: {}'.format(self.num_allowed_fp))
#         log.info('source bottom probs: {}'.format(np.sort(probs_s[:, 1])[:70]))
#         log.info('source top probs: {}'.format(np.sort(probs_s[:, 1])[-70:]))
#         log.info('targ top probs: {}'.format(np.sort(probs_t[:, 1])[-70:]))


        self.log("pred/MPE_estimate_ood" , {"pure_bin": pure_bin_estimate,
                                            "BBE": MP_estimate_BBE,
                                            "CM-EN": MP_estimate_EN,
#                                             "dedpul": MP_estimate_dedpul,
                                            "true": true_label_dist[self.num_classes]})


        dataset_labels = np.concatenate([np.zeros_like(y_s), np.ones_like(y_t)])
        true_labels = np.concatenate([y_s_oracle, y_t_oracle])
        predictions = np.concatenate([probs_s, probs_t])

        pred_idx_s = np.argmax(probs_s, axis=1)

        pred_idx_t = np.argmax(probs_t, axis=1)

        acc_pure_bin_threshold = np.mean(pred_idx_t == y_t_oracle)

        seen_inds = np.setdiff1d(np.arange(len(novel_inds)), novel_inds)
        recall_bin_threshold = np.sum((pred_idx_t[novel_inds]==1)) / len(novel_inds)
        prec_bin_threshold = np.sum(pred_idx_t[novel_inds]==1) / np.sum(pred_idx_t==1)

        val_source_loss = log_loss(np.zeros_like(y_s), probs_s[:, 1], labels=[0, 1])
        biased_val_source_loss = accuracy_score(np.zeros_like(y_s), pred_idx_s) #log_loss(np.zeros_like(y_s), biased_probs_s[:, 1], labels=[0, 1])
        recall_target = np.mean(np.argmax(probs_t, axis=1) == 1)
#         cur_auc_true = roc_auc_score(true_labels, predictions[:, 1])
        cur_auc_true = roc_auc_score(y_t_oracle, probs_t[:, 1])
        cur_ap_true = average_precision_score(y_t_oracle, probs_t[:, 1])
        if not self.warm_start:
            if self.online_alpha_search:
                if pure_bin_estimate >= self.cur_alpha_estimate[1]:
                    self.cur_alpha_estimate = (self.cur_alpha_estimate[0], pure_bin_estimate)
                    self.auc_roc_at_selection = cur_auc_true
                    self.ap_at_selection = cur_ap_true
#             if biased_val_source_loss < self.best_source_loss and recall_target >= self.target_recall:
            if biased_val_source_loss > self.target_precision and recall_target >= self.best_candidate_alpha:
                self.best_source_loss = biased_val_source_loss
                self.best_bin_size = recall_target #pure_bin_estimate
                self.auc_roc_at_selection = cur_auc_true
                self.ap_at_selection = cur_ap_true
                self.recall_at_selection = recall_bin_threshold
                self.precision_at_selection = prec_bin_threshold
                self.acc_at_selection = acc_pure_bin_threshold
                self.fpr_at_selection = 1 - biased_val_source_loss
                self.recall_target_at_selection = recall_target
                self.best_candidate_alpha = recall_target #self.cur_alpha_estimate[0]
                wandb.log({"ROC_s_vs_t_true" : wandb.plot.roc_curve(y_t_oracle, probs_t,
                                                                    classes_to_plot=[1])})
                wandb.log({"ROC_s_vs_t" : wandb.plot.roc_curve(dataset_labels, predictions,
                                                               classes_to_plot=[1])})
                torch.save(self.novelty_detector.state_dict(), self.model_path + "novelty_detector_model.pth")

        self.log("pred/performance", {"curr AU-ROC": cur_auc_true,
                                      "curr ave-precision": cur_ap_true,
#                                       "curr acc": acc_pure_bin_threshold,
                                      "val loss source": val_source_loss,
                                      "val loss source biased": biased_val_source_loss,
                                      "recall target": recall_target,
                                      "selected AU-ROC": self.auc_roc_at_selection,
                                      "selected ave-precision": self.ap_at_selection,
                                      "selected recall": self.recall_at_selection,
                                      "selected recall target": self.recall_target_at_selection
                                      "selected fpr": self.fpr_at_selection # self.precision_at_selection,
                                      "selected acc": self.acc_at_selection,
                                      "selected alpha:": self.best_bin_size})

        log.info('recall {}'.format(recall_target))
        log.info('fpr {}'.format(1-biased_val_source_loss))
        log.info('current inequality defect {}'.format(self.cmp.state.ineq_defect))
        log.info('current pure bin est {}'.format(pure_bin_estimate))
        log.info('current auc {}'.format(cur_auc_true))

#         wandb.log({"ROC_s_vs_t_true" : wandb.plot.roc_curve(true_labels, predictions,
#                                                             classes_to_plot=[1])})
#         if self.current_epoch % 10 == 0:
#             wandb.log({"ROC_s_vs_t_true" : wandb.plot.roc_curve(y_t_oracle, probs_t,
#                                                                 classes_to_plot=[1])})
#             wandb.log({"ROC_s_vs_t" : wandb.plot.roc_curve(dataset_labels, predictions,
#                                                            classes_to_plot=[1])})

        if self.online_alpha_search:
            alpha_upper_bound = 1. if self.upper_bound_alpha[0] is None else self.upper_bound_alpha[0]
            self.log("train/alpha_search", {"cur_search_candidate": self.cur_alpha_estimate[0],
                                            "cur_lower_bound": self.lower_bound_alpha[0],
                                            "cur_upper_bound": alpha_upper_bound}
                    )
#         train_probs = torch.cat([x["probs"] for x in outputs[2]]).detach().cpu().numpy()
#         train_idx = torch.cat([x["idx"] for x in outputs[2]]).detach().cpu().numpy()
        ## LOOKS LIKE THERES A BUG HERE!!
#         self.keep_samples = keep_samples_discriminator(train_probs, train_idx, self.pure_bin_estimate)

        log_everything(self.logging_file, epoch=self.current_epoch,\
#             val_acc=np.array(),\ ##Continue from here!!!
            auc=cur_auc_true, val_acc=acc_pure_bin_threshold, mpe = np.array([pure_bin_estimate, MP_estimate_BBE, \
                                                                              MP_estimate_EN]) ,\
            true_mp = true_label_dist[-1],
            selected_mpe = self.best_bin_size, selected_auc = self.auc_roc_at_selection,
            selected_acc = self.acc_at_selection, selected_recall = self.recall_at_selection,
            selected_prec = self.precision_at_selection)

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

    def configure_optimizers(self):

        return [self.primal_optimizer]