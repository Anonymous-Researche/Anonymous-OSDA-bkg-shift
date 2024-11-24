import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot, cross_entropy

from typing import List, Optional
from src.model_utils import *
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
from sklearn.metrics import roc_auc_score

log = logging.getLogger("app")

class FPRConstrainedClassification(ConstrainedMinimizationProblem):
    def __init__(self, target_fpr=0.1, wd=0., penalty_type='l2'):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.target_fpr = target_fpr
        self.wd = wd
        self.penalty_type = penalty_type
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
        cross_ent = self.criterion(pred_logits[targets==1], targets[targets==1])
        loss = cross_ent + penalty # 0.1*cross_ent + penalty
        fpr, fpr_proxy, preds_temp, negatives_temp = fpr_from_logits(0.5*pred_logits, targets)

        # We want each row of W to have norm less than or equal to 1
        # g(W) >= alpha  ---> alpha - g(W) <= 0
        ineq_defect = fpr - self.target_fpr #torch.Tensor([0.])
        proxy_ineq_defect = fpr_proxy - self.target_fpr #torch.Tensor([0.])

        return CMPState(loss=loss, ineq_defect=ineq_defect, proxy_ineq_defect=proxy_ineq_defect,
                        eq_defect=None, misc={'cross_ent': cross_ent, 'fpr_proxy': fpr_proxy})

class TrainRAtF(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_source_classes: int = 10,
        dataset: str = "CIFAR10",
        learning_rate: float = 0.1,
        dual_learning_rate: float = 2e-2,
        target_fpr: float = 0.01,
        target_precision: float = 0.99,
        precision_confidence: float = 0.9,
        weight_decay: float = 1e-4,
        penalty_type: float = 'l2',
        max_epochs: int = 500,
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

        self.num_outputs = 2 #self.num_classes
        self.dataset = dataset

        self.novelty_detector, self.primal_optimizer = get_model(arch, dataset, self.num_outputs, pretrained= pretrained, \
                            learning_rate= learning_rate, weight_decay=weight_decay,  pretrained_model_dir= pretrained_model_dir)
        self.target_precision = target_precision
        self.precision_confidence = precision_confidence
        self.target_fpr = target_fpr
        self.dual_optimizer = constrained_optimization.optim.partial_optimizer(torch.optim.SGD, lr=dual_learning_rate)

        self.cmp = FPRConstrainedClassification(target_fpr=self.target_fpr, wd=weight_decay, penalty_type=penalty_type)
        self.formulation = LagrangianFormulation(self.cmp)#, ineq_init = torch.Tensor([0.1])) ## start from here tomorrow!!
        self.coop = ConstrainedOptimizer(
            formulation=self.formulation,
            primal_optimizer=self.primal_optimizer,
            dual_optimizer=self.dual_optimizer,
        )

        self.max_epochs = max_epochs

        self.warmup_epochs = self.max_epochs - 1

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.estimate_ood_alpha = 0.5
        self.pure_bin_estimate = 0.
        self.best_bin_size = 0.
        self.num_allowed_fp = -1

        self.pred_save_path = f"{pred_save_path}/{dataset}/"

        self.logging_file = f"{self.pred_save_path}/RAtF_{arch}_{num_source_classes}_{seed}_log_update.txt"

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

        self.warm_start = True
        self.keep_samples = None
        self.reload_model = True

        self.automatic_optimization = False

        self.best_domain_acc = 0.0

    def forward(self, x):
        return self.novelty_detector(x)

    def process_batch(self, batch, stage="train"):

        if stage == "train":
            x_s, y_s, _ = batch["source_full"][:3]
            x_t, y_t, idx_t = batch["target_full"][:3]

            x = torch.cat([x_s, x_t], dim=0)
            y = torch.cat([torch.zeros_like(y_s), torch.ones_like(y_t)], dim=0)

            if self.warm_start:
                logits_novelty = self.forward(x_s)
                lagrangian = self.formulation.composite_objective(
                  self.cmp.closure, self.novelty_detector, x, y
                )
                self.formulation.custom_backward(lagrangian)
                self.coop.step(self.cmp.closure, self.novelty_detector, x, y)

                return self.cmp.state.misc['cross_ent'], self.cmp.get_penalty(self.novelty_detector), self.cmp.state.ineq_defect, self.cmp.state.misc['fpr_proxy']
            else:
                ## This is the CVIR loss I suppose.
                logits_novelty = self.forward(x)
                keep_idx = np.concatenate([np.arange(len(y_s), dtype = np.int32), \
                    len(y_s) + np.where(self.keep_samples[idx_t.cpu().numpy()] == 1)[0]], axis=0)
                loss2 = cross_entropy(logits_novelty[keep_idx], y[keep_idx],\
                    weight=torch.Tensor([1.0 - self.pure_bin_estimate, self.pure_bin_estimate]).to(self.device))
                self.primal_optimizer.zero_grad()
                self.manual_backward(loss2)
                self.primal_optimizer.step()

            if self.trainer.is_last_batch:
                update_optimizer(self.current_epoch, self.primal_optimizer, self.dataset, self.learning_rate)

            return loss2, self.cmp.get_penalty(self.novelty_detector), self.cmp.state.ineq_defect, self.cmp.state.misc['fpr_proxy']

        elif stage == "pred_source":
            x_s, y_s, _ = batch[:3]

            logits = self.novelty_detector(x_s)
            probs = softmax(logits, dim=1)

            return probs, y_s

        elif stage == "pred_disc":

            x_t, y_t, _ = batch[:3]
            logits = self.novelty_detector(x_t)
            probs = softmax(logits, dim=1)

            return probs, y_t

        elif stage == "discard":

            x_t, _, idx_t  = batch[:3]
            logits = self.novelty_detector(x_t)
            probs = softmax(logits, dim = 1)[:,1]

            return probs, idx_t

        else:
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss, penalty, ineq_defect, fpr_proxy = self.process_batch(batch, "train")

        lagrangian_value = loss - penalty
        self.log("train/loss", {"cross_ent": loss, "constraint_penalty": penalty, "lagrangian": lagrangian_value},
                 on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/constraints", {"inequality_violation": ineq_defect, "fpr_proxy": fpr_proxy},
                 on_step=True, on_epoch=True, prog_bar=False)

        return  {"lagrangian_loss": lagrangian_value.detach()} #{"source_loss": loss1.detach(), "discriminator_loss": loss2.detach()}

    def training_epoch_end(self, outputs):
        if self.current_epoch < self.warmup_epochs:
            self.warm_start = True
        else:
            if self.reload_model:
                self.novelty_detector.load_state_dict(torch.load(self.model_path + "novelty_detection_model.pth"))
                self.warm_start = False
                self.reload_model = False

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        if dataloader_idx == 0:
            probs_s, y_s = self.process_batch(batch, "pred_source")

            return {"probs_s": probs_s, "y_s": y_s}#, "disc_probs_s": disc_probs_s }

        elif dataloader_idx == 1:
            probs_t, y_t = self.process_batch(batch, "pred_disc")

            return {"probs_t": probs_t, "y_t": y_t}#, "disc_probs_t": disc_probs_t}

        elif dataloader_idx == 2:
            probs, idx = self.process_batch(batch, "discard")
            return {"probs": probs, "idx": idx}


    def validation_epoch_end(self, outputs):


        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        y_s_oracle = np.zeros_like(y_s)
        novel_inds = np.where(y_t == self.num_classes)[0]
        y_t_oracle = np.zeros_like(y_t)
        y_t_oracle[novel_inds] = 1

        true_label_dist = get_label_dist(y_t, self.num_classes + 1)

        ### IMPORTANT: notice that we put probs_t for source_probs and not prob_s.
        # This is because unlike the original use of BBE which looks for the top positive, we are looking for the top
        # negative bin.
        self.MP_estimate = 1 - BBE_estimate_binary(source_probs = probs_t[:, 0], target_probs = probs_s[:, 0])
        if self.num_allowed_fp < 0.:
            self.num_allowed_fp = number_of_allowed_false_pos(len(y_s), target_p=self.target_precision,
                                                              confidence=self.precision_confidence)
        self.pure_bin_estimate = pure_MPE_estimator(probs_s[:, 1], probs_t[:, 1],
                                                    num_allowed_false_pos=self.num_allowed_fp)
        log.info('num num_allowed_false_pos: {}'.format(self.num_allowed_fp))
        log.info('source top probs: {}'.format(np.sort(probs_s[:, 1])[-70:]))
        log.info('targ top probs: {}'.format(np.sort(probs_t[:, 1])[-70:]))

        self.log("pred/MPE_estimate_ood" , {"pure_bin": self.pure_bin_estimate,
                                            "BBE_neg": self.MP_estimate,
                                            "true": true_label_dist[self.num_classes]})

        dataset_labels = np.concatenate([np.zeros_like(y_s), np.ones_like(y_t)])
        true_labels = np.concatenate([y_s_oracle, y_t_oracle])
        predictions = np.concatenate([probs_s, probs_t])
        cur_auc_true = roc_auc_score(true_labels, predictions[:, 1])
        self.log("pred/preformance", {"AU-ROC": cur_auc_true})

        wandb.log({"ROC_s_vs_t_true" : wandb.plot.roc_curve(true_labels, predictions,
                                                            classes_to_plot=[1])})
        wandb.log({"ROC_s_vs_t" : wandb.plot.roc_curve(dataset_labels, predictions,
                                                       classes_to_plot=[1])})

        train_probs = torch.cat([x["probs"] for x in outputs[2]]).detach().cpu().numpy()
        train_idx = torch.cat([x["idx"] for x in outputs[2]]).detach().cpu().numpy()

        self.keep_samples = keep_samples_discriminator(train_probs, train_idx, self.pure_bin_estimate)

        if self.current_epoch >=4 and self.pure_bin_estimate >= self.best_bin_size and self.reload_model:
            self.best_bin_size = self.pure_bin_estimate
            torch.save(self.novelty_detector.state_dict(), self.model_path + "novelty_detection_model.pth")

    def configure_optimizers(self):

        return [self.primal_optimizer]