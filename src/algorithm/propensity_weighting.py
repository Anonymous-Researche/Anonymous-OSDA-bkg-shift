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
from abstention.calibration import VectorScaling
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import src.algorithm.constrained_optimization as constrained_optimization
from src.algorithm.constrained_optimization.problem import ConstrainedMinimizationProblem
from src.algorithm.constrained_optimization.lagrangian_formulation import LagrangianFormulation
from src.algorithm.constrained_optimization.optim import *
from src.algorithm.constrained_optimization.constrained_optimizer import ConstrainedOptimizer
from src.algorithm.constrained_optimization.problem import CMPState
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, average_precision_score

log = logging.getLogger("app")

class TrainPropensityWeighting(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_source_classes: int = 10,
        dataset: str = "CIFAR10",
        learning_rate: float = 0.1,
        target_precision: float = 0.99,
        precision_confidence: float = 0.95,
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

        self.novelty_detector, self.detector_optimizer = get_model(arch, dataset, self.num_outputs, pretrained= pretrained, \
                            learning_rate= learning_rate, weight_decay= weight_decay,  pretrained_model_dir= pretrained_model_dir)
        self.ratio_estimator, self.ratio_optimizer = get_model(arch, dataset, self.num_outputs, pretrained= pretrained, \
                            learning_rate= learning_rate, weight_decay= weight_decay,  pretrained_model_dir= pretrained_model_dir)

        self.max_epochs = max_epochs
        self.density_estimation_epochs = self.max_epochs // 2

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.pure_bin_estimate = 0.
#         self.best_acc_dataset_labels = 0.
        self.best_weighted_score = -10.
        self.best_nonweighted_score = 0.
        self.auc_roc_at_selection = 0.
        self.ap_at_selection = 0.
        self.acc_at_selection = 0.
        self.num_allowed_fp = -1
        self.target_precision = target_precision
        self.precision_confidence = precision_confidence
        self.mpe_at_selection = 0.
        self.sample_weights_train = np.array([])
        self.sample_weights_val = np.array([])
        self.validation_step_outputs_s = []
        self.validation_step_outputs_t = []
        self.validation_step_outputs_reweight = []

        self.pred_save_path = f"{pred_save_path}/{dataset}/"

        self.logging_file = f"{self.pred_save_path}/PropensityWeighting_{arch}_{num_source_classes}_{seed}_log_update.txt"

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
        self.reload_model = False

        self.automatic_optimization = False

        self.best_domain_acc = 0.0
        self.best_cross_ent_val = 1000.0

    def forward(self, x):
        return self.novelty_detector(x)

    def process_batch(self, batch, stage="train"):

        if stage == "train":
            x_s, y_s, idx_s = batch["source_full"][:3]
            x_t, y_t, _ = batch["target_full"][:3]

            detector_optimizer, ratio_optimizer = self.optimizers()

            x = torch.cat([x_s, x_t], dim=0)
            y = torch.cat([torch.zeros_like(y_s), torch.ones_like(y_t)], dim=0)

            if self.warm_start:
                # Don't use propensity weights during warm start
                logits_ratio = self.ratio_estimator(x)
                loss = cross_entropy(logits_ratio, y)
                ratio_optimizer.zero_grad()
                self.manual_backward(loss)
                ratio_optimizer.step()
            else:
                # Use propensity weights
                logits_detector = self.novelty_detector(x)
                y_s_pseudo = torch.randint(2, y_s.shape).to(self.device)
                sample_weights = 1. / self.propensity_scores_s[idx_s].detach()
                sample_weights[y_s_pseudo==1] = 1-sample_weights[y_s_pseudo==1]
                sample_weights /= self.partition_function.detach()
                sample_weights = torch.cat([sample_weights, torch.ones_like(y_t)], dim=0)
                loss_raw = cross_entropy(logits_detector, torch.cat([y_s_pseudo, torch.ones_like(y_t)], dim=0), reduction='none')
                loss = torch.mean(sample_weights*loss_raw)

                detector_optimizer.zero_grad()
                self.manual_backward(loss)
                detector_optimizer.step()


            if self.trainer.is_last_batch:
                update_optimizer(self.current_epoch, self.ratio_optimizer, self.dataset, self.learning_rate)
                update_optimizer(self.current_epoch, self.detector_optimizer, self.dataset, self.learning_rate)

            return loss

        elif stage == "pred_source":
            x_s, y_s, idx_s = batch[:3]

            if self.warm_start:
                logits = self.ratio_estimator(x_s)
            else:
                logits = self.novelty_detector(x_s)
            probs = softmax(logits, dim=1)

            return probs, y_s, logits, idx_s

        elif stage == "pred_disc":

            x_t, y_t, _ = batch[:3]
            if self.warm_start:
                logits = self.ratio_estimator(x_t)
            else:
                logits = self.novelty_detector(x_t)
            probs = softmax(logits, dim=1)

            return probs, y_t, logits

        elif stage == "reweight":

            x_s, _, idx_s  = batch[:3]
            if self.warm_start:
                logits = self.ratio_estimator(x_s)
            else:
                logits = self.novelty_detector(x_s)
            probs = softmax(logits, dim = 1)

            return probs, idx_s

        else:
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "train")

        # self.log("train/loss", {"cross_ent": loss},
        #          on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss.cross_ent", loss, on_step=True, on_epoch=True, prog_bar=False)
#         self.log("train/constraints", {"inequality_violation": ineq_defect}, #, "recall_proxy": recall_proxy
#                  on_step=True, on_epoch=True, prog_bar=False)

        return  {"discriminator_loss": loss.detach()}

    def on_training_epoch_end(self, outputs):
        if self.current_epoch < self.density_estimation_epochs:
            self.warm_start = True
        else:
            if self.reload_model:
                self.novelty_detector.load_state_dict(torch.load(self.model_path + "novelty_detection_model.pth"))
                self.reload_model = False
            self.warm_start = False

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            probs_s, y_s, logits_s, idx_s_val = self.process_batch(batch, "pred_source")
            outputs = {"probs_s": probs_s, "y_s": y_s, "logits_s": logits_s, "idx_s_val": idx_s_val}
            self.validation_step_outputs_s.append(outputs)
            return outputs

        elif dataloader_idx == 1:
            probs_t, y_t, logits_t = self.process_batch(batch, "pred_disc")
            outputs = {"probs_t": probs_t, "y_t": y_t, "logits_t": logits_t}
            self.validation_step_outputs_t.append(outputs)
            return outputs

        elif dataloader_idx == 2:
            probs_s, idx_s = self.process_batch(batch, "reweight")
            outputs = {"probs_s": probs_s, "idx_s": idx_s}
            self.validation_step_outputs_reweight.append(outputs)
            return outputs

    def calibrate_and_transform(self, logits, labels, weights=None):
        calibrator = LogisticRegression(penalty='none')
        logits_inp = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
        calibrator.fit(logits_inp, labels)
        probs = calibrator.predict_proba(logits_inp)[:, 1]
        return calibrator, probs


    def on_validation_epoch_end(self):
        outputs = (self.validation_step_outputs_s, self.validation_step_outputs_t, self.validation_step_outputs_reweight)
        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        logits_s = torch.cat([x["logits_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        val_idx_s = torch.cat([x["idx_s_val"] for x in outputs[0]], dim=0).detach().cpu().numpy()

        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        logits_t = torch.cat([x["logits_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        dataset_labels = np.concatenate([np.zeros_like(y_s), np.ones_like(y_t)])
        # labels of real hidden class
        y_s_oracle = np.zeros_like(y_s)
        novel_inds = np.where(y_t == self.num_classes)[0]
        y_t_oracle = np.zeros_like(y_t)
        y_t_oracle[novel_inds] = 1
        true_labels = np.concatenate([y_s_oracle, y_t_oracle])

        pred_idx_s = np.argmax(probs_s, axis=1)
        pred_idx_t = np.argmax(probs_t, axis=1)
        preds_disc_val = np.concatenate([pred_idx_s, pred_idx_t])
        cur_acc_true_label = np.mean(preds_disc_val == true_labels)
        cur_acc_dataset_labels = np.mean(preds_disc_val == dataset_labels)

        true_label_dist = get_label_dist(y_t, self.num_classes + 1)

        all_logits = np.concatenate([logits_s, logits_t], axis=0)
        all_logits = np.minimum(all_logits, 100.)
        all_logits = np.maximum(all_logits, -100.)
        log.info('max on logits: {}, min on logits: {}'.format(np.max(all_logits), np.min(all_logits)))
        calibrator = VectorScaling()(all_logits, idx2onehot(dataset_labels, 2))
        cal_probs_s = calibrator(all_logits[:logits_s.shape[0]])
        cal_probs_t = calibrator(all_logits[logits_s.shape[0]:])
        cal_probs = np.concatenate([cal_probs_s[:, 1], cal_probs_t[:, 1]])
        log.info('max/min on cal probs: {}, {}'.format(np.max(cal_probs_s), np.min(cal_probs_s)))
        log.info('max/min on cal probs: {}, {}'.format(np.max(cal_probs_t), np.min(cal_probs_t)))

        MP_estimate_BBE = 1.0 - BBE_estimate_binary(source_probs= probs_s[:, 1],
                                                    target_probs= probs_t[:, 1])
        MP_estimate_EN =  1 - estimator_CM_EN(cal_probs_s[:, 0], cal_probs_t[:, 0])
        MP_estimate_dedpul = 1.0 - dedpul(probs_s[:, 0], probs_t[:, 0])

        if self.num_allowed_fp < 0.:
            self.num_allowed_fp = number_of_allowed_false_pos(len(y_s), target_p=self.target_precision,
                                                              confidence=self.precision_confidence)
        pure_bin_estimate, _ = pure_MPE_estimator(probs_s[:, 1], probs_t[:, 1],
                                                  num_allowed_false_pos=self.num_allowed_fp)

        # self.log("pred/MP_estimate_ood" , {"CM_EN": MP_estimate_EN,
        #                                    "BBE": MP_estimate_BBE,
        #                                    "dedpul": MP_estimate_dedpul,
        #                                    "pure_bin": pure_bin_estimate,
        #                                    "true": true_label_dist[self.num_classes]})
        
        self.log("pred/MP_estimate_ood.CM_EN", MP_estimate_EN)
        self.log("pred/MP_estimate_ood.BBE", MP_estimate_BBE)
        self.log("pred/MP_estimate_ood.dedpul", MP_estimate_dedpul)
        self.log("pred/MP_estimate_ood.pure_bin", pure_bin_estimate)
        self.log("pred/MP_estimate_ood.true", true_label_dist[self.num_classes])


        predictions = np.concatenate([probs_s, probs_t])
        cur_auc_true = roc_auc_score(true_labels, predictions[:, 1])
        cur_ap_true = average_precision_score(true_labels, predictions[:, 1])
        cross_ent_val = log_loss(dataset_labels, cal_probs)
#         cross_ent_val = np.float32(cross_entropy(all_logits,
#                                                  torch.LongTensor(dataset_labels)).detach().cpu().numpy())

        if self.warm_start:
            selection_score = cur_acc_dataset_labels
        else:
#             cross_ent_source_val_negated = np.float32(cross_entropy(torch.Tensor(logits_s),
#                                                                     torch.ones_like(torch.LongTensor(y_s)),
#                                                                     reduction='none').detach().cpu().numpy())
            incorrect_src = np.float32(pred_idx_s == np.ones_like(y_s))
            correct_dataset_preds = np.float32(preds_disc_val == dataset_labels)
            sample_weights_val = 1. / self.propensity_scores_s_val[val_idx_s].detach().cpu().numpy()
            sample_weights_val_negated = 1-sample_weights_val
            sample_weights_val = np.concatenate([sample_weights_val, np.ones_like(y_t), sample_weights_val_negated])
            sample_weights_val /= self.partition_function_val.detach().cpu().numpy()
            correct_vec_total = np.concatenate([correct_dataset_preds, incorrect_src])
            selection_score = np.mean(correct_vec_total*sample_weights_val)

#         print(pred_disc_train.shape)
#         print(torch.Tensor(dataset_labels).shape)
#         print(pred_disc_train == torch.Tensor(dataset_labels))
#         corrects_disc_train = torch.zeros_like(preds_disc_train, dtype=torch.float32)
#         corrects_disc_train[preds_disc_train == torch.Tensor(dataset_labels)] = 1.
#         disc_val_acc = torch.mean(corrects_disc_train)

        if self.current_epoch % 10 == 0:
            wandb.log({"ROC_s_vs_t_true" : wandb.plot.roc_curve(true_labels, predictions,
                                                                classes_to_plot=[1])})
            wandb.log({"ROC_s_vs_t" : wandb.plot.roc_curve(dataset_labels, predictions,
                                                           classes_to_plot=[1])})

        ## calibrate probabilities according to validation and then keep propensity scores
#         self.propensity_scores = keep_samples_discriminator(train_probs, train_idx, self.pure_bin_estimate)

        if self.warm_start and self.current_epoch >= 1 and selection_score >= self.best_nonweighted_score:
            self.best_nonweighted_score = selection_score

            train_probs_s = torch.cat([x["probs_s"] for x in outputs[2]], dim=0).detach().cpu().numpy()
            train_idx_s = torch.cat([x["idx_s"] for x in outputs[2]]).detach().cpu().numpy()

#             all_logits = np.minimum(all_logits, 100.)
#             all_logits = np.maximum(all_logits, -100.)
#             calibrator = VectorScaling()(all_logits.detach().cpu().numpy(), idx2onehot(dataset_labels, 2))
            ## calc propensity scores and sample weights for source training data
            propensity_scores_s = calibrator(inverse_softmax(train_probs_s))[:, 0]
            propensity_scores_s = propensity_scores_s[train_idx_s] + 1e-4 ## small added constant to avoid overflow
            self.propensity_scores_s = torch.Tensor(propensity_scores_s).to(self.device)
#             self.partition_function = np.sum(1./self.propensity_scores_s) + np.sum(1-(1./self.propensity_scores_s)) + float(len(self.trainer._data_connector._train_dataloader_source.dataloader().loaders["target_full"]))
            self.partition_function = torch.sum(1./self.propensity_scores_s) + torch.sum(1-(1./self.propensity_scores_s)) + float(len(self.trainer._data_connector._train_dataloader_source.dataloader().loaders["target_full"]))
            ## calc propensity scores for source validation data
            propensity_scores_s_val = calibrator(inverse_softmax(probs_s))[:, 0]
            propensity_scores_s_val = propensity_scores_s[val_idx_s] + 1e-4 ## small added constant to avoid overflow
            self.propensity_scores_s_val = torch.Tensor(propensity_scores_s_val).to(self.device)
            self.partition_function_val = torch.sum(1./self.propensity_scores_s_val) + torch.sum(1-(1./self.propensity_scores_s_val)) + float(len(y_t))
            log.info('propensity stuff:')
            log.info(self.propensity_scores_s_val)
            log.info(self.partition_function_val)
            log.info(np.std(propensity_scores_s))
#             self.best_acc_dataset_labels = cur_acc_dataset_labels

            if self.reload_model:
                torch.save(self.ratio_estimator.state_dict(), self.model_path + "novelty_detection_model.pth")
        if not self.warm_start and selection_score > self.best_weighted_score:
            self.best_weighted_score = selection_score
            self.mpe_at_selection = MP_estimate_EN
            self.auc_roc_at_selection = cur_auc_true
            self.ap_at_selection = cur_ap_true

            self.acc_at_selection = cur_acc_true_label

        # self.log("pred/performance", {"AU-ROC novel": cur_auc_true,
        #                               "curr ave-precision": cur_ap_true,
        #                               "disc acc val ": cur_acc_dataset_labels,
        #                               "selected AU-ROC": self.auc_roc_at_selection,
        #                               "selected ave-precision": self.ap_at_selection,
        #                               "selected acc": self.acc_at_selection,
        #                               "selected alpha": self.mpe_at_selection,
        #                               "loss val": cross_ent_val,
        #                               "score for selection": selection_score,})
        
        self.log("pred/performance.AU-ROC novel", cur_auc_true)
        self.log("pred/performance.curr ave-precision", cur_ap_true)
        self.log("pred/performance.disc acc val", cur_acc_dataset_labels)
        self.log("pred/performance.selected AU-ROC", self.auc_roc_at_selection)
        self.log("pred/performance.selected ave-precision", self.ap_at_selection)
        self.log("pred/performance.selected acc", self.acc_at_selection)
        self.log("pred/performance.selected alpha", self.mpe_at_selection)
        self.log("pred/performance.loss val", cross_ent_val)
        self.log("pred/performance.score for selection", selection_score)

        log_everything(self.logging_file, epoch=self.current_epoch,\
                       auc=cur_auc_true, val_acc=cur_acc_true_label, acc_disc=cur_acc_dataset_labels,\
                       mpe = np.array([pure_bin_estimate, MP_estimate_BBE, MP_estimate_dedpul, MP_estimate_EN]) ,\
                       true_mp = true_label_dist[-1], selected_mpe = self.mpe_at_selection, \
                       selected_auc = self.auc_roc_at_selection, selected_acc = self.acc_at_selection)

        self.validation_step_outputs_s, self.validation_step_outputs_t, self.validation_step_outputs_reweight = [], [], []

    def configure_optimizers(self):
        return [self.detector_optimizer, self.ratio_optimizer]