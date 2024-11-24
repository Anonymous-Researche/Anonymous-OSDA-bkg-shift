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
from abstention.calibration import VectorScaling
import os
from src.MPE_methods.dedpul import *
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, average_precision_score
import time

log = logging.getLogger("app")

class VanillaPU(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_source_classes: int = 10,
        dataset: str=  "CIFAR10",
        learning_rate: float = 0.1,
        weight_decay: float = 5e-4,
        target_precision: float = 0.99,
        precision_confidence: float = 0.9,
        max_epochs: int = 500,
        pred_save_path: str = ".",
        work_dir: str = ".",
        hash: Optional[str] = None,
        pretrained: bool = False,
        seed: int = 0,
        pretrained_model_dir: Optional[str] = None,
        pretrained_model_path: str = None,
        ood_class_ratio: float = 0.005,
        fraction_ood_class: float = 0.1,
        constrained_penalty: float = 0.01,
        use_superclass: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_source_classes
        self.fraction_ood_class = fraction_ood_class
        self.seed=seed
        self.ood_class_ratio = ood_class_ratio
        self.constrained_penalty = constrained_penalty
        self.use_superclass = use_superclass

        self.dataset = dataset
        self.criterion = torch.nn.CrossEntropyLoss()

        self.num_outputs = self.num_classes

        log.info("pretrained {}".format(pretrained_model_dir))
        
        self.oracle_model, self.optimizer_oracle = get_model(arch, dataset, 2, pretrained= pretrained, \
                                                             learning_rate=learning_rate, weight_decay= weight_decay, features=False,
                                                             pretrained_model_dir=pretrained_model_dir, pretrained_model_path=pretrained_model_path)

        # checkpoint = torch.load("/cis/home/schaud35/shiftpu/models/imagenet_vanillaPU_seed_1024_ood_ratio_1.0_num_source_classes_9/oracle_model.pth", map_location='cpu')
        # state_dict = {k: v for k,v in checkpoint.items()}
        # self.oracle_model.load_state_dict(state_dict, strict=False)

        self.target_precision = target_precision
        self.precision_confidence = precision_confidence
        self.discriminator_model, self.optimizer_discriminator = get_model(arch, dataset, 2, pretrained= pretrained, \
                        learning_rate= learning_rate, weight_decay= weight_decay, features=False, pretrained_model_dir= pretrained_model_dir, pretrained_model_path=pretrained_model_path)


        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        ## Tools for ablation study. Learning vanilla PU just with pure bin estimate and two-stage learning.
        self.pure_bin_estimate = 0.
        self.best_criterion = 1000.
        self.best_oracle_criterion = 1000.
        self.mpe_at_selection = 0.
        self.auc_roc_at_selection = 0.
        self.ap_at_selection = 0.
        self.precision_at_selection = 0.
        self.recall_at_selection = 0.
        self.acc_at_selection = 0.
        self.num_allowed_fp = -1
        self.warm_start = True
        self.keep_samples = None
        self.reload_model = True
        self.warmup_epochs = self.max_epochs#//3
        self.validation_step_outputs_s = []
        self.validation_step_outputs_t = []
        self.val_disc_logits_s = torch.tensor([])
        self.val_disc_logits_t = torch.tensor([])
        self.val_oracle_logits_s = torch.tensor([])
        self.val_oracle_logits_t = torch.tensor([])
        self.start = 0
        self.epoch = 0

        self.pred_save_path = f"{pred_save_path}/{dataset}/"

        self.logging_file = f"{self.pred_save_path}/vanilla_pu_{arch}_{num_source_classes}_{seed}_log.txt"

        self.model_path = "/cis/home/schaud35/shiftpu/models/"

        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)

        if not os.path.exists(self.model_path + self.dataset + "_vanillaPU_seed_" + str(self.seed) +"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+ "_ood_ratio_" + str(self.ood_class_ratio) + "/"):
            os.makedirs(self.model_path + self.dataset + "_vanillaPU_seed_" + str(self.seed) +"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+ "_ood_ratio_" + str(self.ood_class_ratio) + "/")


        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.automatic_optimization = False

    def forward_oracle(self, x):
        return self.oracle_model(x)

    def forward_discriminator(self, x):
        return self.discriminator_model(x)
    
    def get_penalty(self, model, penalty_type='l2', wd=0.01):
        penalty_lambda = wd
        if penalty_type == 'l2':
            penalty_term = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            penalty_term = sum(torch.abs(p).sum() for p in model.parameters())
        return penalty_lambda*penalty_term

    def process_batch(self, batch, stage="train"):
        
        if stage == "train":
            # import pdb; pdb.set_trace()
            if len(batch["source_full"])>2:
                x_s, y_s, _ = batch["source_full"][:3]
                x_t, y_t, idx_t = batch["target_full"][:3]
            elif len(batch["source_full"])==2:
                x_s, y_s = batch["source_full"]
                x_t, y_t = batch["target_full"]
                
            if self.use_superclass:
                y_s = y_s//5
                y_t = y_t//5

            oracle_opt, discriminator_opt = self.optimizers()
            
            x = torch.cat([x_s, x_t], dim=0)
            y = torch.cat([torch.zeros_like(y_s), torch.ones_like(y_t)], dim=0)
            
            y_oracle = torch.zeros_like(y_t)
            novel_inds = np.where(y_t.cpu().numpy() == self.num_classes)[0]
            y_oracle[novel_inds] = 1
            y_oracle = torch.cat([torch.zeros_like(y_s), y_oracle], dim=0)

            # log.debug(f"Batch inputs size {x.shape} ")
            # log.debug(f"Batch targets size {one_hot_y.shape} ")

            logits_oracle = self.forward_oracle(x)
            logits_discriminator = self.forward_discriminator(x)

            if self.warm_start:
                penalty_oracle, penalty_disc = self.get_penalty(self.oracle_model, wd=self.constrained_penalty), self.get_penalty(self.discriminator_model, wd=self.constrained_penalty)
                loss1 = cross_entropy(logits_oracle, y_oracle) + penalty_oracle
                loss2 = cross_entropy(logits_discriminator, y) + penalty_disc
                print("loss oracle:",loss1, "loss disc:", loss2)
                for n,p in self.oracle_model.named_parameters():
                    if (n[-13:] == 'linear.weight' or n[-9:]=="fc.weight") and p.grad is not None:
                        print('===========\n oracle model gradient:{}\n----------\n{}'.format(n,p.grad.data.norm(2)))

                for n,p in self.discriminator_model.named_parameters():
                    if (n[-13:] == 'linear.weight' or n[-9:]=="fc.weight") and p.grad is not None:
                        print('===========\n disc model gradient:{}\n----------\n{}'.format(n,p.grad.data.norm(2)))

                total_grad_norm, total_param_norm = 0, 0
                for p in self.oracle_model.parameters():
                    if p.grad is not None:
                        grad_norm = p.grad.data.norm(2)
                        param_norm = p.data.norm(2)
                        total_grad_norm += grad_norm.item() ** 2
                        total_param_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** (1. / 2)
                total_param_norm = total_param_norm ** (1. / 2)
                print("oracle total grad norm:", total_grad_norm, "oracle total param norm:", total_param_norm)

                total_grad_norm, total_param_norm = 0, 0
                for p in self.discriminator_model.parameters():
                    if p.grad is not None:
                        grad_norm = p.grad.data.norm(2)
                        param_norm = p.data.norm(2)
                        total_grad_norm += grad_norm.item() ** 2
                        total_param_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** (1. / 2)
                total_param_norm = total_param_norm ** (1. / 2)
                print("disc total grad norm:", total_grad_norm, "disc total param norm:", total_param_norm)
            else:
                loss1 = cross_entropy(logits_oracle, y_oracle)
                keep_idx = np.concatenate([np.arange(len(y_s), dtype = np.int32), \
                    len(y_s) + np.where(self.keep_samples[idx_t.cpu().numpy()] == 1)[0]], axis=0)
                loss2 = cross_entropy(logits_discriminator[keep_idx], y[keep_idx],\
                    weight=torch.Tensor([1.0 - self.pure_bin_estimate, self.pure_bin_estimate]).to(self.device))

            # log.debug(f"Batch logits size {logits.shape} ")

            oracle_opt.zero_grad()
            self.manual_backward(loss1)
            oracle_opt.step()

            discriminator_opt.zero_grad()
            self.manual_backward(loss2)
            discriminator_opt.step()

            if self.trainer.is_last_batch:
                update_optimizer(self.current_epoch, oracle_opt, self.dataset, self.learning_rate)
                update_optimizer(self.current_epoch, discriminator_opt, self.dataset, self.learning_rate)

            return loss1, loss2, penalty_oracle, penalty_disc

        elif stage == "pred_source":
            if len(batch)>2:
                x_s, y_s, _ = batch
            elif len(batch)==2:
                x_s, y_s = batch

            if self.use_superclass:
                y_s = y_s//5

            logits = self.discriminator_model(x_s)
            self.val_disc_logits_s = torch.cat((self.val_disc_logits_s, logits.cpu().detach()), dim=0)
            probs = softmax(logits, dim=1)#[:, 0]

            disc_probs_s = probs

            logits_s = self.oracle_model(x_s)
            self.val_oracle_logits_s = torch.cat((self.val_oracle_logits_s, logits_s.cpu().detach()))
            probs_s = softmax(logits_s, dim=1)

            return probs_s, y_s, disc_probs_s

        elif stage == "pred_disc":
            if len(batch)>2:
                x_t, y_t, _ = batch
            elif len(batch)==2:
                x_t, y_t = batch

            if self.use_superclass:
                y_t = y_t//5

            logits = self.discriminator_model(x_t)
            self.val_disc_logits_t = torch.cat((self.val_disc_logits_t, logits.cpu().detach()), dim=0)
            probs = softmax(logits, dim=1)#[:, 0]

            disc_probs_t = probs

            logits_t = self.oracle_model(x_t)
            self.val_oracle_logits_t = torch.cat((self.val_oracle_logits_t, logits_t.cpu().detach()), dim=0)
            probs_t = softmax(logits_t, dim=1)

            return probs_t, y_t, disc_probs_t

        else:
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss1, loss2, penalty_oracle, penalty_disc = self.process_batch(batch, "train")

        self.log("train/loss.constraint_penalty_oracle", penalty_oracle, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss.constraint_penalty_disc", penalty_disc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss.oracle", loss1, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss.discriminator", loss2, on_step=True, on_epoch=True, prog_bar=False)
        # self.log_dict("train/loss", {"oracle" : loss1, "discriminator": loss2}, on_step=True, on_epoch=True)

        return  {"oracle_loss": loss1.detach(), "discriminator_loss": loss2.detach()}

    def on_training_epoch_end(self, outputs):
        if self.current_epoch < self.warmup_epochs:
            self.warm_start = True
        else:
            if self.reload_model:
                self.discriminator_model.load_state_dict(torch.load(self.model_path + self.dataset + "_vanillaPU_seed_" + str(self.seed) + "_ood_ratio_" + str(self.ood_class_ratio) + "/"+ "discriminator_model.pth"))
            self.warm_start = False
            self.reload_model = False

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        if dataloader_idx == 0:
            probs_s, y_s,  disc_probs_s = self.process_batch(batch, "pred_source")
            outputs = {"oracle_probs_s": probs_s, "y_s": y_s, "disc_probs_s": disc_probs_s }
            self.validation_step_outputs_s.append(outputs)
            return outputs

        elif dataloader_idx == 1:
            probs_t, y_t,  disc_probs_t = self.process_batch(batch, "pred_disc")
            outputs = {"oracle_probs_t": probs_t, "y_t": y_t, "disc_probs_t": disc_probs_t}
            self.validation_step_outputs_t.append(outputs)
            return outputs

#         elif dataloader_idx == 2:
#             probs, idx = self.process_batch(batch, "discard")
#             return {"probs": probs, "idx": idx}

    def on_validation_epoch_end(self):
        # if self.epoch==100:
        #     import pdb; pdb.set_trace()
        self.epoch = self.epoch + 1
        outputs = (self.validation_step_outputs_s, self.validation_step_outputs_t)
    
        oracle_probs_s = torch.cat([x["oracle_probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        oracle_probs_t = torch.cat([x["oracle_probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        y_s_oracle = np.zeros_like(y_s)
        novel_inds = np.where(y_t == self.num_classes)[0]
        y_t_oracle = np.zeros_like(y_t)
        y_t_oracle[novel_inds] = 1
        true_labels = np.concatenate([y_s_oracle, y_t_oracle])

        disc_probs_s = torch.cat([x["disc_probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        disc_probs_t = torch.cat([x["disc_probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        pred_prob_s, disc_pred_idx_s = np.max(disc_probs_s, axis=1), np.argmax(disc_probs_s, axis=1)
        pred_prob_t, disc_pred_idx_t = np.max(disc_probs_t, axis=1), np.argmax(disc_probs_t, axis=1)

        oracle_pred_idx_s = np.argmax(oracle_probs_s, axis=1)
        oracle_pred_idx_t = np.argmax(oracle_probs_t, axis=1)
        all_preds_idx = np.concatenate([disc_pred_idx_s, disc_pred_idx_t])
        dataset_labels = np.concatenate([np.zeros_like(y_s), np.ones_like(y_t)])

        true_label_dist = get_label_dist(y_t, self.num_classes + 1)
        # EN_estimate_c = estimator_CM_EN(disc_probs_s, disc_probs_t)
        self.start = time.time()
        all_logits = inverse_softmax(np.concatenate([disc_probs_s, disc_probs_t], axis=0))
        print("Inverse Softmax:", time.time() - self.start)
        self.start = time.time()
        calibrator = VectorScaling()(all_logits, idx2onehot(dataset_labels, 2))
        print("VectorScaling:", time.time() - self.start)
        self.start = time.time()
        cal_probs_s = calibrator(inverse_softmax(disc_probs_s))
        print("calibrator source:", time.time() - self.start)
        self.start = time.time()
        cal_probs_t = calibrator(inverse_softmax(disc_probs_t))
        print("calibrator target", time.time() - self.start)
        self.start = time.time()

        MP_estimate_BBE = 1.0 - BBE_estimate_binary(source_probs= disc_probs_s[:, 1],
                                                    target_probs= disc_probs_t[:, 1])
        print("MPE BBE time:", time.time() - self.start)
        self.start = time.time()
        MP_estimate_EN =  1 - estimator_CM_EN(cal_probs_s[:, 0], cal_probs_t[:, 0])
        print("MPE EN time:", time.time() - self.start)
        self.start = time.time()
        MP_estimate_dedpul = 1.0 - dedpul(pred_prob_s, pred_prob_t)
        print("MPE dedpul time:", time.time() - self.start)
        
        
        ## prediction on target data according the a classifier thresholded at the EN MPE
        pred_t_pos_idx = np.argsort(disc_probs_t[:, 1])[-int(MP_estimate_EN*disc_probs_t.shape[0]):]
        pred_idx_t_EN_thresh = np.zeros_like(disc_pred_idx_t)
        pred_idx_t_EN_thresh[pred_t_pos_idx] = 1
        cur_acc_true_label = np.mean(pred_idx_t_EN_thresh == y_t_oracle)
        
        if self.num_allowed_fp < 0.:
            self.num_allowed_fp = number_of_allowed_false_pos(len(y_s), target_p=self.target_precision,
                                                              confidence=self.precision_confidence)
#         pure_bin_estimate, _ = pure_MPE_estimator(disc_probs_s[:, 1], disc_probs_t[:, 1],
#                                                   num_allowed_false_pos=self.num_allowed_fp)
#
#         pure_bin_oracle, _ = pure_MPE_estimator(oracle_probs_s[:, 1], oracle_probs_t[:, 1],
#                                                   num_allowed_false_pos=self.num_allowed_fp)
        pure_bin_estimate, _ = pure_MPE_estimator(disc_probs_s[:, 1], disc_probs_t[:, 1],
                                                  num_allowed_false_pos=int(0.01*disc_probs_s.shape[0]))

        pure_bin_oracle, _ = pure_MPE_estimator(oracle_probs_s[:, 1], oracle_probs_t[:, 1],
                                                num_allowed_false_pos=int(0.01*disc_probs_s.shape[0]))

        # self.log("pred/MPE_estimate_ood" , {"CM_EN": MP_estimate_EN,
        #                                     "BBE": MP_estimate_BBE,
        #                                     "pure_bin": pure_bin_estimate,
        #                                     "pure_bin_oracle": pure_bin_oracle,
        #                                     "true": true_label_dist[self.num_classes]})
        self.log("pred/MPE_estimate_ood.CM_EN", MP_estimate_EN)
        self.log("pred/MPE_estimate_ood.BBE", MP_estimate_BBE)
        self.log("pred/MPE_estimate_ood.pure_bin", pure_bin_estimate)
        self.log("pred/MPE_estimate_ood.pure_bin_oracle", pure_bin_oracle)
        self.log("pred/MPE_estimate_ood.true", true_label_dist[self.num_classes])
        
#         probs_for_val_loss = np.concatenate([disc_probs_s[:, 1], disc_probs_t[:, 1]])
#         probs_for_val_loss = np.minimum(np.maximum(probs_for_val_loss, 0.0001), 0.9999)

#         disc_val_loss = log_loss(dataset_labels, probs_for_val_loss)
#         disc_val_loss = accuracy_score(dataset_labels, np.concatenate([pred_idx_s, pred_idx_t]))

        predictions_oracle = np.concatenate([oracle_probs_s, oracle_probs_t])
        pred_probs_disc = np.concatenate([disc_probs_s, disc_probs_t], axis=0)
        disc_logits = torch.cat((self.val_disc_logits_s, self.val_disc_logits_t), dim=0)
        oracle_logits = torch.cat((self.val_oracle_logits_s, self.val_oracle_logits_t), dim=0)
        # import pdb; pdb.set_trace()
        #### Important change...
        cur_auc_true = roc_auc_score(y_t_oracle, disc_probs_t[:, 1])
        cur_ap_true = average_precision_score(y_t_oracle, disc_probs_t[:, 1])
        cur_auc_oracle = roc_auc_score(y_t_oracle, oracle_probs_t[:, 1])
        cur_ap_oracle = average_precision_score(y_t_oracle, oracle_probs_t[:, 1])
        cross_ent_disc = cross_entropy(disc_logits, torch.tensor(dataset_labels))
        cross_ent_oracle = cross_entropy(oracle_logits, torch.tensor(true_labels))

#         preds_disc_val = np.concatenate([np.argmax(logits_s, axis=1), np.argmax(logits_t, axis=1)])
        disc_val_acc = accuracy_score(dataset_labels, all_preds_idx)
        selection_criterion = cross_ent_disc # disc_val_acc
        
        # self.log("pred/preformance", {"AU-ROC": cur_auc_true,
        #                               "oracle AU-ROC": cur_auc_oracle,
        #                               "curr ave-precision": cur_ap_true,
        #                               "oracle ave-precision": cur_ap_oracle,
        #                               "validation accuracy": disc_val_acc})
        self.log("pred/preformance.AU-ROC", cur_auc_true)
        self.log("pred/preformance.oracle AU-ROC", cur_auc_oracle)
        self.log("pred/preformance.curr ave-precision", cur_ap_true)
        self.log("pred/preformance.oracle ave-precision", cur_ap_oracle)
        self.log("pred/performance.validation disc cross_entropy", cross_ent_disc)
        self.log("pred/performance.validation oracle cross_entropy", cross_ent_oracle)

        
#         if self.current_epoch % 10 == 0:
#             wandb.log({"ROC_s_vs_t_true" : wandb.plot.roc_curve(true_labels, predictions_disc,
#                                                                 classes_to_plot=[1])})
#             wandb.log({"ROC_s_vs_t" : wandb.plot.roc_curve(dataset_labels, predictions_disc,
#                                                            classes_to_plot=[1])})


        if self.best_criterion >= selection_criterion:
            self.best_criterion = selection_criterion

            self.mpe_at_selection = MP_estimate_EN
            self.auc_roc_at_selection = cur_auc_true
            self.ap_at_selection = cur_ap_true

            # calc precision and recall
            pred_idx = ((disc_probs_t[:, 0]) * (1 - MP_estimate_EN) * 1.0 * len(y_t) / len(y_s) < 0.5)
            recall_EN_threshold = np.sum((pred_idx ==1)*(y_t == self.num_classes)) / np.sum(y_t == self.num_classes)
            prec_EN_threshold = np.sum((pred_idx ==1)*(y_t == self.num_classes)) / np.sum(pred_idx ==1)
            self.recall_at_selection = recall_EN_threshold
            self.precision_at_selection = prec_EN_threshold
            self.acc_at_selection = cur_acc_true_label

            wandb.log({"ROC_target_true" : wandb.plot.roc_curve(y_t_oracle, disc_probs_t,
                                                                classes_to_plot=[1])})
            wandb.log({"ROC_s_vs_t" : wandb.plot.roc_curve(dataset_labels, pred_probs_disc,
                                                           classes_to_plot=[1])})

            torch.save(self.discriminator_model.state_dict(), self.model_path + self.dataset + "_vanillaPU_seed_" + str(self.seed) +"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+ "_ood_ratio_" + str(self.ood_class_ratio) + "/"+ "discriminator_model.pth")

        if self.best_oracle_criterion >= cross_ent_oracle:
            self.best_oracle_criterion = cross_ent_oracle
            self.log("pred/preformance.selected oracle AU-ROC", cur_auc_oracle)
            self.log("pred/preformance.selected oracle ave-precision", cur_ap_oracle)
            torch.save(self.oracle_model.state_dict(), self.model_path + self.dataset + "_vanillaPU_seed_" + str(self.seed) +"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+ "_ood_ratio_" + str(self.ood_class_ratio) + "/" + "oracle_model.pth")
#             train_probs_s = torch.cat([x["probs_s"] for x in outputs[2]], dim=0).detach().cpu().numpy()
#             train_idx_s = torch.cat([x["idx_s"] for x in outputs[2]]).detach().cpu().numpy()
#             all_logits = np.minimum(all_logits, 100.)
#             all_logits = np.maximum(all_logits, -100.)
#             calibrator = VectorScaling()(all_logits.detach().cpu().numpy(), idx2onehot(dataset_labels, 2))
#             self.propensity_scores_s = calibrator(inverse_softmax(train_probs_s))[:, 0]
#             self.propensity_scores_s = self.propensity_scores_s[train_idx_s]


#         if self.current_epoch >=4 and self.pure_bin_estimate >= self.best_bin_size and self.discriminator_model:
#             self.best_bin_size = self.pure_bin_estimate
#             torch.save(self.discriminator_model.state_dict(), self.model_path + "discriminator_model.pth")

        # self.log("pred/performance", {"disc acc val ": disc_val_acc,
        #                               "task acc val": cur_acc_true_label,
        #                               "selected AU-ROC": self.auc_roc_at_selection,
        #                               "selected ave-precision": self.ap_at_selection,
        #                               "selected recall": self.recall_at_selection,
        #                               "selected precision": self.precision_at_selection,
        #                               "selected acc": self.acc_at_selection,
        #                               "selected alpha": self.mpe_at_selection})
        

        self.log("pred/performance.disc acc val ", disc_val_acc)
        self.log("pred/performance.task acc val", cur_acc_true_label)
        self.log("pred/performance.selected AU-ROC", self.auc_roc_at_selection)
        self.log("pred/performance.selected ave-precision", self.ap_at_selection)
        self.log("pred/performance.selected recall", self.recall_at_selection)
        self.log("pred/performance.selected precision", self.precision_at_selection)
        self.log("pred/performance.selected acc", self.acc_at_selection)
        self.log("pred/performance.selected alpha", self.mpe_at_selection)

        log_everything(self.logging_file, epoch=self.current_epoch,\
            auc=cur_auc_true, val_acc=cur_acc_true_label, acc_disc=disc_val_acc,\
            mpe = np.array([pure_bin_estimate, MP_estimate_BBE, MP_estimate_dedpul, MP_estimate_EN]) ,\
            true_mp = true_label_dist[-1], selected_mpe = self.mpe_at_selection, \
            selected_auc = self.auc_roc_at_selection, selected_acc = self.acc_at_selection, \
            selected_recall = self.recall_at_selection, selected_prec = self.precision_at_selection)
        

        torch.save(self.discriminator_model.state_dict(), self.model_path + self.dataset + "_vanillaPU_seed_" + str(self.seed) +"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+ "_ood_ratio_" + str(self.ood_class_ratio) + "/"+ "discriminator_model_latest.pth")
        torch.save(self.oracle_model.state_dict(), self.model_path + self.dataset + "_vanillaPU_seed_" + str(self.seed) +"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+ "_ood_ratio_" + str(self.ood_class_ratio) + "/" + "oracle_model_latest.pth")
        self.validation_step_outputs_s, self.validation_step_outputs_t = [], []
        self.val_disc_logits_s = torch.tensor([])
        self.val_disc_logits_t = torch.tensor([])
        self.val_oracle_logits_s = torch.tensor([])
        self.val_oracle_logits_t = torch.tensor([])
        


    def configure_optimizers(self):
        return self.optimizer_oracle, self.optimizer_discriminator