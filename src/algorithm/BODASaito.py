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

from src.baseline_utils.backpropODASaito19_model_utils import *
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, average_precision_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler
from src.plots.tsne_plot import *
from src.data_utils import *
from tqdm import tqdm

log = logging.getLogger("app")

class BODASaito(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_source_classes: int = 10,
        dataset: str = "CIFAR10",
        learning_rate: float = 0.1,
        logit_multiplier: float = 2.,
        target_precision: float = 0.99,
        precision_confidence: float = 0.95,
        weight_decay: float = 1e-4,
        penalty_type: float = 'l2',
        max_epochs: int = 500,
        warmup_epochs: int = 0,
        warmup_patience: int = 0,
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
    ):
        super().__init__()
        self.num_classes = num_source_classes
        self.fraction_ood_class = fraction_ood_class
        self.use_superclass = use_superclass
        self._device = device
        self.clip = clip
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_outputs = self.num_classes + 1
        self.dataset = dataset
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.constrained_penalty = constrained_penalty
        self.penalty_type = penalty_type
        self.mode = mode
        self.start = 0
        # self.pretrained_model_dir = pretrained_model_dir
        # self.pretrained_model_path = pretrained_model_path + self.dataset + "_vanillaPU_seed_" + str(seed) +"_num_source_cls_"+str(num_source_classes)+"_fraction_ood_class_"+str(fraction_ood_class)+ "_ood_ratio_" + str(ood_class_ratio) +"/"+ "discriminator_model.pth"
        self.pretrained_model_dir = save_model_path + self.dataset
        self.pretrained_model_path = save_model_path + self.dataset + "_" + "CoNoC_seed_"+str(seed)+"_num_source_cls_"+str(num_source_classes)+"_fraction_ood_class_"+str(fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"/supervised_pretrained_novelty_detector_constrained_opt.pth" # "/cis/home/schaud35/shiftpu/models/imagenet_CoNoC_seed_"+str(seed)+"_ood_ratio_"+str(ood_class_ratio)+"/"
        self.feature_model, self.classifier, \
            self.optimizer_feat, self.optimizer_classifier  = \
            get_model_backprob(arch, dataset, self.num_outputs, \
                               learning_rate=learning_rate, weight_decay=weight_decay, \
                                pretrained=pretrained, features=True, pretrained_model_dir=self.pretrained_model_dir)
        
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_patience = warmup_patience

        self.best_valid_source_loss = 1000.
        self.selected_roc_auc = 0.
        self.selected_ap = 0.
        self.selected_f1 = 0.
        self.selected_oscr = 0.
        self.selected_oscpr = 0.    
        self.validation_step_outputs_s = []
        self.validation_step_outputs_t = []
        self.validation_step_outputs_discard = []
        self.val_features_s = torch.tensor([], device=device)
        self.val_features_t = torch.tensor([], device=device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Some variables for the alpha line search
        self.best_valid_supervised_loss, self.epoch_at_best_valid_supervised_loss = 1000., 0
        self.num_allowed_fp = -1
        
        # once constraint is approximately satisifed, allow 5 epochs to train with it, and then reexamine alpha

        self.pred_save_path = f"{pred_save_path}/{dataset}/"

        self.logging_file = f"{self.pred_save_path}/BODA_{arch}_{num_source_classes}_{seed}_log_update.txt"
        
        self.model_path = save_model_path + self.dataset + "_" + "BADO_seed_"+str(seed)+"_num_source_cls_"+str(num_source_classes)+"_ood_class_"+str(ood_class)+"_fraction_ood_class_"+str(fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"_use_labels_"+str(use_labels)+"_use_superclass_"+str(use_superclass)+"/" # "/cis/home/schaud35/shiftpu/models/imagenet_CoNoC_seed_"+str(seed)+"_ood_ratio_"+str(ood_class_ratio)+"/"
        # self.model_path = "/cis/home/schaud35/shiftpu/models/CoNoC_seed_"+str(seed)+"_ood_ratio_"+str(ood_class_ratio)+"/"

        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)

        if not os.path.exists(save_model_path + self.dataset + "_" + "BADO_seed_"+str(seed)+"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"/" ):
            os.makedirs(save_model_path + self.dataset + "_" + "BADO_seed_"+str(seed)+"_num_source_cls_"+str(self.num_classes)+"_fraction_ood_class_"+str(self.fraction_ood_class)+"_ood_ratio_"+str(ood_class_ratio)+"/" )


        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.warm_start = False if self.warmup_epochs == 0 else True
        self.reload_model = False

        self.automatic_optimization = False
    
    def grad_norm(self, model):
        total_norm = 0.
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def get_penalty(self, model):
        penalty_lambda = self.constrained_penalty
        if self.penalty_type == 'l2':
            penalty_term = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            penalty_term = sum(torch.abs(p).sum() for p in model.parameters())
        return penalty_lambda*penalty_term
    
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
            
            # y_s_oracle = torch.zeros_like(y_s, device=self._device)
            # novel_inds = torch.where(y_t == self.num_classes)[0]
            # y_t_oracle = torch.zeros_like(y_t, device=self._device)
            # y_t_oracle[novel_inds] = 1
            # y = torch.cat([y_s_oracle, y_t_oracle], dim=0)

            feat_opt, classifier_opt = self.optimizers()
            ## Optimize 
            feat_opt.zero_grad()
            classifier_opt.zero_grad()
            feat_s = self.feature_model.forward(x_s)
            logit_s = self.classifier.forward(feat_s)
            loss_s = self.criterion(logit_s, y_s)
            self.manual_backward(loss_s+self.get_penalty(self.classifier)+self.get_penalty(self.feature_model)) 

            target_t = torch.ones([x_t.size()[0], 2], device = self._device )*0.5
            feat_t = self.feature_model.forward(x_t)
            logit_t = self.classifier.forward(feat_t, reverse=True)
            prob_t = softmax(logit_t, dim=1)
            prob1 = torch.sum(prob_t[:,:self.num_classes], dim=1)
            prob2 = prob_t[:,self.num_classes]
            prob_t = torch.stack([prob1, prob2], dim=1) 
            bce_loss_t = bce_loss(prob_t, target_t)
            self.manual_backward(bce_loss_t+self.get_penalty(self.classifier)+self.get_penalty(self.feature_model))

            torch.nn.utils.clip_grad_norm_(self.feature_model.parameters(), self.clip)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.clip)
            feat_grad_norm, classifier_grad_norm = self.grad_norm(self.feature_model), self.grad_norm(self.classifier)
            # log.info('feature model gradient norm after training {}'.format(feat_grad_norm))
            # log.info('classifier gradient norm after training {}'.format(classifier_grad_norm))

            feat_opt.step()
            classifier_opt.step()
            return loss_s, bce_loss_t, feat_grad_norm, classifier_grad_norm

        elif stage == "pred_source":
            # import pdb; pdb.set_trace()
            if len(batch)>2:
                x_s, y_s, _ = batch[:3]
            elif len(batch)==2:
                x_s, y_s = batch

            if self.use_superclass & (self.dataset in ["cifar100","newsgroups20"]) :
                y_s = y_s//5 if self.dataset=="cifar100" else y_s//5

            feat_s = self.feature_model.forward(x_s)
            logit_s = self.classifier.forward(feat_s)
            loss_s = self.criterion(logit_s, y_s)

            prob_s = softmax(logit_s, dim=1)

            return prob_s, y_s, logit_s, loss_s

        elif stage == "pred_target":
            # import pdb; pdb.set_trace()
            if len(batch)>2:
                x_t, y_t, _ = batch[:3]
            elif len(batch)==2:
                x_t, y_t = batch

            if self.use_superclass & (self.dataset in ["cifar100","newsgroups20"]) :
                y_t = y_t//5 if self.dataset=="cifar100" else y_t//5

            feat_t = self.feature_model.forward(x_t)
            logit_t = self.classifier.forward(feat_t, reverse=True)
            target_t = torch.ones([x_t.size()[0], 2], device = self._device )*0.5
            probs = softmax(logit_t, dim=1)
            prob1 = torch.sum(probs[:,:self.num_classes], dim=1)
            prob2 = probs[:,self.num_classes]
            prob_t = torch.stack([prob1, prob2], dim=1) 
            bce_loss_t = bce_loss(prob_t, target_t)

            return probs, y_t, logit_t, bce_loss_t

        else:
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss1, loss2, feat_grad_norm, classifier_grad_norm = self.process_batch(batch, "train")
        self.log("train/loss.source", loss1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss.target", loss2, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss.feature_grad_norm", feat_grad_norm, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss.classifier_grad_norm", classifier_grad_norm, on_step=False, on_epoch=True, prog_bar=False)

        return  {"source_loss": loss1.detach(), "target_loss": loss2.detach()}

    def on_train_epoch_end(self):
        if self.current_epoch > self.warmup_epochs:
            self.warm_start = False

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        
        if dataloader_idx == 0:
            probs_s, y_s, logits_s, loss_s = self.process_batch(batch, "pred_source")
            outputs = {"probs_s": probs_s, "y_s": y_s, "logits_s": logits_s} 
            self.validation_step_outputs_s.append(outputs)
            self.log("val/loss.source", loss_s, on_step=False, on_epoch=True, prog_bar=False)
            return outputs

        elif dataloader_idx == 1:
            probs_t, y_t, logits_t, bce_loss_t = self.process_batch(batch, "pred_target")
            outputs = {"probs_t": probs_t, "y_t": y_t, "logits_t": logits_t} 
            self.validation_step_outputs_t.append(outputs)
            self.log("val/loss.target", bce_loss_t, on_step=False, on_epoch=True, prog_bar=False)
            return outputs


    def on_validation_epoch_end(self):
        
        outputs = (self.validation_step_outputs_s, self.validation_step_outputs_t, self.validation_step_outputs_discard)
    
        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        labels = np.concatenate((y_s, y_t), axis=0)
        logits_s = torch.cat([x["logits_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        
        logits_t = torch.cat([x["logits_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        
        # import pdb; pdb.set_trace()
        probs = np.concatenate((probs_s, probs_t), axis=0)
        source_preds = np.argmax(probs_s, axis=1)
        target_preds = np.argmax(probs_t, axis=1)
        target_seen_idx = np.where(y_t < self.num_classes)[0]
        target_unseen_idx = np.where(y_t == self.num_classes)[0]
        source_seen_acc = np.mean(source_preds == y_s)
        target_seen_acc = np.mean(target_preds[target_seen_idx] == y_t[target_seen_idx])
        target_unseen_acc = np.mean(target_preds[target_unseen_idx] == y_t[target_unseen_idx])
        self.log("pred/target_seen_acc", target_seen_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("pred/source_seen_acc", source_seen_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("pred/target_unseen_acc", target_unseen_acc, on_step=False, on_epoch=True, prog_bar=False)

        ood_idx = np.where(y_t == self.num_classes)[0]

        ood_recall = np.sum(target_preds[ood_idx] == y_t[ood_idx]) / len(ood_idx)
        ood_precision = np.sum(target_preds[ood_idx] == y_t[ood_idx]) / np.sum(target_preds == self.num_classes) if np.sum(target_preds == self.num_classes)>0. else 0. 
        # print("ood recall:", ood_recall, "ood_precision:", ood_precision)
        self.log("pred/ood_recall", ood_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log("pred/ood_precision", ood_precision, on_step=False, on_epoch=True, prog_bar=False)

        overall_acc = np.mean(target_preds == y_t)

        self.log("pred/orig_acc", overall_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.MP_estimate = np.zeros(self.num_classes+1)
        for i in range(self.num_classes + 1):
            self.MP_estimate[i] = np.mean(target_preds == i)
        true_label_dist = get_label_dist(y_t, self.num_classes + 1)
        # log_everything(self.logging_file, epoch=self.current_epoch,\
        #     target_orig_acc= overall_acc,\
        #     target_seen_acc=target_seen_acc, source_acc =source_seen_acc,\
        #     precision=ood_precision, recall=ood_recall, 
        #     target_marginal_estimate = self.MP_estimate, target_marginal = true_label_dist) 

        y = np.concatenate((np.zeros_like(y_s), np.ones_like(y_t)), axis=0)
        
        # disc_ce_loss = cross_entropy(torch.cat((self.val_features_s, self.val_features_t),dim=0).cpu().detach(), torch.tensor(y))
        
        y_s_oracle = np.zeros_like(y_s)
        novel_inds = np.where(y_t == self.num_classes)[0]
        y_t_oracle = np.zeros_like(y_t)
        y_t_oracle[novel_inds] = 1
        true_labels = torch.cat((torch.tensor(y_s_oracle), torch.tensor(y_t_oracle)),dim=0)
        
        true_label_dist = get_label_dist(y_t, self.num_classes + 1)
        true_prior = 1 - y_t_oracle.sum().item()/len(y_t_oracle)
        binary_probs_t = np.concatenate((np.expand_dims(np.sum(probs_t[:, :self.num_classes],axis=1), axis=1), np.expand_dims(probs_t[:, self.num_classes], axis=1)),axis=1)
        roc_auc = roc_auc_score(y_t_oracle, binary_probs_t[:,1])
        ap = average_precision_score(y_t_oracle, binary_probs_t[:,1])
        target_preds[target_preds != self.num_classes] = 0.
        target_preds[target_preds == self.num_classes] = 1.
        f1 = f1_score(y_t_oracle, target_preds)
        # oscr = self.compute_oscr(np.sum(probs_t[:,:self.num_classes][y_t_oracle==0],axis=-1), probs_t[:,self.num_classes][y_t_oracle==1], np.argmax(probs_t[:,:self.num_classes], axis=1)[y_t_oracle==0], y_t[y_t_oracle==0])
        # oscpr = self.compute_oscpr(np.sum(probs_t[:,:self.num_classes][y_t_oracle==0],axis=-1), probs_t[:,self.num_classes][y_t_oracle==1], np.argmax(probs_t[:,:self.num_classes], axis=1)[y_t_oracle==0], y_t[y_t_oracle==0])
        oscr = self.compute_bodasaito_oscr(probs_t, y_t, mode='oscr')
        oscpr = self.compute_bodasaito_oscr(probs_t, y_t, mode='oscpr')
        # print("f1:",f1, "roc:", roc_auc, "ap:", ap)
        
        self.log("val/performance.OSCR", oscr, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/performance.OSCPR", oscpr, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/performance.AU-ROC", roc_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/performance.AP", ap, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/performance.F1", f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/estimated_prior", self.MP_estimate[-1], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/true_prior", true_prior, on_step=False, on_epoch=True, prog_bar=False)

        loss_s = self.criterion(torch.tensor(logits_s), torch.tensor(y_s))
        self.log("val/performance.loss_source", loss_s, on_step=False, on_epoch=True)
        target_t = torch.ones([torch.tensor(y_t).size()[0], 2])*0.5
        probs = softmax(torch.tensor(logits_t), dim=1)
        prob1 = torch.sum(probs[:,:self.num_classes], dim=1)
        prob2 = probs[:,self.num_classes]
        prob_t = torch.stack([prob1, prob2], dim=1) 
        bce_loss_t = bce_loss(prob_t, target_t)
        self.log("val/performance.bce_loss_target", bce_loss_t, on_step=False, on_epoch=True)
        if self.best_valid_source_loss > loss_s:
            self.best_valid_source_loss = loss_s
            self.selected_roc_auc = roc_auc
            self.selected_ap = ap
            self.selected_f1 = f1 
            self.selected_oscr = oscr
            self.selected_oscpr = oscpr   
        self.log("val/performance.selected OSCR", self.selected_oscr, on_step=False, on_epoch=True)
        self.log("val/performance.selected OSCPR", self.selected_oscpr, on_step=False, on_epoch=True)
        self.log("val/performance.selected_AU-ROC", self.selected_roc_auc, on_step=False, on_epoch=True)
        self.log("val/performance.selected_AP", self.selected_ap, on_step=False, on_epoch=True)
        self.log("val/performance.selected_F1", self.selected_f1, on_step=False, on_epoch=True)


        self.validation_step_outputs_s = []
        self.validation_step_outputs_t = []
        self.validation_step_outputs_discard = []
        self.val_features_s = torch.tensor([], device=self._device)
        self.val_features_t = torch.tensor([], device=self._device)
    
    def compute_bodasaito_oscr(self, prob, labels, mode='oscr'):
        # import pdb; pdb.set_trace()
        # Cutoffs are of prediction values
        pred = np.argmax(prob, axis=-1)
        m_x1 = np.zeros(len(labels[labels!=self.num_classes]))
        m_x1[pred[labels!=self.num_classes] == labels[labels!=self.num_classes]] = 1
        k_target = np.concatenate((m_x1, np.zeros(len(labels[labels==self.num_classes]))), axis=0)
        u_target = np.concatenate((np.zeros(len(labels[labels!=self.num_classes])), np.ones(len(labels[labels==self.num_classes]))), axis=0)
        predict = np.concatenate((np.max(prob[labels!=self.num_classes],axis=-1), np.max(prob[labels==self.num_classes],axis=-1)), axis=0)

        correct = (pred == labels)
        n = len(prob)
        CCR = [0 for x in range(n + 2)]
        FPR = [0 for x in range(n + 2)]
        recall = [0 for x in range(n + 2)]
        precision = [0 for x in range(n + 2)]

        idx = predict.argsort()

        s_k_target = k_target[idx]
        s_u_target = u_target[idx]

        for k in range(n - 1):
            CC = s_k_target[k + 1:].sum()
            FP = s_u_target[k:].sum()
            FN = (1 - s_u_target[:k]).sum() # s_k_target[:k + 1].sum()


            # True	Positive Rate
            recall[k] = float(CC) / float(len(labels!=self.num_classes))
            # False Positive Rate
            precision[k] = float(CC) / (float(CC)+float(FN)) if (float(CC)+float(FN))!=0. else 0.

            # True	Positive Rate
            CCR[k] = float(CC) / float(len(labels[labels!=self.num_classes]))
            # False Positive Rate
            FPR[k] = float(FP) / float(len(labels[labels==self.num_classes]))
        # import pdb; pdb.set_trace()
        CCR[n] = 0.0
        FPR[n] = 0.0
        CCR[n + 1] = 1.0
        FPR[n + 1] = 1.0
        recall[n] = 0.0
        precision[n] = 0.0
        recall[n + 1] = 1.0
        precision[n + 1] = 1.0

        # Positions of ROC curve (FPR, TPR)
        ROC = sorted(zip(FPR, CCR), reverse=True) if mode=='oscr' else sorted(zip(recall, precision), reverse=True)

        OSCR = 0

        # Compute AUROC Using Trapezoidal Rule
        for j in range(n + 1):
            h = ROC[j][0] - ROC[j + 1][0]
            w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

            OSCR = OSCR + h * w

        # if self.current_epoch==200:
        #     import pdb; pdb.set_trace()
        return OSCR
    
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

    def configure_optimizers(self):

        return [self.optimizer_feat, self.optimizer_classifier]