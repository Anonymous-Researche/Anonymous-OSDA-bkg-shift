import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List
import numpy as np
from src.data_utils import *
import logging 
# from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities import CombinedLoader
from pytorch_lightning import seed_everything
import torch
import random
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.plots.tsne_plot import *
from sarpu.pu_learning import *
from tqdm import tqdm

log = logging.getLogger("app")

class SAREMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        dataset: str = "CIFAR10", 
        fraction_ood_class: float = 0.1,
#         frac_of_new_class: float = 1.,
        train_fraction: float = 0.8,
        num_source_classes: int = 10,
        use_aug: bool = False,
        batch_size: int = 200,
        seed: int = 42,
        mode: str = 'domain_disc',
        ood_class: int = 0,
        ood_class_ratio: float = 0.005,
        arch: str = 'Roberta',
        use_superclass: bool=True,
        preprocess = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_aug = use_aug
        
        self.fraction_ood_class = fraction_ood_class
        self.ood_class = ood_class
        self.ood_class_ratio = ood_class_ratio
#         self.frac_of_new_class = frac_of_new_class
        self.train_fraction = train_fraction
        self.num_source_classes = num_source_classes

        ## Fix this to avoid exploding importance weights
        self.min_source_fraction = 0.2 #0.2 #0.01 #0.1
        
        self.train_transform = get_preprocessing(self.dataset, self.use_aug, train=True)
        self.test_transform = get_preprocessing(self.dataset, self.use_aug, train=False) 
        
        self.seed = seed
        self.mode = mode
        self.arch = arch
        self.use_superclass = use_superclass
        self.ood_class = ood_class
    def setup(self, stage: Optional[str] = None):
        seed_everything(self.seed)
        random.seed(self.seed)
        x_train, y_train, s_train, x_test, y_test, s_test, classification_attributes, propensity_attributes, classification_model_type, propensity_model_type = get_combined_data(self.data_dir, self.dataset, arch=self.arch, \
            transform=[self.train_transform, self.test_transform],\
            train_fraction=self.train_fraction, seed=self.seed, mode=self.mode, use_superclass=self.use_superclass)

        # assign all samples where y_train==0 and s_train==0 to 2 in y_train
        y_train[(y_train==0) & (s_train==0)] = 2
        y_test[(y_test==0) & (s_test==0)] = 2
        self.labeled_source = ng20Dataset(torch.tensor(x_train[s_train==1]), torch.tensor(y_train[s_train==1]), classification_attributes, propensity_attributes)
        self.unlabeled_target = ng20Dataset(torch.tensor(x_train[s_train==0]), torch.tensor(y_train[s_train==0]), classification_attributes, propensity_attributes)
        self.valid_labeled_source = ng20Dataset(torch.tensor(x_test[s_test==1]), torch.tensor(y_test[s_test==1]), classification_attributes, propensity_attributes)
        self.valid_labeled_target = ng20Dataset(torch.tensor(x_test[s_test==0]), torch.tensor(y_test[s_test==0]), classification_attributes, propensity_attributes)
        
        log.info("Done ")
        log.debug(f"OOD class {self.ood_class}, OOD subsample {self.ood_class_ratio}")
        log.debug("Stats of training data ... ")
        log.debug(f"Labeled source data {len(self.labeled_source)} and Unlabeled target samples {len(self.unlabeled_target)}")

        log.debug("Stats of validation data ... ")
        log.debug(f"Labeled source data {len(self.valid_labeled_source)} and Labeled target data {len(self.valid_labeled_target)} ")
    
    def train_dataloader(self):
        log.info("batch size: {}".format(self.batch_size))

        full_dataloaders =  ( DataLoader( self.labeled_source, batch_size=10000, shuffle=True, \
            num_workers=8,  pin_memory=True), DataLoader( self.unlabeled_target, batch_size=10000,\
            shuffle=True, num_workers=8,  pin_memory=True))

        train_loader = {
            "source_full": full_dataloaders[0], 
            "target_full": full_dataloaders[1], 
        }
        return CombinedLoader(train_loader)
    def val_dataloader(self):
        log.info("val batch size: {}".format(self.batch_size))

        full_dataloaders =  ( DataLoader( self.valid_labeled_source, batch_size=10000, shuffle=True, \
            num_workers=8,  pin_memory=True), DataLoader( self.valid_labeled_target, batch_size=10000,\
            shuffle=True, num_workers=8,  pin_memory=True))
        
        full_val_loader = CombinedLoader({
            "source_full": full_dataloaders[0], 
            "target_full": full_dataloaders[1], 
        })

        train_target_dataloader = DataLoader(self.unlabeled_target, batch_size=10000, \
            shuffle=True, num_workers=8, pin_memory=True)
        
        return [full_dataloaders[0], full_dataloaders[1], train_target_dataloader, full_val_loader]

class RandomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        dataset: str = "CIFAR10", 
        fraction_ood_class: float = 0.1,
#         frac_of_new_class: float = 1.,
        train_fraction: float = 0.8,
        num_source_classes: int = 10,
        use_aug: bool = False,
        batch_size: int = 200,
        seed: int = 42,
        mode: str = 'domain_disc',
        ood_class: int = 0,
        ood_class_ratio: float = 0.005,
        arch: str = 'Resnet18',
        use_superclass: bool=False,
        preprocess = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_aug = use_aug
        
        self.fraction_ood_class = fraction_ood_class
        self.ood_class_ratio = ood_class_ratio
#         self.frac_of_new_class = frac_of_new_class
        self.train_fraction = train_fraction
        self.num_source_classes = num_source_classes

        ## Fix this to avoid exploding importance weights
        self.min_source_fraction = 0.2 #0.2 #0.01 #0.1

        self.train_transform = get_preprocessing(self.dataset, self.use_aug, train=True)
        self.test_transform = get_preprocessing(self.dataset, self.use_aug, train=False) 
        self.seed = seed
        self.mode = mode
        self.arch=arch
        self.use_superclass = use_superclass

    def select_ood_samples(self, dataset, ood_class, ood_class_ratio):
        random.seed(self.seed)
        if isinstance(dataset, Subset):
            ood_idxs = np.where(np.array(dataset.dataset.targets)==ood_class)[0]
            id_idxs = np.setdiff1d(dataset.indices, ood_idxs)
            sub_ood_idxs = list(np.setdiff1d(dataset.indices, id_idxs))
            select_ood_idxs = np.array(random.sample(sub_ood_idxs, int(ood_class_ratio*len(sub_ood_idxs))),dtype=np.int64) 
            selected_idxs = np.concatenate((id_idxs, select_ood_idxs), axis=0, dtype=np.int64)
            dataset.indices = selected_idxs
            return dataset
        else:
            print("Expected a Subset to select OOD samples based on the defined OOD ratio")
            # ood_idxs = np.where(np.array(dataset.targets)==ood_class)[0]
            # id_idxs = np.setdiff1d(range(len(dataset)), ood_idxs)
            # select_ood_idxs = np.array(random.sample(list(ood_idxs), int(ood_class_ratio*len(ood_idxs))),dtype=np.int64) 
            # selected_idxs = np.concatenate((id_idxs, select_ood_idxs), axis=0, dtype=np.int64)

        return dataset


    def setup(self, stage: Optional[str] = None):
        # import pdb; pdb.set_trace()
        seed_everything(self.seed)
        train_data, val_data, train_data_AL = get_combined_data(self.data_dir, self.dataset, arch=self.arch,\
            transform=[self.train_transform, self.test_transform],\
            train_fraction=self.train_fraction, seed=self.seed, mode=self.mode, use_superclass=self.use_superclass)
        
        
        if isinstance(train_data, Subset):
            labels = get_labels(train_data.dataset.targets)
        else:
            labels = get_labels(train_data.targets)
        
        labels = labels[:int(np.ceil(self.num_source_classes/(1 - self.fraction_ood_class)))]
        
        print('seed in datamodule: {}'.format(self.seed))
        
        self.source_classes = list(np.random.choice(labels, int(len(labels)*(1 - self.fraction_ood_class)), replace=False))
        # self.source_classes = labels
        ood_class = list(np.setdiff1d(labels, self.source_classes))
        # ood_class = [1000]

        self.target_classes = self.source_classes.copy()
        
        self.target_classes.append(list(ood_class))

        self.source_marginal = np.round(np.random.uniform(self.min_source_fraction, 1.0, len(self.source_classes)), 2)
        self.source_marginal_valid = np.round(np.random.uniform(0.2, 1.0, len(self.source_classes)), 2)

        self.target_marginal = 1.0 - self.source_marginal
        self.target_marginal_valid = 1.0 - self.source_marginal_valid

        self.target_marginal =  np.concatenate((self.target_marginal, np.array([1.0])))
#         self.target_marginal =  np.concatenate((self.target_marginal, np.array([self.frac_of_new_class])))

        log.debug(f"Source classes: {self.source_classes}")
        log.debug(f"Source marginal: {self.source_marginal}")
        log.debug(f"Source marginal validation: {self.source_marginal_valid}")

        log.debug(f"Target classes: {self.target_classes}")
        log.debug(f"Target marginal: {self.target_marginal}")
        log.debug(f"Target marginal validation: {self.target_marginal_valid}")

        log.info("Creating training data ... ")
        self.labeled_source, self.unlabeled_target, self.source_idx, self.target_idx, self.target_transform =\
            get_splits_from_data(train_data,\
            source_classes = self.source_classes, source_marginal =self.source_marginal, \
            target_classes=self.target_classes, target_marginal=self.target_marginal)

        # Active Learning dataset without Random torchvision transforms
        # _, self.unlabeled_AL_pool, _, _, _ =\
        #     get_splits_from_data(train_data_AL,\
        #     source_classes = self.source_classes, source_marginal =self.source_marginal, \
        #     target_classes=self.target_classes, target_marginal=self.target_marginal)
        
        log.info("Done ")
        
        log.info("Creating validation data ... ")
        self.valid_labeled_source, self.valid_labeled_target, _, _, _ = \
            get_splits_from_data(val_data, \
            source_classes = self.source_classes, source_marginal =self.source_marginal, \
            target_classes=self.target_classes, target_marginal=self.target_marginal)
        
        self.labeled_source = self.select_ood_samples(self.labeled_source, ood_class, self.ood_class_ratio)
        self.unlabeled_target = self.select_ood_samples(self.unlabeled_target, ood_class, self.ood_class_ratio)         
        self.valid_labeled_source = self.select_ood_samples(self.valid_labeled_source, ood_class, self.ood_class_ratio)
        self.valid_labeled_target = self.select_ood_samples(self.valid_labeled_target, ood_class, self.ood_class_ratio)

        # self.unlabeled_AL_pool.indices = self.unlabeled_target.indices

        log.info("Done ")

        log.debug("Stats of training data ... ")
        log.debug(f"Labeled source data {len(self.labeled_source)} and Unlabeled target samples {len(self.unlabeled_target)}")

        log.debug("Stats of validation data ... ")
        log.debug(f"Labeled source data {len(self.valid_labeled_source)} and Labeled target data {len(self.valid_labeled_target)} ")
        # import pdb; pdb.set_trace()

    def train_dataloader(self):
        
        # source_batch_size = int(self.batch_size*1.0*self.source_train_size/(self.source_train_size + self.target_train_size)) 
        # target_batch_size = int(self.batch_size*1.0*self.target_train_size/(self.source_train_size + self.target_train_size)) 

        # split_dataloaders = ( DataLoader( self.labeled_source, batch_size=source_batch_size, shuffle=True, \
        #     num_workers=4,  pin_memory=True), DataLoader( self.unlabeled_target, batch_size=target_batch_size,\
        #     shuffle=True, num_workers=4,  pin_memory=True))
        log.info("batch size: {}".format(self.batch_size))

        full_dataloaders =  ( DataLoader( self.labeled_source, batch_size=self.batch_size, shuffle=True, \
            num_workers=8,  pin_memory=True), DataLoader( self.unlabeled_target, batch_size=self.batch_size,\
            shuffle=True, num_workers=8,  pin_memory=True))
        # full_dataloaders =  ( DataLoader( self.labeled_source, batch_size=self.batch_size, shuffle=True), DataLoader( self.unlabeled_target, batch_size=self.batch_size,\
        #     shuffle=True))


        train_loader = {
            "source_full": full_dataloaders[0], 
            "target_full": full_dataloaders[1], 
        }
        
        return CombinedLoader(train_loader)
       

    def val_dataloader(self):
        
        # source_batch_size = int(self.batch_size*1.0*self.source_valid_size/(self.source_valid_size + self.target_valid_size)) 
        # target_batch_size = int(self.batch_size*1.0*self.target_valid_size/(self.source_valid_size + self.target_valid_size)) 

        # split_dataloaders = ( DataLoader( self.valid_labeled_source, batch_size=source_batch_size, shuffle=True, \
        #     num_workers=4,  pin_memory=True), DataLoader( self.valid_labeled_target, batch_size=target_batch_size,\
        #     shuffle=True, num_workers=4,  pin_memory=True) )

        log.info("val batch size: {}".format(self.batch_size))

        full_dataloaders =  ( DataLoader( self.valid_labeled_source, batch_size=self.batch_size, shuffle=True, \
            num_workers=8,  pin_memory=True), DataLoader( self.valid_labeled_target, batch_size=self.batch_size,\
            shuffle=True, num_workers=8,  pin_memory=True))
        # full_dataloaders =  ( DataLoader( self.valid_labeled_source, batch_size=self.batch_size, shuffle=True), DataLoader( self.valid_labeled_target, batch_size=self.batch_size,\
        #     shuffle=True))

        train_target_dataloader = DataLoader(self.unlabeled_target, batch_size=self.batch_size, \
            shuffle=True, num_workers=8, pin_memory=True)
        # train_target_dataloader = DataLoader(self.unlabeled_target, batch_size=self.batch_size, \
        #     shuffle=True)

        # valid_loader = {
        #     "source_full": full_dataloaders[0], 
        #     "target_full": full_dataloaders[1], 
        # }
        
        return [full_dataloaders[0], full_dataloaders[1], train_target_dataloader]
        

class RandomCovariateShiftDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        dataset: str = "CIFAR10", 
        fraction_ood_class: float = 0.1,
#         frac_of_new_class: float = 1.,
        train_fraction: float = 0.8,
        num_source_classes: int = 10,
        use_aug: bool = False,
        batch_size: int = 200,
        seed: int = 42,
        mode: str = 'domain_disc',
        ood_class: int = 0,
        ood_class_ratio: float = 0.005,
        arch: str = 'Roberta',
        use_superclass: bool=True,
        preprocess = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_aug = use_aug
        
        self.fraction_ood_class = fraction_ood_class
        self.ood_class = ood_class
        self.ood_class_ratio = ood_class_ratio
#         self.frac_of_new_class = frac_of_new_class
        self.train_fraction = train_fraction
        self.num_source_classes = num_source_classes

        ## Fix this to avoid exploding importance weights
        self.min_source_fraction = 0.2 #0.2 #0.01 #0.1

        self.train_transform = get_preprocessing(self.dataset, self.use_aug, train=True)
        self.test_transform = get_preprocessing(self.dataset, self.use_aug, train=False) 
        self.seed = seed
        self.mode = mode
        self.arch = arch
        self.use_superclass = use_superclass
        self.ood_class = ood_class

    def select_ood_samples(self, dataset, ood_class, ood_class_ratio):
        random.seed(self.seed)
        if isinstance(dataset, Subset):
            ood_idxs = np.where(np.array(dataset.dataset.targets)==ood_class)[0]
            id_idxs = np.setdiff1d(dataset.indices, ood_idxs)
            sub_ood_idxs = list(np.setdiff1d(dataset.indices, id_idxs))
            select_ood_idxs = np.array(random.sample(sub_ood_idxs, int(ood_class_ratio*len(sub_ood_idxs))),dtype=np.int64) 
            selected_idxs = np.concatenate((id_idxs, select_ood_idxs), axis=0, dtype=np.int64)
            dataset.indices = selected_idxs
            return dataset, len(select_ood_idxs), len(id_idxs), len(selected_idxs)
        else:
            print("Expected a Subset to select OOD samples based on the defined OOD ratio")
            # ood_idxs = np.where(np.array(dataset.targets)==ood_class)[0]
            # id_idxs = np.setdiff1d(range(len(dataset)), ood_idxs)
            # select_ood_idxs = np.array(random.sample(list(ood_idxs), int(ood_class_ratio*len(ood_idxs))),dtype=np.int64) 
            # selected_idxs = np.concatenate((id_idxs, select_ood_idxs), axis=0, dtype=np.int64)

        return dataset


    def setup(self, stage: Optional[str] = None):
        
        seed_everything(self.seed)
        random.seed(self.seed)
        train_data, val_data, train_data_AL = get_combined_data(self.data_dir, self.dataset, arch=self.arch, \
            transform=[self.train_transform, self.test_transform],\
            train_fraction=self.train_fraction, seed=self.seed, mode=self.mode, use_superclass=self.use_superclass)
        
        
        if isinstance(train_data, Subset):
            labels = get_labels(train_data.dataset.targets)
        else:
            labels = get_labels(train_data.targets)
        
        # labels = labels[:int(np.ceil(self.num_source_classes/(1 - self.fraction_ood_class)))]

        print('seed in datamodule: {}'.format(self.seed))
        
        # CIFAR100 classes for covariate shift
        self.source_classes = [[54, 62, 70, 82, 92],[4, 30, 55, 72, 95], [1, 32, 67, 73, 91], [0, 51, 53, 57, 83]] # [[54, 62, 70, 82, 92],[4, 30, 55, 72, 95]]
        ood_class = [self.ood_class] # [46] #[30] # [27] # [92] # [2]
        
        # Newsgroups20 classes for covariate shift
        # self.source_classes = [[1,2,3,4],[7,8,9,10],[11,12,13,14],[16,17,18,19]] # [[0],[1,2,3,4,5],[6],[7,8,9,10],[11,12,13,14],[15],[16,17,18,19]]
        # ood_class = [6]
        
        # Shift
        # self.subclass_marginal = np.round(np.random.uniform(self.min_source_fraction, 1.0, len(self.source_classes[0])), 2)
        
        # No shift
        # self.subclass_marginal = np.round(np.random.uniform(self.min_source_fraction, 0.5, len(self.source_classes[0])), 2)
        self.subclass_marginal = np.round(np.random.uniform(0.45, 0.5, len(self.source_classes[0])), 4)
        
        # self.source_marginal_valid = np.round(np.random.uniform(0.2, 1.0, len(self.source_classes[0])), 2)
        self.source_marginal = np.array([])
        for src_cls in self.source_classes:
            self.source_marginal = np.append(self.source_marginal, self.subclass_marginal)

        self.source_classes = [sub_cls for src_cls in self.source_classes for sub_cls in src_cls]
        # lower_source_marginal = np.round(np.random.uniform(0.08, 0.15, len(self.source_classes)), 4)
        # higher_source_marginal = np.round(np.random.uniform(0.85, 0.92, len(self.source_classes)), 4)
        # self.source_marginal = np.array(random.sample(list(np.concatenate((lower_source_marginal, higher_source_marginal),axis=0)),len(self.source_classes)))
        self.target_classes = self.source_classes.copy()
        # self.target_classes_valid = self.source_classes.copy()
        # self.target_classes_valid.append(list([self.ood_class, 98]))
        self.target_classes.append(list(ood_class))

        # self.target_marginal = 1.0 - self.source_marginal
        self.target_marginal = self.source_marginal
        # self.target_marginal_valid = 1.0 - self.source_marginal_valid
        # self.target_marginal_valid = np.concatenate((self.target_marginal, np.array([1.0]*len(ood_class))))
        self.target_marginal =  np.concatenate((self.target_marginal, np.array([1.0]*len(ood_class))))
        # self.target_marginal =  np.concatenate((self.target_marginal, np.array([self.frac_of_new_class])))
        
        # self.source_classes = [54, 62, 70, 82, 92, 4, 30, 55, 72, 95]
        # ood_class = [2]
        # self.target_classes = self.source_classes.copy()
        # self.target_classes.append(list(ood_class))
        # self.source_marginal = [1., 1., 0., 0., 0., 1., 1., 0., 0., 0.]
        # self.target_marginal = [0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 1.]        

        log.debug(f"Source classes: {self.source_classes}")
        log.debug(f"Source marginal: {self.source_marginal}")
        # log.debug(f"Source marginal validation: {self.source_marginal_valid}")

        log.debug(f"Target classes: {self.target_classes}")
        log.debug(f"Target marginal: {self.target_marginal}")
        # log.debug(f"Target marginal validation: {self.target_marginal_valid}")

        log.info("Creating training data ... ")
        self.labeled_source, self.unlabeled_target, self.source_idx, self.target_idx, self.target_transform =\
            get_splits_from_data(train_data,\
            source_classes = self.source_classes, source_marginal =self.source_marginal, \
            target_classes=self.target_classes, target_marginal=self.target_marginal)

        # # Active Learning dataset without Random torchvision transforms
        # _, self.unlabeled_AL_pool, _, _, _ =\
        #     get_splits_from_data(train_data_AL,\
        #     source_classes = self.source_classes, source_marginal =self.source_marginal, \
        #     target_classes=self.target_classes, target_marginal=self.target_marginal)
        
        log.info("Done ")
        
        log.info("Creating validation data ... ")
        
        self.valid_labeled_source, self.valid_labeled_target, _, _, _ = \
            get_splits_from_data(val_data, \
            source_classes = self.source_classes, source_marginal =self.source_marginal, \
            target_classes=self.target_classes, target_marginal=self.target_marginal)
        for ood_cls in self.target_classes[-1]:
            self.labeled_source, source_ood, source_id, source_total = self.select_ood_samples(self.labeled_source, ood_cls, self.ood_class_ratio)
            self.unlabeled_target, target_ood, target_id, target_total = self.select_ood_samples(self.unlabeled_target, ood_cls, self.ood_class_ratio)         
        # for ood_cls in self.target_classes_valid[-1]:
            self.valid_labeled_source, val_source_ood, val_source_id, val_source_total = self.select_ood_samples(self.valid_labeled_source, ood_cls, self.ood_class_ratio)
            self.valid_labeled_target, val_target_ood, val_target_id, val_target_total = self.select_ood_samples(self.valid_labeled_target, ood_cls, self.ood_class_ratio)

        # self.unlabeled_AL_pool.indices = self.unlabeled_target.indices

        log.info("Done ")
        log.debug(f"OOD class {ood_class}, OOD subsample {self.ood_class_ratio}")
        log.debug("Stats of training data ... ")
        log.debug(f"Labeled source data {len(self.labeled_source)} and Unlabeled target samples {len(self.unlabeled_target)}")
        log.debug(f"Source ood data: {source_ood}, Source id data: {source_id}, Source total data: {source_total}")
        log.debug(f"Target ood data: {target_ood}, Target id data: {target_id}, Target total data: {target_total}")
        log.debug(f"target alpha: {target_ood/target_total}")
        log.debug("Stats of validation data ... ")
        log.debug(f"Labeled source data {len(self.valid_labeled_source)} and Labeled target data {len(self.valid_labeled_target)} ")
        log.debug(f"Source ood data: {val_source_ood}, Source id data: {val_source_id}, Source total data: {val_source_total}")
        log.debug(f"Target ood data: {val_target_ood}, Target id data: {val_target_id}, Target total data: {val_target_total}")
        log.debug(f"target alpha: {val_target_ood/val_target_total}")
        
        # st_df = self.save_dataset(self.labeled_source, name='source_train_extreme_w_shift')
        # tt_df = self.save_dataset(self.unlabeled_target, name='target_train_extreme_w_shift')
        # sv_df = self.save_dataset(self.valid_labeled_source, name='source_val_extreme_w_shift')
        # tv_df = self.save_dataset(self.valid_labeled_target, name='target_val_extreme_w_shift')
        # exit()

        # from models import ResNet18
        # net_w_labels = ResNet18(num_classes=22, features=True)
        # net_wo_labels = ResNet18(num_classes=22, features=True)
        # net_w_labels.load_state_dict(torch.load('/cis/home/schaud35/shiftpu/shiftpu/ResNet18/novelty_detector_model_w_labels_seed_'+str(self.seed)+'_ood_'+str(self.ood_class)+'.pth'))
        # net_wo_labels.load_state_dict(torch.load('/cis/home/schaud35/shiftpu/shiftpu/ResNet18/novelty_detector_model_wo_labels_seed_'+str(self.seed)+'_ood_'+str(self.ood_class)+'.pth'))
        # net_w_labels.eval()
        # net_w_labels.cuda()
        # net_wo_labels.eval()
        # net_wo_labels.cuda()

        # train_target_loader = DataLoader(self.unlabeled_target, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        # val_target_loader = DataLoader(self.valid_labeled_target, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        # feats_w_labels = torch.tensor([], device='cuda')
        # feats_wo_labels = torch.tensor([], device='cuda')
        # labels = torch.tensor([], device='cuda')
        # with torch.no_grad():
        #     for id, batch in tqdm(enumerate(train_target_loader)):
        #         x, y, _ = batch
        #         x, y = x.cuda(), y.cuda()
        #         feats_w_labels = torch.cat((feats_w_labels, net_w_labels(x).detach()), dim=0)
        #         feats_wo_labels = torch.cat((feats_wo_labels, net_wo_labels(x).detach()), dim=0)
        #         labels = torch.cat((labels, y), dim=0)
        # feats_w_labels = feats_w_labels.cpu().detach().numpy()
        # feats_wo_labels = feats_wo_labels.cpu().detach().numpy()
        # labels = labels.cpu().detach().numpy()
        # labels = labels//5

        # tsne_w_labels = compute_tsne(feats_w_labels, perplexity=50)
        # tsne_wo_labels = compute_tsne(feats_wo_labels, perplexity=50)
        # plot_2d_scatterplot(tsne_w_labels, labels, num_classes=3, save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/ResNet18/2d_tsne_feats_w_labels_seed_'+str(self.seed)+'_ood_'+str(self.ood_class)+'.png')
        # plot_2d_scatterplot(tsne_wo_labels, labels, num_classes=3, save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/ResNet18/2d_tsne_feats_wo_labels_seed_'+str(self.seed)+'_ood_'+str(self.ood_class)+'.png')

        # tsne_w_labels_3d = compute_tsne(feats_w_labels, n_components=3, perplexity=50)
        # tsne_wo_labels_3d = compute_tsne(feats_wo_labels, n_components=3, perplexity=50)
        # plot_3d_scatterplot(tsne_w_labels_3d, labels, reduction_algo="tsne", save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/ResNet18/3d_tsne_feats_w_labels_seed_'+str(self.seed)+'_ood_'+str(self.ood_class)+'.png')
        # plot_3d_scatterplot(tsne_wo_labels_3d, labels, reduction_algo="tsne", save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/ResNet18/3d_tsne_feats_wo_labels_seed_'+str(self.seed)+'_ood_'+str(self.ood_class)+'.png')
        # import pdb; pdb.set_trace()

        # ls_x = np.array([self.labeled_source[i][0].cpu().detach().numpy() for i in range(len(self.labeled_source))])
        # ut_x = np.array([self.unlabeled_target[i][0].cpu().detach().numpy() for i in range(len(self.unlabeled_target))])
        # val_ls_x = np.array([self.valid_labeled_source[i][0].cpu().detach().numpy() for i in range(len(self.valid_labeled_source))])
        # val_lt_x = np.array([self.valid_labeled_target[i][0].cpu().detach().numpy() for i in range(len(self.valid_labeled_target))])
        # ls_y = np.array([self.labeled_source[i][1] for i in range(len(self.labeled_source))])
        # ut_y = np.array([self.unlabeled_target[i][1] for i in range(len(self.unlabeled_target))])
        # val_ls_y = np.array([self.valid_labeled_source[i][1] for i in range(len(self.valid_labeled_source))])
        # val_lt_y = np.array([self.valid_labeled_target[i][1] for i in range(len(self.valid_labeled_target))])
        # all_x = np.concatenate([ls_x, ut_x, val_ls_x, val_lt_x],axis=0)
        # all_y = np.concatenate([ls_y, ut_y, val_ls_y, val_lt_y],axis=0)
        # domain_y = np.zeros_like(all_y)
        # domain_y[len(ls_y):(len(ls_y)+len(ut_y))] = 1
        # domain_y[-len(val_lt_y):] = 1
        # domain_y[all_y==10] = 2

        # tsne_x = compute_tsne(all_x)
        # tsne_x_3d = compute_tsne(all_x, n_components=3)
        # pca_x = compute_PCA(all_x)
        # pca_x_3d = compute_PCA(all_x, n_components=3)
        
        # plot_2d_scatterplot(tsne_x, all_y, num_classes=11, save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/2d_tsne_ImageNet_ResNet18_feats_y_cifar100_normalize.png') 
        # plot_2d_scatterplot(tsne_x, domain_y, num_classes=3, save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/2d_tsne_ImageNet_ResNet18_feats_domain_cifar100_normalize.png') 
        # plot_3d_scatterplot(tsne_x_3d, all_y, save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/3d_tsne_ImageNet_ResNet18_feats_y_cifar100_normalize.png') 
        # plot_3d_scatterplot(tsne_x_3d, domain_y, save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/3d_tsne_ImageNet_ResNet18_feats_domain_cifar100_normalize.png') 
        # plot_2d_scatterplot(pca_x, all_y, num_classes=11, save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/2d_pca_ImageNet_ResNet18_feats_y_cifar100_normalize.png') 
        # plot_2d_scatterplot(pca_x, domain_y, num_classes=3, save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/2d_pca_ImageNet_ResNet18_feats_domain_cifar100_normalize.png') 
        # plot_3d_scatterplot(pca_x_3d, all_y, save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/3d_pca_ImageNet_ResNet18_feats_y_cifar100_normalize.png') 
        # plot_3d_scatterplot(pca_x_3d, domain_y, save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/3d_pca_ImageNet_ResNet18_feats_domain_cifar100_normalize.png') 

        # target_class_avg_dists, target_avg_dists, target_ood_dists, target_id_dists = self.get_kNN_scores(self.labeled_source, self.unlabeled_target, self.valid_labeled_source, self.valid_labeled_target, 10, same_sets=False)
        # print('target knn anomaly class scores:',[round(i,3) for i in target_class_avg_dists])
        # source_class_avg_dists, source_avg_dist, _, _ = self.get_kNN_scores(self.labeled_source, self.labeled_source, self.valid_labeled_source, self.valid_labeled_source, 10, same_sets=True)
        # print('source knn anomaly class scores:',[round(i,3) for i in source_class_avg_dists])
        
        # plt.hist(target_id_dists.cpu().detach(), label='target known classes P_[T,[k]]', alpha=0.3)
        # plt.hist(target_ood_dists.cpu().detach(), label='target novel class P_[T,1]', alpha=0.9)
        # plt.hist(source_avg_dist.cpu().detach(), label='source known classes P_S', alpha=0.3)
        # plt.xlabel('Anomaly scores')
        # plt.ylabel('Frequency')
        # plt.legend(loc='upper right')
        # plt.title('Anomaly score histogram')
        # plt.savefig('/cis/home/schaud35/shiftpu/shiftpu/plots/cifar100_'+str(self.seed)+'_'+str(self.ood_class)+'_'+str(self.ood_class_ratio)+'_histogram_w_shift.png')
        
        # import pdb; pdb.set_trace()
    
    def save_dataset(self, dataset, name='source_tr'):
        data_dir = '/cis/home/schaud35/shiftpu/shiftpu/data_splits/cifar100/'
        idx_ls = np.zeros((len(dataset)), dtype=np.int32)
        super_target_ls = np.zeros((len(dataset)), dtype=np.int32)
        target_ls = np.zeros((len(dataset)), dtype=np.int32)
        # image_file_ls = []
        for i,data in enumerate(dataset):
            idx = dataset.indices[i]
            feature, super_target, target  = data[0], data[1], data[2]
            # image_file_ls.append(dataset.dataset._image_files[idx])
            feature = feature.cpu().detach().numpy()
            idx_ls[i] = idx
            target_ls[i] = target
            super_target_ls[i] = super_target
        
        data_dir = os.path.join(data_dir, str(self.seed)+'_'+str(self.ood_class)+'_'+str(self.ood_class_ratio)+'_'+str(self.fraction_ood_class))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        df = pd.DataFrame({'idx': idx_ls, 'super_target': super_target_ls, 'target': target_ls})
        df.to_csv(os.path.join(data_dir,name+'_split.csv'))
        return df

    def get_kNN_scores(self, source_data, target_data, val_source_data, val_target_data, k, same_sets=False):
        train_source = [source_data[i][0].cpu().detach().numpy() for i in range(len(source_data))]
        val_source = [val_source_data[i][0].cpu().detach().numpy() for i in range(len(val_source_data))]
        train_source.extend(val_source)
        source_tensor = torch.tensor(np.array(train_source), device='cuda:0' if torch.cuda.is_available() else 'cpu')

        source_labels = [source_data[i][1] for i in range(len(source_data))]
        val_source_labels = [val_source_data[i][1] for i in range(len(val_source_data))]
        source_labels.extend(val_source_labels)
        source_labels = torch.tensor(source_labels)

        source_sents = [source_data[i][2] for i in range(len(source_data))]
        val_source_sents = [val_source_data[i][2] for i in range(len(val_source_data))]
        source_sents.extend(val_source_sents)
        source_sents = torch.tensor(source_sents)

        train_target = [target_data[i][0].cpu().detach().numpy() for i in range(len(target_data))]
        val_target = [val_target_data[i][0].cpu().detach().numpy() for i in range(len(val_target_data))]
        train_target.extend(val_target)
        target_tensor = torch.tensor(np.array(train_target), device='cuda:0' if torch.cuda.is_available() else 'cpu')

        target_labels = [target_data[i][1] for i in range(len(target_data))]
        val_target_labels = [val_target_data[i][1] for i in range(len(val_target_data))]
        target_labels.extend(val_target_labels)
        target_labels = torch.tensor(target_labels)

        # Calculate pairwise distances between target and source samples
        distances = torch.cdist(target_tensor, source_tensor)
        # Calculate cosine similarity between target and source samples
        # similarities = F.cosine_similarity(target_tensor.unsqueeze(1), source_tensor.unsqueeze(0), dim=2)

        # Get the indices of the k-nearest neighbors for each target sample
        if same_sets:
            _, indices = torch.topk(distances, k+1, largest=False)
            indices = indices[:,1:]
        else:
            _, indices = torch.topk(distances, k, largest=False)
        # _, cosine_indices = torch.topk(similarities, k, largest=True)
        
        # Gather the k-nearest neighbors from the source dataset
        nearest_neighbors = torch.gather(source_tensor.unsqueeze(0).expand(target_tensor.size(0), -1, -1), 1, indices.unsqueeze(2).expand(-1, -1, source_tensor.size(1)))        
        # cosine_nearest_neighbors = torch.gather(source_tensor.unsqueeze(0).expand(target_tensor.size(0), -1, -1), 1, cosine_indices.unsqueeze(2).expand(-1, -1, source_tensor.size(1)))        

        # Calculate the average distance for each target sample
        avg_distances = torch.mean(torch.norm(nearest_neighbors - target_tensor.unsqueeze(1), dim=2), dim=1)
        # cosine_avg_distances = torch.mean(torch.norm(cosine_nearest_neighbors - target_tensor.unsqueeze(1), dim=2), dim=1)
        class_avg_dists = [torch.mean(avg_distances[target_labels==i]).cpu().detach().tolist() for i in target_labels.unique()]
        
        val_target_labels = torch.tensor(val_target_labels)
        novel_labels = torch.zeros_like(val_target_labels)
        novel_labels[val_target_labels==val_target_labels.unique()[-1]] = 1.
        auroc = roc_auc_score(novel_labels, avg_distances[-len(val_target_labels):].cpu().detach())
        auprc = average_precision_score(novel_labels, avg_distances[-len(val_target_labels):].cpu().detach())
        print('auroc:', auroc, 'auprc:', auprc)

        # plot histogram
        novel_labels = torch.zeros_like(target_labels)
        novel_labels[target_labels==target_labels.unique()[-1]] = 1.
        ood_dists = avg_distances[novel_labels==1.]
        id_dists = avg_distances[novel_labels!=1.]
        # import pdb; pdb.set_trace()
        return class_avg_dists, avg_distances, ood_dists, id_dists #, cosine_avg_distances
    
    def train_dataloader(self):
        
        # source_batch_size = int(self.batch_size*1.0*self.source_train_size/(self.source_train_size + self.target_train_size)) 
        # target_batch_size = int(self.batch_size*1.0*self.target_train_size/(self.source_train_size + self.target_train_size)) 

        # split_dataloaders = ( DataLoader( self.labeled_source, batch_size=source_batch_size, shuffle=True, \
        #     num_workers=4,  pin_memory=True), DataLoader( self.unlabeled_target, batch_size=target_batch_size,\
        #     shuffle=True, num_workers=4,  pin_memory=True))
        log.info("batch size: {}".format(self.batch_size))

        full_dataloaders =  ( DataLoader( self.labeled_source, batch_size=self.batch_size, shuffle=True, \
            num_workers=8,  pin_memory=True), DataLoader( self.unlabeled_target, batch_size=self.batch_size,\
            shuffle=True, num_workers=8,  pin_memory=True))
        # full_dataloaders =  ( DataLoader( self.labeled_source, batch_size=self.batch_size, shuffle=True), DataLoader( self.unlabeled_target, batch_size=self.batch_size,\
        #     shuffle=True))


        train_loader = {
            "source_full": full_dataloaders[0], 
            "target_full": full_dataloaders[1], 
        }
        
        return CombinedLoader(train_loader)
       

    def val_dataloader(self):
        
        # source_batch_size = int(self.batch_size*1.0*self.source_valid_size/(self.source_valid_size + self.target_valid_size)) 
        # target_batch_size = int(self.batch_size*1.0*self.target_valid_size/(self.source_valid_size + self.target_valid_size)) 

        # split_dataloaders = ( DataLoader( self.valid_labeled_source, batch_size=source_batch_size, shuffle=True, \
        #     num_workers=4,  pin_memory=True), DataLoader( self.valid_labeled_target, batch_size=target_batch_size,\
        #     shuffle=True, num_workers=4,  pin_memory=True) )

        log.info("val batch size: {}".format(self.batch_size))

        full_dataloaders =  ( DataLoader( self.valid_labeled_source, batch_size=self.batch_size, shuffle=True, \
            num_workers=8,  pin_memory=True), DataLoader( self.valid_labeled_target, batch_size=self.batch_size,\
            shuffle=True, num_workers=8,  pin_memory=True))
        
        full_val_loader = CombinedLoader({
            "source_full": full_dataloaders[0], 
            "target_full": full_dataloaders[1], 
        })
        # full_dataloaders =  ( DataLoader( self.valid_labeled_source, batch_size=self.batch_size, shuffle=True), DataLoader( self.valid_labeled_target, batch_size=self.batch_size,\
        #     shuffle=True))

        train_target_dataloader = DataLoader(self.unlabeled_target, batch_size=self.batch_size, \
            shuffle=True, num_workers=8, pin_memory=True)
        
        # train_target_dataloader = DataLoader(self.unlabeled_target, batch_size=self.batch_size, \
        #     shuffle=True)

        # valid_loader = {
        #     "source_full": full_dataloaders[0], 
        #     "target_full": full_dataloaders[1], 
        # }
        
        return [full_dataloaders[0], full_dataloaders[1], train_target_dataloader, full_val_loader]


class TabulaCovariateShiftDataModule(RandomCovariateShiftDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        dataset: str = "CIFAR10", 
        fraction_ood_class: float = 0.1,
#         frac_of_new_class: float = 1.,
        train_fraction: float = 0.8,
        num_source_classes: int = 10,
        use_aug: bool = False,
        batch_size: int = 200,
        seed: int = 42,
        mode: str = 'domain_disc',
        ood_class: int = 0,
        ood_class_ratio: float = 0.005,
        arch: str = 'Roberta',
        use_superclass: bool=True,
        preprocess = None,
    ):
        super().__init__(data_dir, dataset, fraction_ood_class, train_fraction,
            num_source_classes, use_aug, batch_size, seed, mode, ood_class, ood_class_ratio, arch)

        self.min_source_fraction = 0.1
        self.use_superclass=use_superclass

    def get_ood_idxs(self, dataset, idxs, ood_class, ood_class_ratio):
        random.seed(self.seed)
        ood_idxs = np.where(np.array(dataset.targets)==ood_class)[0]
        id_idxs = np.setdiff1d(idxs, ood_idxs)
        sub_ood_idxs = list(np.setdiff1d(idxs, id_idxs))
        selected_ood_idxs = np.array(random.sample(sub_ood_idxs, int(ood_class_ratio*len(sub_ood_idxs))),dtype=np.int64) 
        return id_idxs, selected_ood_idxs

    def get_splits_from_data(self, data, source_classes, target_classes, source_marginal, shift_attr, ood_class, ood_class_ratio):
        source_idxs, target_idxs = [], []
        for lidx, label in enumerate(source_classes):
            for i,attr in enumerate(shift_attr):
                idxs = np.intersect1d(np.where(data.targets == label)[0], np.where(data.age == attr)[0])
                np.random.shuffle(idxs)
                source_idxs.extend(idxs[:int(len(idxs)*source_marginal[lidx*len(shift_attr)+i])])
                target_idxs.extend(idxs[int(len(idxs)*source_marginal[lidx*len(shift_attr)+i]):])
        ood_idxs = np.where(data.targets==ood_class)[0]
        random.shuffle(ood_idxs)
        ood_idxs = ood_idxs[:int(len(ood_idxs)*ood_class_ratio)]
        target_idxs.extend(ood_idxs)
        labeled_source = Subset(data, source_idxs)
        unlabeled_target = Subset(data, target_idxs)  
        
        return labeled_source, unlabeled_target, source_idxs, target_idxs, ood_idxs

        
    def setup(self, stage: Optional[str] = None):
        
        seed_everything(self.seed)
        random.seed(self.seed)

        # ood class mapped to the last label below 
        train_data, val_data, train_data_AL = get_combined_data(self.data_dir, self.dataset, self.arch, self.ood_class,\
            transform=[self.train_transform, self.test_transform],\
            train_fraction=self.train_fraction, seed=self.seed, mode=self.mode, use_superclass=self.use_superclass)

        if isinstance(train_data, Subset):
            labels = get_labels(train_data.dataset.targets)
            shift_attr = get_labels(train_data.dataset.sex)
        else:
            labels = get_labels(train_data.targets)
            shift_attr = get_labels(train_data.sex)
        
        random.shuffle(labels)
        train_data.label_map = {labels[i]:i for i in range(len(labels))}
        val_data.label_map = {labels[i]:i for i in range(len(labels))}
        train_data_AL.label_map = {labels[i]:i for i in range(len(labels))}
        
        ood_class = labels[-1]
        self.source_classes = labels[:-1]
        self.target_classes = self.source_classes.copy()
        self.target_classes.append(ood_class)
        
        lower_source_marginal = np.round(np.random.uniform(0.05, 0.15, len(shift_attr)*len(self.source_classes)//2), 4) # np.round(np.random.uniform(0.08, 0.15, len(shift_attr)//2), 2)
        higher_source_marginal = np.round(np.random.uniform(0.85, 0.95, (len(shift_attr) - len(shift_attr)//2)*len(self.source_classes)), 4) # np.round(np.random.uniform(0.85, 0.92, len(shift_attr) - len(shift_attr)//2), 2)
        self.source_marginal = list(np.concatenate((lower_source_marginal, higher_source_marginal),axis=0))
        random.shuffle(self.source_marginal)

        
        log.debug(f"Source classes: {self.source_classes}")
        log.debug(f"Target classes: {self.target_classes}")
        log.debug(f"Source marginal: {self.source_marginal}")

        log.debug(f"OOD class ratio: {self.ood_class_ratio}")

        log.info("Creating training data ... ")
        self.labeled_source, self.unlabeled_target, train_source_idxs, train_target_idxs, train_ood_idxs = self.get_splits_from_data(train_data, \
                source_classes=self.source_classes, target_classes=self.target_classes, \
                source_marginal=self.source_marginal, shift_attr=shift_attr, ood_class=ood_class, ood_class_ratio=self.ood_class_ratio)
        log.debug(f"Label mapping during training: {train_data.label_map}")
        log.info("Done.")

        log.info("Creating validation data ... ")
        self.valid_labeled_source, self.valid_labeled_target, val_source_idxs, val_target_idxs, val_ood_idxs = self.get_splits_from_data(val_data, \
                source_classes=self.source_classes, target_classes=self.target_classes, \
                source_marginal=self.source_marginal, shift_attr=shift_attr, ood_class=ood_class, ood_class_ratio=self.ood_class_ratio)
        log.debug(f"Label mapping during eval: {val_data.label_map}")
        log.info("Done.")

        log.debug("Stats of training data ... ")
        log.debug(f"Labeled source data {len(self.labeled_source)} and Unlabeled target samples {len(self.unlabeled_target)}")
        log.debug(f"OOD samples: {len(train_ood_idxs)}")
        log.debug(f"alpha: {len(train_ood_idxs)/len(train_target_idxs)}")

        log.debug("Stats of validation data ... ")
        log.debug(f"Labeled source data {len(self.valid_labeled_source)} and Labeled target data {len(self.valid_labeled_target)} ")
        log.debug(f"OOD samples: {len(val_ood_idxs)}")
        log.debug(f"alpha: {len(val_ood_idxs)/len(val_target_idxs)}")

        print('seed in datamodule: {}'.format(self.seed))
        # import pdb; pdb.set_trace()


class SentimentCovariateShiftDataModule(RandomCovariateShiftDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        dataset: str = "CIFAR10", 
        fraction_ood_class: float = 0.1,
#         frac_of_new_class: float = 1.,
        train_fraction: float = 0.8,
        num_source_classes: int = 10,
        use_aug: bool = False,
        batch_size: int = 200,
        seed: int = 42,
        mode: str = 'domain_disc',
        ood_class: int = 0,
        ood_class_ratio: float = 0.005,
        arch: str = 'Roberta',
        use_superclass: bool=True,
        preprocess = None,
    ):
        super().__init__(data_dir, dataset, fraction_ood_class, train_fraction,
            num_source_classes, use_aug, batch_size, seed, mode, ood_class, ood_class_ratio, arch)

        self.min_source_fraction = 0.1
        self.use_superclass=use_superclass

    def get_ood_idxs(self, dataset, idxs, ood_class, ood_class_ratio):
        random.seed(self.seed)
        ood_idxs = np.where(np.array(dataset.targets)==ood_class)[0]
        id_idxs = np.setdiff1d(idxs, ood_idxs)
        sub_ood_idxs = list(np.setdiff1d(idxs, id_idxs))
        selected_ood_idxs = np.array(random.sample(sub_ood_idxs, int(ood_class_ratio*len(sub_ood_idxs))),dtype=np.int64) 
        return id_idxs, selected_ood_idxs

    def get_splits_from_idxs(self, data, neg_idxs, pos_idxs, source_marginal, source_classes, num_max_idxs=2000, shift=True):
        
        np.random.seed(self.seed)
        size_per_class = get_size_per_class(data)
        source_split_idxs, target_split_idxs= [], []
        # import pdb; pdb.set_trace()
        for src_cls, src_mrg in zip(source_classes, source_marginal):
            neg_cls_idxs = [i for i in neg_idxs if data[i][1]==src_cls]
            pos_cls_idxs = [i for i in pos_idxs if data[i][1]==src_cls]
            np.random.shuffle(neg_cls_idxs)
            np.random.shuffle(pos_cls_idxs)
            neg_cls_idxs = neg_cls_idxs[:num_max_idxs]
            pos_cls_idxs = pos_cls_idxs[:len(neg_cls_idxs)]
            
            if shift:
                source_split_idxs.extend(neg_cls_idxs[:round(len(neg_cls_idxs)*source_marginal[src_cls])])
                source_split_idxs.extend(pos_cls_idxs[:round(len(pos_cls_idxs)*(1-source_marginal[src_cls]))])
                target_split_idxs.extend(neg_cls_idxs[round(len(neg_cls_idxs)*source_marginal[src_cls]):])
                target_split_idxs.extend(pos_cls_idxs[round(len(pos_cls_idxs)*(1-source_marginal[src_cls])):])
            else:
                source_split_idxs.extend(neg_cls_idxs[:round(len(neg_cls_idxs)*source_marginal[src_cls])])
                source_split_idxs.extend(pos_cls_idxs[:round(len(pos_cls_idxs)*(1-source_marginal[src_cls]))])
                target_split_idxs.extend(neg_cls_idxs[round(len(neg_cls_idxs)*source_marginal[src_cls]):round(2*len(neg_cls_idxs)*source_marginal[src_cls])])
                target_split_idxs.extend(pos_cls_idxs[round(len(pos_cls_idxs)*(1-source_marginal[src_cls])):round(2*len(pos_cls_idxs)*(1-source_marginal[src_cls]))])
            
            # source_split_idxs.extend(neg_cls_idxs[:round(size_per_class[src_cls]*source_marginal[src_cls])])
            # source_split_idxs.extend(pos_cls_idxs[:round(size_per_class[src_cls]*(1-source_marginal[src_cls]))])
            # target_split_idxs.extend(neg_cls_idxs[round(size_per_class[src_cls]*source_marginal[src_cls]):])
            # target_split_idxs.extend(pos_cls_idxs[round(size_per_class[src_cls]*(1-source_marginal[src_cls])):])

        return source_split_idxs, target_split_idxs

    def setup(self, stage: Optional[str] = None):
        ood_class = self.ood_class
        seed_everything(self.seed)
        random.seed(self.seed)

        # ood class mapped to the last label below 
        train_data, val_data, train_data_AL = get_combined_data(self.data_dir, self.dataset, self.arch, ood_class,\
            transform=[self.train_transform, self.test_transform],\
            train_fraction=self.train_fraction, seed=self.seed, mode=self.mode, use_superclass=self.use_superclass)
        
        if isinstance(train_data, Subset):
            labels = get_labels(train_data.dataset.targets)
        else:
            labels = get_labels(train_data.targets)
        
        # labels = labels[:int(np.ceil(self.num_source_classes/(1 - self.fraction_ood_class)))]

        print('seed in datamodule: {}'.format(self.seed))
        
        train_sents = np.array([train_data[i][-1] for i in range(len(train_data))])
        val_sents = np.array([val_data[i][-1] for i in range(len(val_data))])
        
        train_neg_idxs = np.where(train_sents == 0)[0]
        train_neg_id_idxs, train_neg_ood_idxs = self.get_ood_idxs(train_data, train_neg_idxs, self.num_source_classes, self.ood_class_ratio)
        
        train_pos_idxs = np.where(train_sents == 1)[0]
        train_pos_id_idxs, train_pos_ood_idxs = self.get_ood_idxs(train_data, train_pos_idxs, self.num_source_classes, self.ood_class_ratio)

        source_classes = labels[:-1]
        # negative sentiment ratio in each class: [0.0347, 0.0804, 0.0116, 0.1030], [0.0608, 0.0852, 0.1980, 0.0246, 0.0538]
        # source_marginal = np.array([0.5, 0.5, 0.5, 0.5, 0.5])# np.round(np.random.uniform(0.45, 0.55, len(source_classes)), 4) # np.array([0.05, 0.95, 0.07, 0.97, 0.1]) # np.array([0.1, 0.9, 0.13, 0.92, 0.15]) # np.round(np.random.uniform(self.min_source_fraction, 0.9, len(source_classes)), 2) # [0.01, 0.02, 0.15, 0.01, 0.04] # [0.01, 0.02, 0.01, 0.09] # np.round(np.random.uniform(self.min_source_fraction, 0.8, len(source_classes)), 2)
        lower_source_marginal = np.round(np.random.uniform(0.08, 0.15, len(source_classes)), 4)
        higher_source_marginal = np.round(np.random.uniform(0.85, 0.92, len(source_classes)), 4)
        # Shift
        source_marginal = random.sample(list(np.concatenate((lower_source_marginal, higher_source_marginal),axis=0)),len(source_classes))
        
        # No shift
        # source_marginal = np.round(np.random.uniform(0.45, 0.5, len(source_classes)), 4) # [random.sample([lower_source_marginal[i], higher_source_marginal[i]],1)[0] for i in range(len(lower_source_marginal))]
        # import pdb; pdb.set_trace()
        train_source_idxs, train_target_idxs = self.get_splits_from_idxs(train_data, train_neg_id_idxs, train_pos_id_idxs, source_marginal, source_classes, 500, shift=True) 
        train_target_idxs = np.concatenate((train_target_idxs, train_neg_ood_idxs, train_pos_ood_idxs), axis=0, dtype=np.int64)

        val_neg_idxs = np.where(val_sents == 0)[0]
        val_neg_id_idxs, val_neg_ood_idxs = self.get_ood_idxs(val_data, val_neg_idxs, self.num_source_classes, self.ood_class_ratio)

        val_pos_idxs = np.where(val_sents == 1)[0]
        val_pos_id_idxs, val_pos_ood_idxs = self.get_ood_idxs(val_data, val_pos_idxs, self.num_source_classes, self.ood_class_ratio)
        val_source_idxs, val_target_idxs = self.get_splits_from_idxs(val_data, val_neg_id_idxs, val_pos_id_idxs, source_marginal, source_classes, 125, shift=True)
        val_target_idxs = np.concatenate((val_target_idxs, val_neg_ood_idxs, val_pos_ood_idxs), axis=0, dtype=np.int64) 

        log.info("Creating training data ... ")
        self.labeled_source = Subset(train_data, train_source_idxs)    
        self.unlabeled_target = Subset(train_data, train_target_idxs)
        log.info("Done ")
        
        log.info("Creating validation data ... ")
        self.valid_labeled_source = Subset(val_data, val_source_idxs)
        self.valid_labeled_target = Subset(val_data, val_target_idxs)
        log.info("Done ")

        

        log.debug(f"OOD class {ood_class}, OOD subsample {self.ood_class_ratio}")
        log.debug("Stats of training data ... ")
        log.debug(f"Labeled source data {len(self.labeled_source)} and Unlabeled target samples {len(self.unlabeled_target)}")
        log.debug(f"OOD data {len(train_neg_ood_idxs) + len(train_pos_ood_idxs)}")
        log.debug(f"alpha {(len(train_neg_ood_idxs) + len(train_pos_ood_idxs))/len(self.unlabeled_target)}")
        log.debug(f"Source marginal {source_marginal}")
        log.debug("Stats of validation data ... ")
        log.debug(f"Labeled source data {len(self.valid_labeled_source)} and Labeled target data {len(self.valid_labeled_target)} ")
        log.debug(f"OOD data {len(val_neg_ood_idxs) + len(val_pos_ood_idxs)}")
        log.debug(f"alpha {(len(val_neg_ood_idxs) + len(val_pos_ood_idxs))/len(self.valid_labeled_target)}")
        
        # target_class_avg_dists, target_avg_dists, target_ood_dists, target_id_dists = self.get_kNN_scores(self.labeled_source, self.unlabeled_target, self.valid_labeled_source, self.valid_labeled_target, 10, same_sets=False)
        # print('target knn anomaly class scores:',[round(i,3) for i in target_class_avg_dists])
        # source_class_avg_dists, source_avg_dist, _, _ = self.get_kNN_scores(self.labeled_source, self.labeled_source, self.valid_labeled_source, self.valid_labeled_source, 10, same_sets=True)
        # print('source knn anomaly class scores:',[round(i,3) for i in source_class_avg_dists])
        
        # plt.hist(target_id_dists.cpu().detach(), label='target known classes P_[T,[k]]', alpha=0.3)
        # plt.hist(target_ood_dists.cpu().detach(), label='target novel class P_[T,1]', alpha=0.9)
        # plt.hist(source_avg_dist.cpu().detach(), label='source known classes P_S', alpha=0.3)
        # plt.xlabel('Anomaly scores')
        # plt.ylabel('Frequency')
        # plt.legend(loc='upper right')
        # plt.title('Anomaly score histogram')
        # plt.savefig('/cis/home/schaud35/shiftpu/shiftpu/plots/amazon_reviews_'+str(self.seed)+'_'+str(self.ood_class)+'_'+str(self.ood_class_ratio)+'_histogram_w_shift.png')
        
        # import pdb; pdb.set_trace()

    def get_kNN_scores(self, source_data, target_data, val_source_data, val_target_data, k, same_sets=False):
        train_source = [source_data[i][0] for i in range(len(source_data))]
        val_source = [val_source_data[i][0] for i in range(len(val_source_data))]
        train_source.extend(val_source)
        source_tensor = torch.tensor(np.array(train_source), device='cuda:0' if torch.cuda.is_available() else 'cpu')

        source_labels = [source_data[i][1] for i in range(len(source_data))]
        val_source_labels = [val_source_data[i][1] for i in range(len(val_source_data))]
        source_labels.extend(val_source_labels)
        source_labels = torch.tensor(source_labels)

        source_sents = [source_data[i][2] for i in range(len(source_data))]
        val_source_sents = [val_source_data[i][2] for i in range(len(val_source_data))]
        source_sents.extend(val_source_sents)
        source_sents = torch.tensor(source_sents)

        train_target = [target_data[i][0] for i in range(len(target_data))]
        val_target = [val_target_data[i][0] for i in range(len(val_target_data))]
        train_target.extend(val_target)
        target_tensor = torch.tensor(np.array(train_target), device='cuda:0' if torch.cuda.is_available() else 'cpu')

        target_labels = [target_data[i][1] for i in range(len(target_data))]
        val_target_labels = [val_target_data[i][1] for i in range(len(val_target_data))]
        target_labels.extend(val_target_labels)
        target_labels = torch.tensor(target_labels)

        target_sents = [target_data[i][2] for i in range(len(target_data))]
        val_target_sents = [val_target_data[i][2] for i in range(len(val_target_data))]
        target_sents.extend(val_target_sents)
        target_sents = torch.tensor(target_sents)

        # Calculate pairwise distances between target and source samples
        distances = torch.cdist(target_tensor, source_tensor)
        # Calculate cosine similarity between target and source samples
        # similarities = F.cosine_similarity(target_tensor.unsqueeze(1), source_tensor.unsqueeze(0), dim=2)

        # Get the indices of the k-nearest neighbors for each target sample
        if same_sets:
            _, indices = torch.topk(distances, k+1, largest=False)
            indices = indices[:,1:]
        else:
            _, indices = torch.topk(distances, k, largest=False)
        # _, cosine_indices = torch.topk(similarities, k, largest=True)
        
        # Gather the k-nearest neighbors from the source dataset
        nearest_neighbors = torch.gather(source_tensor.unsqueeze(0).expand(target_tensor.size(0), -1, -1), 1, indices.unsqueeze(2).expand(-1, -1, source_tensor.size(1)))        
        # cosine_nearest_neighbors = torch.gather(source_tensor.unsqueeze(0).expand(target_tensor.size(0), -1, -1), 1, cosine_indices.unsqueeze(2).expand(-1, -1, source_tensor.size(1)))        

        # Calculate the average distance for each target sample
        avg_distances = torch.mean(torch.norm(nearest_neighbors - target_tensor.unsqueeze(1), dim=2), dim=1)
        # cosine_avg_distances = torch.mean(torch.norm(cosine_nearest_neighbors - target_tensor.unsqueeze(1), dim=2), dim=1)
        class_avg_dists = [torch.mean(avg_distances[target_labels==i]).cpu().detach().tolist() for i in target_labels.unique()]
        
        val_target_labels = torch.tensor(val_target_labels)
        novel_labels = torch.zeros_like(val_target_labels)
        novel_labels[val_target_labels==val_target_labels.unique()[-1]] = 1.
        auroc = roc_auc_score(novel_labels, avg_distances[-len(val_target_labels):].cpu().detach())
        auprc = average_precision_score(novel_labels, avg_distances[-len(val_target_labels):].cpu().detach())
        print('auroc:', auroc, 'auprc:', auprc)

        # plot histogram
        novel_labels = torch.zeros_like(target_labels)
        novel_labels[target_labels==target_labels.unique()[-1]] = 1.
        ood_dists = avg_distances[novel_labels==1.]
        id_dists = avg_distances[novel_labels!=1.]
        

        return class_avg_dists, avg_distances, ood_dists, id_dists #, cosine_avg_distances


