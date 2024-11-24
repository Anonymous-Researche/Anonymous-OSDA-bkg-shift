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
import pandas as pd

class SubpopulationShiftDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        dataset: str = "imagenet", 
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
        preprocess=None,
        ood_subclass=None,
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
        self.ood_subclass = ood_subclass

        ## Fix this to avoid exploding importance weights
        self.min_source_fraction = 0.2 #0.2 #0.01 #0.1
        if preprocess:
            mean, std = preprocess.transforms[-1].mean, preprocess.transforms[-1].std
        else:
            mean, std = None, None
        self.train_transform = get_preprocessing(self.dataset, self.use_aug, train=True, mean=None, std=None, arch=arch)
        self.test_transform = get_preprocessing(self.dataset, self.use_aug, train=False, mean=None, std=None, arch=arch) 
        self.seed = seed
        self.mode = mode
        self.arch = arch
        self.use_superclass = use_superclass
        self.ood_class = ood_class

    def select_ood_samples(self, dataset, ood_class, ood_class_ratio):
        random.seed(self.seed)
        
        if isinstance(dataset, Subset):
            ood_idxs = np.where([target in ood_class for target in np.array(dataset.dataset.targets)])[0]
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
        train_data, val_data = get_combined_data(self.data_dir, self.dataset, arch=self.arch, \
            transform=[self.train_transform, self.test_transform],\
            train_fraction=self.train_fraction, seed=self.seed, mode=self.mode, use_superclass=self.use_superclass)

        print('seed in datamodule: {}'.format(self.seed))
        
        # ImageNet classes for covariate shift
        
        # self.source_classes = np.unique(self.source_supertargets)[:int(self.num_source_classes)]
        # self.source_subclasses = [subclass for subclass in np.unique(train_data.targets) if train_data.target_to_supertarget[subclass] in self.source_classes]
        # self.unknown_classes = np.unique(self.source_supertargets)[int(self.num_source_classes):]
        # self.unknown_subclasses = [subclass for subclass in np.unique(train_data.targets) if train_data.target_to_supertarget[subclass] in self.unknown_classes]
        
        # ood_subclass = random.sample(self.unknown_subclasses, int(self.fraction_ood_class*len(self.source_subclasses)/(1-self.fraction_ood_class)))
        # ood_class = np.unique([train_data.target_to_supertarget[subclass] for subclass in ood_subclass])
        # ood_class = self.ood_class
        # ood_subclass = [subclass for subclass in np.unique(train_data.targets) if train_data.target_to_supertarget[subclass] == ood_class]
        
        # train_data.supertarget_transform = lambda x: x if x in self.source_classes else len(self.source_classes)
        # val_data.supertarget_transform = lambda x: x if x in self.source_classes else len(self.source_classes)
        
        source_hierarchies = pd.read_csv('/cis/home/schaud35/shiftpu/shiftpu/src/sun397_hierarchy.csv', header=None)
        self.source_classes = [[train_data.dataset.class_to_idx[row[3:]] for row in source_hierarchies[col] if isinstance(row, str)] for col in source_hierarchies.columns]
        self.source_classes = self.source_classes[:self.num_source_classes]
        # for col in source_hierarchies.columns:
        #     self.source_classes.append([row[3:] for row in source_hierarchies[col] if isinstance(row, str)])
        ood_sun397_outdoors = pd.read_csv('/cis/home/schaud35/shiftpu/shiftpu/src/sun397_ood.csv', header=None)
        ood_classes = [[train_data.dataset.class_to_idx[row[3:]] for row in ood_sun397_outdoors[col] if isinstance(row, str)] for col in ood_sun397_outdoors.columns]
        ood_class = ood_classes[self.ood_class]
        ood_subclass = random.sample(ood_class, int(self.fraction_ood_class*len(ood_class)))
        
        # self.source_classes = [[train_data.dataset.class_to_idx[subclass] for subclass in superclass] for superclass in self.source_classes]
        superclass_mapping = {subclass: superclass for superclass, subclasses in enumerate(self.source_classes) for subclass in subclasses}
        self.source_subclasses = [subclass for superclass in self.source_classes for subclass in superclass]
        # self.ood_subclass = ['bazaar/outdoor', 'arrival_gate/outdoor', 'boardwalk', 'mountain']
        # ood_subclass = [train_data.dataset.class_to_idx[ood_subcls] for ood_subcls in self.ood_subclass] 
        for ood_subcls in ood_subclass:
            superclass_mapping[ood_subcls] = len(self.source_classes)
        if isinstance(train_data, Subset) and isinstance(val_data, Subset):
            train_data.dataset.class_to_superclass, val_data.dataset.class_to_superclass = superclass_mapping, superclass_mapping
        elif not isinstance(train_data, Subset) and not isinstance(val_data, Subset):
            train_data.class_to_superclass, val_data.class_to_superclass = superclass_mapping, superclass_mapping
        else:
            raise TypeError("Expected both train and val data to be Subset or not Subset")

        # Shift
        # self.source_marginal = np.round(np.random.uniform(self.min_source_fraction, 1.0, len(self.source_subclasses)), 2)
        lower_source_marginal = np.round(np.random.uniform(0.08, 0.15, len(self.source_subclasses)), 4)
        higher_source_marginal = np.round(np.random.uniform(0.85, 0.92, len(self.source_subclasses)), 4)
        self.source_marginal = np.array(random.sample(list(np.concatenate((lower_source_marginal, higher_source_marginal),axis=0)),len(self.source_subclasses)))

        # No shift
        # self.source_marginal = np.round(np.random.uniform(0.45, 0.5, len(self.source_subclasses)), 4)

        
        self.target_subclasses = self.source_subclasses.copy()
        self.target_subclasses.append(list(ood_subclass))

        self.target_marginal = 1.0 - self.source_marginal
        self.target_marginal =  np.concatenate((self.target_marginal, np.array([1.0]*len(ood_subclass))))
        
        log.debug(f"Source classes: {self.source_subclasses}")
        log.debug(f"Source marginal: {self.source_marginal}")
        # log.debug(f"Source marginal validation: {self.source_marginal_valid}")

        log.debug(f"Target classes: {self.target_subclasses}")
        log.debug(f"Target marginal: {self.target_marginal}")
        # log.debug(f"Target marginal validation: {self.target_marginal_valid}")

        log.info("Creating training data ... ")
        
        # Use train_data.dataset for full dataset (used in paper for test time methods like CoLOR/CoNoC). Otherwise for train test split use train_data (Subset).
        self.labeled_source, self.unlabeled_target, self.source_idx, self.target_idx, self.target_transform =\
            get_splits_from_data(train_data.dataset,\
            source_classes = self.source_subclasses, source_marginal =self.source_marginal, \
            target_classes=self.target_subclasses, target_marginal=self.target_marginal, dataset=self.dataset)

        # # Active Learning dataset without Random torchvision transforms
        # _, self.unlabeled_AL_pool, _, _, _ =\
        #     get_splits_from_data(train_data_AL,\
        #     source_classes = self.source_subclasses, source_marginal =self.source_marginal, \
        #     target_classes=self.target_classes, target_marginal=self.target_marginal)
        
        log.info("Done ")
        
        log.info("Creating validation data ... ")
        
        # Use val_data.dataset for full dataset (used in paper for test time methods like CoLOR/CoNoC). Otherwise for train test split use val_data (Subset).
        self.valid_labeled_source, self.valid_labeled_target, _, _, _ = \
            get_splits_from_data(val_data.dataset, \
            source_classes = self.source_subclasses, source_marginal =self.source_marginal, \
            target_classes=self.target_subclasses, target_marginal=self.target_marginal, dataset=self.dataset)
        
        self.labeled_source, source_ood, source_id, source_total = self.select_ood_samples(self.labeled_source, ood_subclass, self.ood_class_ratio)
        self.unlabeled_target, target_ood, target_id, target_total = self.select_ood_samples(self.unlabeled_target, ood_subclass, self.ood_class_ratio)         
        
        self.valid_labeled_source, val_source_ood, val_source_id, val_source_total = self.select_ood_samples(self.valid_labeled_source, ood_subclass, self.ood_class_ratio)
        self.valid_labeled_target, val_target_ood, val_target_id, val_target_total = self.select_ood_samples(self.valid_labeled_target, ood_subclass, self.ood_class_ratio)

        # self.unlabeled_AL_pool.indices = self.unlabeled_target.indices

        log.info("Done ")
        log.debug(f"OOD class {ood_subclass}, OOD subsample {self.ood_class_ratio}")
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
        
        # TSNE plotting
        # vls = np.array([self.valid_labeled_source[i][0].cpu().detach().numpy() for i in range(len(self.valid_labeled_source))])
        # vlt = np.array([self.valid_labeled_target[i][0].cpu().detach().numpy() for i in range(len(self.valid_labeled_target))])
        # vl = np.concatenate((vls, vlt), axis=0)
        # gts_super = np.array([self.valid_labeled_source[i][1] for i in range(len(self.valid_labeled_source))])
        # gtt_super = np.array([self.valid_labeled_target[i][1] for i in range(len(self.valid_labeled_target))])
        # gt_super = np.concatenate((gts_super, gtt_super), axis=0)
        # gt = np.concatenate((np.zeros_like(gts_super), np.ones_like(gtt_super)), axis=0)
        # gt[gt_super==np.unique(gt_super)[-1]]=2
        # gts = np.array([self.valid_labeled_source[i][2] for i in range(len(self.valid_labeled_source))])
        # gtt = np.array([self.valid_labeled_target[i][2] for i in range(len(self.valid_labeled_target))])

        # tsne_vls = compute_tsne(vls, perplexity=50, n_iter=5000, n_jobs=8)
        # tsne_vlt = compute_tsne(vlt, perplexity=50, n_iter=5000, n_jobs=8)
        # tsne_vl = compute_tsne(vl, perplexity=50, n_iter=5000, n_jobs=8)
        # plot_2d_scatterplot(tsne_vls, gts, num_classes=len(np.unique(gts)), save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/tsne_CLIP_ImageNet_valid_source_'+str(len(np.unique(gts)))+'classes.png')
        # plot_2d_scatterplot(tsne_vlt, gtt, num_classes=len(np.unique(gtt)), save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/tsne_CLIP_ImageNet_valid_target_'+str(len(np.unique(gtt)))+'classes.png')
        # plot_2d_scatterplot(tsne_vls, gts_super, num_classes=len(np.unique(gts_super)), save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/tsne_CLIP_ImageNet_valid_source_'+str(len(np.unique(gts_super)))+'_super_classes.png')
        # plot_2d_scatterplot(tsne_vlt, gtt_super, num_classes=len(np.unique(gtt_super)), save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/tsne_CLIP_ImageNet_valid_target_'+str(len(np.unique(gtt_super)))+'_super_classes.png')
        # plot_2d_scatterplot(tsne_vl, gt, num_classes=len(np.unique(gt)), save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/tsne_CLIP_ImageNet_valid_full.png')
        # plot_2d_scatterplot(tsne_vl, gt_super, num_classes=len(np.unique(gt_super)), save_plt_path='/cis/home/schaud35/shiftpu/shiftpu/plots/tsne_CLIP_ImageNet_valid_full_'+str(len(np.unique(gt_super)))+'_super_classes.png')
        # import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
        st_df = self.save_dataset(self.labeled_source, name='source_train_w_shift')
        tt_df = self.save_dataset(self.unlabeled_target, name='target_train_w_shift')
        sv_df = self.save_dataset(self.valid_labeled_source, name='source_val_w_shift')
        tv_df = self.save_dataset(self.valid_labeled_target, name='target_val_w_shift')
        exit()

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

    def save_dataset(self, dataset, name='source_tr'):
        data_dir = '/cis/home/schaud35/shiftpu/shiftpu/data_splits/sun397/'
        idx_ls = np.zeros((len(dataset)), dtype=np.int32)
        super_target_ls = np.zeros((len(dataset)), dtype=np.int32)
        target_ls = np.zeros((len(dataset)), dtype=np.int32)
        image_file_ls = []
        for i,data in enumerate(dataset):
            idx = dataset.indices[i]
            feature, super_target, target  = data[0], data[1], data[2]
            image_file_ls.append(dataset.dataset._image_files[idx])
            feature = feature.cpu().detach().numpy()
            super_target = super_target
            idx_ls[i] = idx
            target = target
            target_ls[i] = target
            super_target_ls[i] = super_target
        
        data_dir = os.path.join(data_dir, str(self.seed)+'_'+str(self.ood_class)+'_'+str(self.ood_class_ratio)+'_'+str(self.fraction_ood_class))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        df = pd.DataFrame({'idx': idx_ls, 'image_file': image_file_ls, 'super_target': super_target_ls, 'target': target_ls})
        df.to_csv(os.path.join(data_dir,name+'_split.csv'))
        return df
            # np.save('image_file'image_file, feature=feature, super_target=super_target, target=target)
        