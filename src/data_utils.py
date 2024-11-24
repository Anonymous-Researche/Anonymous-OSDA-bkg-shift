from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder, ImageNet, SUN397
from torchvision.models import resnet18, ResNet18_Weights
import torchvision
from torchvision import transforms
from typing import Callable, Optional, List
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import torch
from src.simple_utils import load_pickle
import pathlib
import json
import os
import logging 
from collections import defaultdict, Counter
from src.datasets.tabula_munis.dataset import *
from wilds.datasets.rxrx1_dataset import RxRx1Dataset
import torchvision.transforms.functional as TF
import random
from transformers import RobertaTokenizer

from src.datasets.newsgroups_utils import *
from src.datasets.amazon_reviews_utils import *
import torch
import torch.multiprocessing
from tqdm import tqdm
from sarpu.labeling_mechanisms import label_data
from sarpu.experiments import *
from sarpu.evaluation import *
from sarpu.input_output import *
from sarpu.labeling_mechanisms import parse_labeling_model
from sarpu.paths_and_names import *
from sarpu.pu_learning import *

import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
torch.multiprocessing.set_sharing_strategy('file_system')

# log = logging.getLogger(__name__)
log = logging.getLogger("app")

osj = os.path.join

class SUN397Dataset(SUN397):
    def __init__(self, root, features=None, transform=None):
        super(SUN397Dataset, self).__init__(root, transform=transform)
        self.class_to_superclass = {}
        self.features = features
        self.targets = self._labels
        self.supertarget_transform = None
        if self.features is not None:
            self.input_features = torch.load(features)
            if 'supertargets' in self.input_features.keys():
                self.input_features.pop('supertargets')
            for key,value in self.input_features.items():
                self.input_features[key] = value.cpu().detach()
            
    def __getitem__(self,index):
        if self.features is not None:
            sample, target, feat_index = self.input_features['features'][index], int(self.input_features['targets'][index].item()), int(self.input_features['indices'][index].item())
            assert feat_index==index
        else:
            sample, target = super(SUN397Dataset, self).__getitem__(index)
        if len(self.class_to_superclass)>0:
            supertarget = self.class_to_superclass[target]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.supertarget_transform is not None:
            supertarget = self.super_target_transform(supertarget)
        
        return sample, supertarget, target

class ImageNetDataset(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""
    def __init__(self, root, transform=None, features=None):
        super(ImageNetDataset, self).__init__(root, transform=transform)
        # self.superclass_mapping = {'n014': 'fish', 'n015': 'bird', 'n016': 'plant', 
        #                            'n017': 'plant', 'n018': 'plant', 'n019': 'plant', 
        #                            'n020': 'plant', 'n021': 'plant', 'n022': 'plant', 
        #                            'n023': 'plant', 'n024': 'plant', 'n025': 'plant', 
        #                            'n026': 'plant', 'n027': 'plant', 'n028': 'plant',}
        self.class_to_super_idx = {} 
        self.class_to_superclass = {}
        self.target_to_supertarget = {}
        self.target_transform = None
        self.supertarget_transform = None
        self.supertargets = []
        self.features = features
        self.transform = transform
        
        for (idx,(path, target)) in enumerate(self.samples):
            primary_class = path.split('/')[-2]
            if primary_class[:4] not in self.class_to_super_idx.keys():
                self.class_to_super_idx[primary_class[:4]] = len(self.class_to_super_idx) 
            supertarget = self.class_to_super_idx[primary_class[:4]]
            self.supertargets.append(supertarget)
            self.samples[idx] = (path, supertarget, target)
            self.target_to_supertarget[target] = supertarget 

        if self.features is not None:
            self.input_features = torch.load(features)
            for key,value in self.input_features.items():
                self.input_features[key] = value.cpu().detach()
        
        # import pdb; pdb.set_trace()
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, subclass target) where target is class_index of the target class.
        """
        # sample, target = super(ImageNetDataset, self).__getitem__(index)
        
        if self.features is not None:
            # print(self.input_features['features'][index].shape, self.input_features['supertargets'][index], self.input_features['targets'][index])
            sample, supertarget, target = self.input_features['features'][index], int(self.input_features['supertargets'][index].item()), int(self.input_features['targets'][index].item())
        else:
            path, supertarget, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
                
        assert target==self.samples[index][2]
        if len(self.class_to_superclass)>0:
            supertarget = self.class_to_superclass[target]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.supertarget_transform is not None:
            supertarget = self.supertarget_transform(supertarget)
        
        return sample, supertarget, target 


class ng20Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, classification_attributes, propensity_attributes):
        self.data = data
        self.targets = np.array(targets).astype(np.int_)
        self.classification_attributes = classification_attributes
        self.propensity_attributes = propensity_attributes
        self.target_transform = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].type(torch.float32), self.targets[idx]

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = np.array(targets).astype(np.int_)

        self.target_transform = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class TabulaMurisDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, age, sex, tissue):
        self.data = data
        self.targets = np.array(targets).astype(np.int_)
        self.age = np.array(age).astype(np.int_)
        self.sex = np.array(sex).astype(np.int_)
        self.tissue = np.array(tissue).astype(np.int_)
        self.label_map = {label:label for label in np.unique(self.targets)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label_map[self.targets[idx]], self.age[idx], self.sex[idx], self.tissue[idx]


class RobertaSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = np.array(targets).astype(np.int_)

        self.target_transform = None
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(self.data[idx], None, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_token_type_ids=True)
        data = {'input_ids':torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask':torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids':torch.tensor(inputs['token_type_ids'], dtype=torch.long)
            } 
        return data, self.targets[idx]

class ResNetSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets.cpu().detach().numpy().astype(np.int_)

        # self.target_transform = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        return self.data[idx], self.targets[idx]

class DatasetwithSentiments(torch.utils.data.Dataset):
    def __init__(self, data, targets, sentiments, arch):
        self.data = data
        self.targets = np.array(targets).astype(np.int_)
        self.sentiments = np.array(sentiments).astype(np.int_)
        self.arch = arch
        self.target_transform = None
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if self.arch=='Roberta:':
            inputs = self.tokenizer.encode_plus(self.data[idx], None, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_token_type_ids=True) 
            data = {'input_ids':torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask':torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids':torch.tensor(inputs['token_type_ids'], dtype=torch.long)
            } 
            return self.data, self.targets[idx], self.sentiments[idx]

        return self.data[idx], self.targets[idx], self.sentiments[idx]


# class AmazonReviewsRobertaFeatures(torch.utils.data.Dataset):
#     def __init__(self, data_dir, targets, sentiments, arch):

def get_labels(targets): 
    counter = Counter(targets)
    return sorted(list(counter.keys()))

def get_size_per_class(dataset):

    if isinstance(dataset, Subset):
        targets = np.array(dataset.dataset.targets)[dataset.indices] 
        counter = Counter(targets)
    else: 
        counter = Counter(dataset.targets)

    return counter


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    
    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        transform_idx = self.transform_idx
        return (data[0], data[1], transform_idx[index]) + data[2:]

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def dataset_transform_labels(cls): 

    def __getitem__(self, index):
        
        data = cls.__getitem__(self, index)
        
        return (data[0], self.target_transform(data[1])) + data[2:]
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def get_data(data_dir, dataset, train = None, transform=None): 

    if dataset.lower() == "cifar10":
        CIFAR10withIndices = dataset_with_indices(CIFAR10)
        data = CIFAR10withIndices(root = f"{data_dir}/cifar10", train=train, transform=transform, download=True)
        
        return data

    elif dataset.lower() == "cifar100":
        CIFAR100withIndices = dataset_with_indices(CIFAR100)
        data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=train, transform=transform, download=True)    

        return data

    elif dataset.lower() == "mnist": 
        MNISTwithIndices = dataset_with_indices(MNIST)
        data = MNISTwithIndices(root = data_dir, train=train, transform=transform, download=True)

        return data
    
    elif dataset.lower() == "imagenet":
        IMAGENETwithIndices = dataset_with_indices(ImageNet)
        data = IMAGENETwithIndices(root = f"{data_dir}/imagenet", train=train, transform=transform, download=True)
        return data

    else: 
        raise NotImplementedError("Please add support for %s dataset" % dataset)
    

def get_combined_data(data_dir, dataset, arch, ood_class=None, transform=None, train_fraction = None , seed=42, mode='domain_disc', use_superclass=False):
    np.random.seed(seed)
    if dataset.lower() == "cifar10":
        CIFAR10withIndices = dataset_with_indices(CIFAR10)
        train_data = CIFAR10withIndices(root = f"{data_dir}/cifar10", train=True, transform=transform[0], download=True)
        val_data = CIFAR10withIndices(root = f"{data_dir}/cifar10", train=False, transform=transform[1], download=True)
        train_data_AL = CIFAR10withIndices(root = f"{data_dir}/cifar10", train=True, transform=transform[1], download=True)
        # random.seed(seed)
        
        # domain_disc_train_idxs = random.sample(range(0,len(train_data)), len(train_data)//2)
        # constrained_opt_train_idxs = np.setdiff1d(list(range(0,len(train_data))), domain_disc_train_idxs)
        # domain_disc_val_idxs = random.sample(range(0,len(val_data)), len(val_data)//2)
        # constrained_opt_val_idxs = np.setdiff1d(list(range(0,len(train_data))), domain_disc_val_idxs)
        
        # domain_disc_train_data = torch.utils.data.Subset(train_data, domain_disc_train_idxs)
        # domain_disc_val_data = torch.utils.data.Subset(val_data, domain_disc_val_idxs)
        # constrained_opt_train_data = torch.utils.data.Subset(train_data, constrained_opt_train_idxs)
        # constrained_opt_val_data = torch.utils.data.Subset(val_data, constrained_opt_val_idxs)
        
        # if mode=='domain_disc':
        #     return domain_disc_train_data.dataset, domain_disc_val_data.dataset
        # elif mode=='constrained_opt': 
        #     return constrained_opt_train_data.dataset, constrained_opt_val_data.dataset
        # else:
        #     return train_data, val_data
        return train_data, val_data, train_data_AL

    if dataset.lower() == "cifar20":

        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
            6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
            0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
            5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
            16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
            10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
            2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
            16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
            18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

        CIFAR10withIndices = dataset_with_indices(CIFAR100)
        train_data = CIFAR10withIndices(root = f"{data_dir}/cifar100", train=True, transform=transform[0], download=True)
        val_data = CIFAR10withIndices(root = f"{data_dir}/cifar100", train=False, transform=transform[1],  download=True)

        train_data.targets = [coarse_labels[i] for i in train_data.targets]
        val_data.targets = [coarse_labels[i] for i in val_data.targets]

        return train_data, val_data

    elif dataset.lower() == "cifar100":
        CIFAR100withIndices = dataset_with_indices(CIFAR100)
        train_data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=True, transform=transform[0], download=True)    
        val_data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=False, transform=transform[1], download=True)
        train_data_AL = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=True, transform=transform[1], download=True)

        # ResNet18 pretrained features
        # CIFAR100withIndices = dataset_with_indices(ResNetSimpleDataset)
        # # pretrained = torch.load("/export/r36a/data/schaud35/shiftpu/cifar100_ResNet18_features.pth")
        # pretrained = torch.load("/export/r36a/data/schaud35/shiftpu/cifar100_ResNet18_features_imagenet_preprocessing.pth")
        # train_feats = pretrained['features'][:-10000]
        # val_feats = pretrained['features'][-10000:]
        # train_targets = pretrained['targets'][:-10000]
        # val_targets = pretrained['targets'][-10000:]
        # train_data = CIFAR100withIndices(train_feats, train_targets)
        # val_data = CIFAR100withIndices(val_feats, val_targets)
        # train_data_AL = CIFAR100withIndices(train_feats, train_targets)
        
        return train_data, val_data, train_data_AL

    elif dataset.lower() == "20ng":
        
        data_folder= "/cis/home/schaud35/Data/"
        results_folder="/cis/home/schaud35/shiftpu/SAR-EM_Results/"

        data_name = "20ng"
        propensity_attributes = [111,112,113,114]
        propensity_attributes_signs = [1,1,1,1]
        settings = "lr._.lr._.0-111"
        labeling_model_type = "simple_0.2_0.8"

        labeling=0
        partition=1
        nb_assignments=5
        nb_labelings=5

        relabel_data = False
        rerun_experiments = False
        labeling_model = label_data(
            data_folder, 
            data_name, 
            labeling_model_type, 
            propensity_attributes, 
            propensity_attributes_signs, 
            nb_assignments,
            relabel_data=relabel_data
        )
        classification_model_type, propensity_model_type, classification_attributes = parse_settings(settings)
        # Load data
        x_path = data_path(data_folder,data_name)
        y_path = classlabels_path(data_folder,data_name)
        s_path = propensity_labeling_path(data_folder, data_name, labeling_model, labeling)
        e_path = propensity_scores_path(data_folder, data_name, labeling_model)
        f_path = partition_path(data_folder, data_name, partition)
        (x_train,y_train,s_train,e_train),(x_test,y_test,s_test,e_test) = read_data((x_path,y_path,s_path,e_path),f_path)
        c = (e_train[y_train==1]).mean()
        
        classification_attributes = classification_attributes(x_train)
        
        log.info("Training SAR-EM original baseline:")
        f_model, e_model, info = pu_learn_sar_em(x_train, s_train, propensity_attributes, classification_model=classification_model_type(), classification_attributes=classification_attributes, propensity_model=propensity_model_type())
        out_info = experiment_info_path(results_folder, data_name, labeling_model, labeling, partition, settings, 'sar-em')
        print('OUT', out_info)
        results_test = evaluate_all(y_test,s_test,e_test, f_model.predict_proba(x_test),e_model.predict_proba(x_test))
        results_train = evaluate_all(y_train,s_train,e_train, f_model.predict_proba(x_train),e_model.predict_proba(x_train))
        
        results = {
            **{"train_" + k: v for k, v in results_train.items()},
            **{"test_" + k: v for k, v in results_test.items()}
        }
        measures_to_plot = ['test_f_roc_auc','test_f_average_precision', 'test_f_mse', 'test_f_mae', 'test_e_mse','test_e_mae', 'train_e_prior_abs_err']
        
        for key in measures_to_plot:
            print(key, results[key]) 
        
        return x_train, y_train, s_train, x_test, y_test, s_test, labeling_model.propensity_attributes, classification_attributes, classification_model_type, propensity_model_type


    elif dataset.lower() == "mnist": 
        MNISTwithIndices = dataset_with_indices(MNIST)
        train_data = MNISTwithIndices(root = data_dir, train=True, transform=transform[0], download=True)
        val_data = MNISTwithIndices(root = data_dir, train=False, transform=transform[1], download=True)
        
        return train_data, val_data

    elif dataset.lower() == "tabula_muris":
        
        TabulaMuniswithIndices = dataset_with_indices(SimpleDataset)

        train_data = load_tabular_muris(root=f"{data_dir}/tabula-muris-comet/", mode="train")
        val_data = load_tabular_muris(root=f"{data_dir}/tabula-muris-comet/", mode="val")
        test_data = load_tabular_muris(root=f"{data_dir}/tabula-muris-comet/", mode="test")

        # samples = np.array(train_data[0])
        # targets = np.array(train_data[1])

        samples = np.concatenate((train_data[0], val_data[0], test_data[0]), axis=0)
        targets = np.concatenate((np.array(train_data[1]), np.array(val_data[1]) , np.array(test_data[1])), axis=0)
        vals_temp = np.unique(np.array(train_data[1]))
        counts_temp = np.bincount(np.array(train_data[1]))
        log.info('num_labels: {}'.format(len(vals_temp)))
        log.info('label_counts: {}'.format(counts_temp))

        # shift_attr = pd.concat([train_data[5], val_data[5], test_data[5]], axis=0)
        # shift_attr = targets.astype('category').cat.codes.to_numpy(dtype=np.int32)

        # samples = np.concatenate((train_data[0], val_data[0], test_data[0]), axis=0)
        # targets = np.concatenate((np.array(train_data[1]), np.array(val_data[1]) , np.array(test_data[1])), axis=0)
        # targets = np.concatenate((np.array(train_data[1]), np.array(val_data[1]) + 57 , np.array(test_data[1]) + 96), axis=0)

        labels = get_labels(targets)
        idx_per_class = []

        if use_superclass:
            age = pd.concat([train_data[3], val_data[3], test_data[3]], axis=0)
            age = age.astype('category').cat.codes.to_numpy(dtype=np.int32)
            sex = pd.concat([train_data[4], val_data[4], test_data[4]], axis=0)
            sex = sex.astype('category').cat.codes.to_numpy(dtype=np.int32)
            tissues = pd.concat([train_data[5], val_data[5], test_data[5]], axis=0)
            tissues = tissues.astype('category').cat.codes.to_numpy(dtype=np.int32)
            select_tissues = np.unique(tissues)
            select_attr = np.unique(age)
            
            for label in labels:
                for attr in select_attr:
                    idx_i = np.intersect1d(np.where(targets == label)[0], np.where(age == attr)[0])
                    np.random.shuffle(idx_i)
                    idx_per_class.append(idx_i)
            
            train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(idx_per_class))])   
            val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(idx_per_class))])
            
            train_data = TabulaMurisDataset(samples[train_idx], targets[train_idx], age[train_idx], sex[train_idx], tissues[train_idx])
            val_data = TabulaMurisDataset(samples[val_idx], targets[val_idx], age[val_idx], sex[val_idx], tissues[val_idx])
            train_data_AL = TabulaMurisDataset(samples[train_idx], targets[train_idx], age[train_idx], sex[train_idx], tissues[train_idx])
        
        else:
            for label in labels:
                idx_i = np.where(targets == label)[0]
                np.random.shuffle(idx_i)
                idx_per_class.append(idx_i)

        
            train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(idx_per_class))])   
            val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(idx_per_class))])

            train_data = TabulaMuniswithIndices(samples[train_idx], targets[train_idx])
            val_data = TabulaMuniswithIndices(samples[val_idx], targets[val_idx])
            train_data_AL = TabulaMuniswithIndices(samples[train_idx], targets[train_idx])
        
        return train_data, val_data, train_data_AL

    elif dataset.lower() == "dermnet":

        train_data = ImageFolder(root=f"{data_dir}/dermnet/train/", transform=transform[0])
        test_data = ImageFolder(root=f"{data_dir}/dermnet/train/", transform=transform[1])

        targets = np.array(train_data.targets)
        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
    

        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])   
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        DermnetwithIndices = dataset_with_indices(Subset)

        train_data = DermnetwithIndices(train_data, train_idx)
        val_data = DermnetwithIndices(test_data, val_idx)

        return train_data, val_data
    
    elif dataset.lower() == "imagenet":
        # train_data = ImageNetDataset(root = f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/train", transform=transform[0])
        # val_data = ImageNetDataset(root = f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/val", transform=transform[1])
        
        # # Pretrained CLIP features
        train_data = ImageNetDataset(root=f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/train", features=f"{data_dir}/train_imagenet_CLIP_RN50_features_imagenet_pretraining.pth")
        val_data = ImageNetDataset(root=f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/val", features=f"{data_dir}/val_imagenet_CLIP_RN50_features_imagenet_pretraining.pth")

        # # Pretrained ViTL16_SWAG features
        # train_data = ImageNetDataset(root=f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/train", features=f"{data_dir}/train_imagenet_ViTB16_SWAG_features_imagenet_pretraining.pth")
        # val_data = ImageNetDataset(root=f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/val", features=f"{data_dir}/val_imagenet_ViTB16_SWAG_features_imagenet_pretraining.pth")

        # Pretrained ResNet50 features
        # train_data = ImageNetDataset(root=f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/train", features=f"{data_dir}/train_imagenet_ResNet50_features_imagenet_pretraining.pth")
        # val_data = ImageNetDataset(root=f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/val", features=f"{data_dir}/val_imagenet_ResNet50_features_imagenet_pretraining.pth")

        # train_data = ImageFolder(root = f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/train")
        # val_data = ImageFolder(root = f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/val")
        return train_data, val_data
    
    elif dataset.lower() == "sun397":
        train_txt = pd.read_csv(f"{data_dir}/sun397/Partitions/Training_01.txt", header=None)
        test_txt = pd.read_csv(f"{data_dir}/sun397/Partitions/Testing_01.txt", header=None)
        train_files = [train_txt.iloc[i,0] for i in range(len(train_txt))]
        test_files = [test_txt.iloc[i,0] for i in range(len(test_txt))]
        
        if arch=='Resnet50':
            data = SUN397Dataset(root=f"{data_dir}/sun397", features=f"{data_dir}/train_"+dataset+"_ResNet50_features_imagenet_pretraining.pth")
        elif arch=='CLIP_RN50':
            data = SUN397Dataset(root=f"{data_dir}/sun397", features=f"{data_dir}/train_"+dataset+"_CLIP_RN50_features_pretrained.pth")
        elif arch=='CLIP_ViT-L14':
            data = SUN397Dataset(root=f"{data_dir}/sun397", features=f"{data_dir}/train_"+dataset+"_CLIP_ViT-L14_features_pretrained.pth")
        file_names = [i.__str__().split('SUN397')[-1] for i in data._image_files]
        train_idxs = [i for i in range(len(file_names)) if file_names[i] in train_files]
        val_idxs = [i for i in range(len(file_names)) if file_names[i] in test_files]
        
        # ResNet50 pretrained on ImageNet features over SUN397
        train_data = Subset(data, train_idxs)
        val_data = Subset(data, val_idxs)
        
        return train_data, val_data

    elif dataset.lower() == "breakhis":

        train_data = ImageFolder(root=f"{data_dir}/BreaKHis_v1/", transform=transform[0])
        test_data = ImageFolder(root=f"{data_dir}/BreaKHis_v1/", transform=transform[1])

        targets = np.array(train_data.targets)
        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
        
        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        BreakHiswithIndices = dataset_with_indices(Subset)

        train_data = BreakHiswithIndices(train_data, train_idx)
        val_data = BreakHiswithIndices(test_data, val_idx)

        return train_data, val_data
    
    elif dataset.lower() == "newsgroups":  
        
        if arch=='Roberta': 
            NewsgroupswithIndices = dataset_with_indices(RobertaSimpleDataset)
        else:
            NewsgroupswithIndices = dataset_with_indices(SimpleDataset)
        
        data, targets, _ = get_newsgroups(arch)
        labels = get_labels(targets)

        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
        
        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])   
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        train_data = NewsgroupswithIndices(data[train_idx], targets[train_idx])
        val_data = NewsgroupswithIndices(data[val_idx], targets[val_idx])
        train_data_AL = NewsgroupswithIndices(data[train_idx], targets[train_idx])

        return train_data, val_data, train_data_AL
    
    elif dataset.lower() == "amazon_reviews":
        
        if arch=="Roberta_linear_classifier":
            data, targets, sentiments, _ = get_amazon_reviews_features(f"{data_dir}/amazon_reviews_roberta_features.pth", ood_class, arch)
        else:
            data, targets, sentiments, _ = get_amazon_reviews(f"{data_dir}/amazon_reviews_tp", ood_class, arch)
        labels = get_labels(targets)
        sentiment_labels = get_labels(sentiments)

        idx_per_class, sent_idx = [],[]
        
        for label in labels:
            for sent_label in sentiment_labels:
                idx_i = np.intersect1d(np.where(targets == label)[0], np.where(sentiments == sent_label)[0])
                np.random.shuffle(idx_i)
                idx_per_class.append(idx_i)

        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(idx_per_class))])   
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(idx_per_class))])

        train_data = DatasetwithSentiments(data[train_idx], targets[train_idx], sentiments[train_idx], arch)
        val_data = DatasetwithSentiments(data[val_idx], targets[val_idx], sentiments[val_idx], arch)
        train_data_AL = DatasetwithSentiments(data[train_idx], targets[train_idx], sentiments[train_idx], arch)

        return train_data, val_data, train_data_AL

    elif dataset.lower() == "entity30":
        
        from robustness.tools.helpers import get_label_mapping
        from robustness.tools import folder
        from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26

        ret = make_entity30(f"{data_dir}/Imagenet-resize/ImageNet_hierarchy/", split="good")

        label_mapping = get_label_mapping('custom_imagenet', ret[1][0]) 

        train_data = folder.ImageFolder(root=f"{data_dir}/Imagenet-resize/imagenet/train/", transform = transform[0], label_mapping = label_mapping)
        test_data = folder.ImageFolder(root=f"{data_dir}/Imagenet-resize/imagenet/train/", transform = transform[1], label_mapping = label_mapping)

        targets = np.array(train_data.targets)
        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
        
        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        Entity30withIndices = dataset_with_indices(Subset)

        train_data = Entity30withIndices(train_data, train_idx)
        val_data = Entity30withIndices(test_data, val_idx)

        return train_data, val_data
    
    elif dataset.lower() == "utkface":

        train_data = ImageFolder(root=f"{data_dir}/UTKDataset", transform=transform[0])
        test_data = ImageFolder(root=f"{data_dir}/UTKDataset/", transform=transform[1])

        targets = np.array(train_data.targets)
        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
        
        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        UTKwithIndices = dataset_with_indices(Subset)

        train_data = UTKwithIndices(train_data, train_idx)
        val_data = UTKwithIndices(test_data, val_idx)

        return train_data, val_data
    
    elif dataset.lower() == "rxrx1":

        data = RxRx1Dataset(download=False, root_dir=data_dir)
        train_data = data.get_subset('train', transform = transform[0])
        val_data = data.get_subset('id_test', transform = transform[1])

        train_data.targets = train_data.y_array.numpy()
        val_data.targets = val_data.y_array.numpy()

        targets = np.array(train_data.y_array.numpy())

        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
        
        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        RxRx1withIndices = dataset_with_indices(Subset)

        train_data = RxRx1withIndices(train_data, train_idx)
        val_data = RxRx1withIndices(val_data, val_idx)

        return train_data, val_data
    else: 
        raise NotImplementedError("Please add support for %s dataset" % dataset)


def get_classes(classes : List): 
    if isinstance(classes[0], list):
        return [list(map(int, i)) for i in classes]
    else:  
        return list(map(int, classes))

def get_marginal(marginal_type: str, marginal:  List[int], num_classes: int): 
    if marginal_type == "Uniform": 
        return np.array([1.0/num_classes]*num_classes)
    elif marginal_type == "Dirichlet": 
        return np.random.dirichlet(marginal[0]*num_classes)
    elif marginal_type == "Manual":
        marginal =  np.array(marginal)
        assert np.sum(marginal) == 1.0
        return marginal
    else: 
        raise NotImplementedError("Please check your marginal type for source and target")


def get_idx(targets, classes, total_per_class):

    idx = None
    log.debug(f"Target length {len(targets)} of type {type(targets)} and elements are {targets[:50]}...")
    targets = np.array(targets)
    for i in range(len(classes)):
        c_idx = None
        if isinstance(classes[i], list): 
            log.debug(f"Class {i} is a list {classes[i]}")
            for j in classes[i]:
                log.debug(f"Class {i} has {type(j)} {j}")
                if c_idx is None: 
                    c_idx = np.where(j == targets)[0]
                else: 
                    c_idx = np.concatenate((c_idx, np.where(j == targets)[0]))
            log.debug(f"Number of instances for class {i} are {len(c_idx)}")
        else: 
            log.debug(f"Class {i} is a {type(classes[i])} {classes[i]}")
            c_idx = np.where(classes[i] == targets)[0]
            log.debug(f"Number of instances for class {i} are {len(c_idx)}")

        if len(c_idx) >= total_per_class[i]:     
            c_idx = np.random.choice(c_idx, size = total_per_class[i], replace= False)
        else: 
            log.error("Not enough samples to get the split for class %d. \n\
                       Needed %f. Obtained %f" %(i, total_per_class[i], len(c_idx)))
        
        if idx is None:
            idx = [c_idx]
        else: 
            idx.append(c_idx) 

    label_map = {}
    for i in range(len(classes)): 
        if isinstance(classes[i], list): 
            for j in classes[i]:
                label_map[j] = i
        else: 
            label_map[classes[i]] = i
    
    log.debug(label_map)
    target_transform = lambda x: label_map[x]
    
    return idx, target_transform


def split_indicies(targets, source_classes, target_classes,\
     source_marginal, target_marginal, source_size, target_size): 

    source_per_class = np.concatenate((np.array([ int(i*source_size) for i in source_marginal]),\
         np.array([0], dtype=np.int32)))
    target_per_class = np.array([ int(i*target_size) for i in target_marginal])

    total_per_class = source_per_class + target_per_class

    log.debug(f"Needed <{source_per_class}> samples for source")
    log.debug(f"Needed <{target_per_class}> samples for target")
    
    idx, target_transform = get_idx(targets, target_classes, total_per_class)
    
    source_idx = [idx[c][:source_per_class[c]] for c in range(len(source_classes))]
    target_idx = [idx[c][source_per_class[c]:] for c in range(len(target_classes))]

    return source_idx, target_idx, target_transform


def split_indicies_with_size(targets, source_classes, target_classes,
                             source_marginal, target_marginal, size_per_class):

    source_per_class = np.concatenate((np.array([ int(source_marginal[class_idx]*size_per_class[i]) for  class_idx, i in enumerate(source_classes)]),\
            np.array([0], dtype=np.int32)))

    target_per_class = np.array([ int(target_marginal[class_idx]*size_per_class[i]) for  class_idx, i in enumerate(target_classes[:-1])])
    
    len_ood_data = np.sum([size_per_class[i] for i in target_classes[-1]])

    target_per_class = np.concatenate((target_per_class, np.array([len_ood_data], dtype=np.int32)))

    total_per_class = source_per_class + target_per_class
    
    log.debug(f"Needed <{source_per_class}> samples for source")
    log.debug(f"Needed <{target_per_class}> samples for target")
    
    idx, target_transform = get_idx(targets, target_classes, total_per_class)
    
    source_idx = [idx[c][:source_per_class[c]] for c in range(len(source_classes))]
    target_idx = [idx[c][source_per_class[c]:] for c in range(len(target_classes))]
    # import pdb; pdb.set_trace()
    return source_idx, target_idx, target_transform

def remap_idx(idx): 
    default_func = lambda: -1 

    def_map = defaultdict(default_func)
    sorted_idx = np.sort(idx)

    for i in range(len(sorted_idx)):
        def_map[sorted_idx[i]] = i
    
    return def_map


def get_splits(data_dir, dataset, source_classes, source_marginal, source_size,\
    target_classes, target_marginal, target_size, train = False, transform: Optional[Callable] = None): 

    data = get_data(data_dir, dataset, train=train, transform=transform)

    source_idx, target_idx, target_transform = split_indicies(data.targets, source_classes, target_classes,\
        source_marginal, target_marginal, source_size, target_size)
    
    data.target_transform = target_transform

    data.transform_idx = remap_idx(np.concatenate(target_idx).ravel())

    source_per_class = []
    for i in range(len(source_idx)):
        source_per_class.append(Subset(data, source_idx[i]))

    log.debug("Creating labeled and unlabeled splits}")
    source_data = Subset(data, np.concatenate(source_idx).ravel())
    target_data = Subset(data, np.concatenate(target_idx).ravel())

    return source_per_class, source_data, target_data


def get_splits_from_data(data, source_classes, source_marginal,\
    target_classes, target_marginal, dataset='', train = False, transform: Optional[Callable] = None):
    
    size_per_class = get_size_per_class(data)

    log.debug(f"Size per class: {size_per_class}")
    
    if isinstance(data, Subset):
        targets = np.array(data.dataset.targets)[data.indices] 
        source_idx, target_idx, target_transform = split_indicies_with_size(targets, source_classes, target_classes,\
            source_marginal, target_marginal, size_per_class)
        source_idx = [[data.indices[j]] for i in source_idx for j in i]
        target_idx = [[data.indices[j]] for i in target_idx for j in i] 
    else:
        source_idx, target_idx, target_transform = split_indicies_with_size(data.targets, source_classes, target_classes,\
            source_marginal, target_marginal, size_per_class)
    
    source_idx = np.concatenate(source_idx).ravel()
    target_idx = np.concatenate(target_idx).ravel()

    if isinstance(data, Subset):
        data.dataset.transform_idx = remap_idx(target_idx)
    else:
        data.transform_idx = remap_idx(target_idx)

    log.debug("Creating labeled and unlabeled splits}")

    SubsetwithTransform = dataset_transform_labels(Subset)
    
    if isinstance(data, Subset):
        source_data = SubsetwithTransform(data.dataset, source_idx) if dataset not in ['imagenet', 'sun397'] else Subset(data.dataset, source_idx)
        target_data = SubsetwithTransform(data.dataset, target_idx) if dataset not in ['imagenet', 'sun397'] else Subset(data.dataset, target_idx)
        
    else:
        source_data = SubsetwithTransform(data, source_idx) if dataset not in ['imagenet', 'sun397'] else Subset(data, source_idx)
        target_data = SubsetwithTransform(data, target_idx) if dataset not in ['imagenet', 'sun397'] else Subset(data, target_idx)
    source_data.target_transform = target_transform  
    target_data.target_transform = target_transform
    # import pdb; pdb.set_trace()
    return source_data, target_data, source_idx, target_idx, target_transform



def get_preprocessing(dset: str, use_aug: bool = False, train: bool = False, mean=None, std=None, arch=None):

    log.info(f"Using {dset} dataset with augmentation {use_aug} and training {train}")
    if dset.lower().startswith("cifar10"):
        mean = (0.4914, 0.4822, 0.4465) if mean is None else mean
        std = (0.2023, 0.1994, 0.2010) if std is None else std

    elif dset.lower().startswith("imagenet"):
        mean = (0.485, 0.456, 0.406) if mean is None else mean
        std = (0.229, 0.224, 0.225) if std is None else std
        # mean = (0.48145466, 0.4578275, 0.40821073) if mean is None else mean
        # std = (0.26862954, 0.26130258, 0.27577711) if std is None else std

    elif dset.lower() == 'cifar100':
        mean = (0.5074, 0.4867, 0.4411) if mean is None else mean
        std = (0.2011, 0.1987, 0.2025) if std is None else std

    elif dset.lower().startswith("cinic10"): 
        mean = (0.47889522, 0.47227842, 0.43047404) if mean is None else mean
        std = (0.24205776, 0.23828046, 0.25874835) if std is None else std
    
    elif dset.lower().startswith("mnist"): 
        mean = (0.1307,) if mean is None else mean
        std =  (0.3081,) if std is None else std
    elif dset.lower().startswith("tabula"):
        return None
    elif dset.lower().startswith("dermnet") \
        or dset.lower().startswith("breakhis")\
        or dset.lower().startswith("utkface")\
        or dset.lower().startswith("entity30"):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225] if std is None else std
    else:
        mean = (0.5, 0.5, 0.5) if mean is None else mean
        std = (0.5, 0.5, 0.5) if std is None else std

    if  dset.lower().startswith("cifar"):
        if use_aug and train : 
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else: 
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )
    elif dset.lower().startswith("imagenet"):
        if use_aug and train:
            if arch=='ResNet18':
                transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            else:
                transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
        else:
            if arch=='ResNet18':
                transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            else:
                transform = transforms.Compose(
                    [transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean, std)]
                )

    elif dset.lower().startswith("dermnet"):
        if use_aug and train:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    elif dset.lower().startswith("breakhis") or dset.lower().startswith("utkface"):
        if use_aug and train:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    elif dset.lower().startswith("entity30"):
        if use_aug and train:
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
                
    elif dset.lower().startswith("rxrx1"):
        if use_aug and train:
            return initialize_rxrx1_transform(is_training=True)
        else:
            return initialize_rxrx1_transform(is_training=False)
    elif dset.lower().startswith("novophen"):
        return None
    else: 
        transform = None

    return transform

def initialize_rxrx1_transform(is_training):

    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform
