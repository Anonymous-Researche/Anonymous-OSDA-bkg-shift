import os
import sys

# Get the absolute path of the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the root directory to sys.path
sys.path.append(root_dir)

from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder, ImageNet
# sort relative packaging to import contents from data_utils.py
from src.data_utils import *
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from torchvision.transforms._presets import ImageClassification 
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from tensordict import TensorDict
from sklearn.datasets import fetch_20newsgroups
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import clip

def knn_distances(train_data=None, val_data=None, ood_subclasses=[], k=100, verbose=False):
        train_feats = [train_data[i][0].cpu().detach().numpy() for i in range(len(train_data)) if train_data[i][2] not in ood_subclasses] if train_data else []
        val_feats = [val_data[i][0].cpu().detach().numpy() for i in range(len(val_data)) if val_data[i][2] not in ood_subclasses] if val_data else []
        train_feats.extend(val_feats)
        data = torch.tensor(np.array(train_feats), device='cuda:0' if torch.cuda.is_available() else 'cpu')
        ood_data = []
        
        for ood_subclass in ood_subclasses:
            ood_train_feats =  [train_data[i][0].cpu().detach().numpy() for i in range(len(train_data)) if train_data[i][2] == ood_subclass] if train_data else []
            ood_val_feats = [val_data[i][0].cpu().detach().numpy() for i in range(len(val_data)) if val_data[i][2] == ood_subclass] if val_data else []
            ood_train_feats.extend(ood_val_feats)
            ood_data.append(ood_train_feats)
        ood_data = torch.tensor(np.array(ood_data), device='cuda:0' if torch.cuda.is_available() else 'cpu')

        train_targets = [train_data[i][1] for i in range(len(train_data)) if train_data[i][2] not in ood_subclasses] if train_data else []
        val_targets = [val_data[i][1] for i in range(len(val_data)) if val_data[i][2] not in ood_subclasses] if val_data else []
        train_targets.extend(val_targets)
        targets = torch.tensor(train_targets)
        ood_targets = []
        for ood_subclass in ood_subclasses:
            ood_train_targets = [train_data[i][1] for i in range(len(train_data)) if train_data[i][2] == ood_subclass] if train_data else []
            ood_val_targets = [val_data[i][1] for i in range(len(val_data)) if val_data[i][2] == ood_subclass] if val_data else []
            ood_train_targets.extend(ood_val_targets)
            ood_targets.append(ood_train_targets)
        ood_targets = torch.tensor(ood_train_targets)

        train_subtargets = [train_data[i][2] for i in range(len(train_data)) if train_data[i][2] not in ood_subclasses] if train_data else []
        val_subtargets = [val_data[i][2] for i in range(len(val_data)) if val_data[i][2] not in ood_subclasses] if val_data else []
        train_subtargets.extend(val_subtargets)
        subtargets = torch.tensor(train_subtargets)
        ood_subtargets = []
        for ood_subclass in ood_subclasses:
            ood_train_subtargets = [train_data[i][2] for i in range(len(train_data)) if train_data[i][2] in ood_subclasses] if train_data else []
            ood_val_subtargets = [val_data[i][2] for i in range(len(val_data)) if val_data[i][2] in ood_subclasses] if val_data else []
            ood_train_subtargets.extend(ood_val_subtargets)
            ood_subtargets.append(ood_train_subtargets)
        ood_subtargets = torch.tensor(ood_train_subtargets)
        
        # Calculate pairwise distances between target and source samples
        inter_grp_distances = torch.cdist(ood_data, data)
        intra_grp_distances = torch.cdist(ood_data, ood_data)
        # Calculate cosine similarity between target and source samples
        # similarities = F.cosine_similarity(target_tensor.unsqueeze(1), source_tensor.unsqueeze(0), dim=2)
        knn_inter_grp_dist, knn_inter_grp_idxs = torch.topk(inter_grp_distances, k, dim=-1, largest=False)
        knn_intra_grp_dist, knn_intra_grp_idxs = torch.topk(intra_grp_distances, k+1, dim=-1, largest=False)
        knn_intra_grp_dist, knn_intra_grp_idxs = knn_intra_grp_dist[:,:,1:], knn_intra_grp_idxs[:,:,1:]
        mean_knn_inter_grp_dist = knn_inter_grp_dist.mean(dim=-1)
        mean_knn_intra_grp_dist = knn_intra_grp_dist.mean(dim=-1)
        knn_classes = [[[targets[idx].item() for idx in row] for row in batch] for batch in knn_inter_grp_idxs]
        knn_subclasses = [[[subtargets[idx].item() for idx in row] for row in batch] for batch in knn_inter_grp_idxs]
        if verbose:
            for batch in range(len(ood_subclasses)):
                print("OOD class:", ood_subclasses[batch])
                print("avg inter group distance: ", mean_knn_inter_grp_dist[batch].mean(dim=-1).item())
                print("avg intra group distance: ", mean_knn_intra_grp_dist[batch].mean(dim=-1).item())
                print("knn class frequency:", Counter([item for sublist in knn_classes[batch] for item in sublist]))
                print("knn subclass frequency:", Counter([item for sublist in knn_subclasses[batch] for item in sublist]))
        # import pdb; pdb.set_trace()
        return [mean_knn_inter_grp_dist.mean(dim=-1).cpu().detach(), mean_knn_intra_grp_dist.mean(dim=-1).cpu().detach()]

def get_dataset_from_classes(dataset, subclasses):
    indices = []
    for i in range(len(dataset)):
        if dataset[i][2] in subclasses:
            indices.append(i)
    return Subset(dataset, indices)
     
def main():

    data_dir = "/export/r36a/data/schaud35/shiftpu/" 
    ood_subclasses = ['n01689811', 'n01644373', 'n01644900', 'n02692877', 'n01692333'] # ['n01644373', 'n01644900', 'n02123597', 'n02676566', 'n02692877', 'n01616318']
    ks = [10] #[10, 100, 1000]
    train_data = ImageNetDataset(root=f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/train", features=f"{data_dir}/train_imagenet_ViTB16_SWAG_features_imagenet_pretraining.pth")
    val_data = ImageNetDataset(root=f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/val", features=f"{data_dir}/val_imagenet_ViTB16_SWAG_features_imagenet_pretraining.pth")   
    source_classes =  [['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041'], 
                                ['n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084'], 
                                ['n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264'],#, n01749939, n01751748, n01753488, n01755581, n01756291],
                                ['n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029'],#, n02104365, n02105056, n02105162, n02105251, n02105412, n02105505, n02105641, n02105855, n02106030, n02106166, n02106382, n02106550, n02106662, n02107142, n02107312, n02107574, n02107683, n02107908, n02108000, n02108089, n02108422, n02108551, n02108915, n02109047, n02109525, n02109961, n02110063, n02110185, n02110341, n02110627, n02110806, n02110958, n02111129, n02111277, n02111500, n02111889, n02112018, n02112137, n02112350, n02112706, n02113023, n02113186, n02113186, n02113624, n02113712, n02113799, n02113978]
                                ['n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853']]
    source_classes = [[train_data.class_to_idx[subclass] for subclass in superclass] for superclass in source_classes]
    superclass_mapping = {subclass: superclass for superclass, subclasses in enumerate(source_classes) for subclass in subclasses}
    source_subclasses = [subclass for superclass in source_classes for subclass in superclass]
    ood_subclasses = [train_data.class_to_idx[ood_subclass] for ood_subclass in ood_subclasses] # n03770679 n01689811 n01644373 n02123597
    # ood_subclasses = [i for i in range(600,610) if i not in source_subclasses]
    distances = []
    for ood_subcls in tqdm(ood_subclasses):
        # print("OOD class:", ood_subcls)
        superclass_mapping[ood_subcls] = len(source_classes)
    all_classes = source_subclasses + ood_subclasses
    train_subset = get_dataset_from_classes(train_data, all_classes)
    val_subset = get_dataset_from_classes(val_data, all_classes)
    train_subset.dataset.class_to_superclass, val_subset.dataset.class_to_superclass = superclass_mapping, superclass_mapping
    for k in ks:
        distances.append(knn_distances(train_subset, val_subset, ood_subclasses=ood_subclasses, k=k, verbose=True))
    train_subset.dataset.class_to_superclass, val_subset.dataset.class_to_superclass = {}, {}
    distances = np.array(distances)
    # pd.DataFrame(data=distances, index=ood_subclasses, columns=['knn_inter_grp_dist','knn_intra_grp_dist']).to_excel('/cis/home/schaud35/shiftpu/shiftpu/knn_distances_k_'+str(ks[0])+'_all_unknown_classes.xlsx')
    # import pdb; pdb.set_trace() 

if __name__ == "__main__":
    main()

