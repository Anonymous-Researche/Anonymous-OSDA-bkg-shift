from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder, ImageNet, SUN397
from amazon_reviews_utils import *
from newsgroups_utils import *
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from torchvision.transforms._presets import ImageClassification 
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from tensordict import TensorDict
from sklearn.datasets import fetch_20newsgroups
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import clip

class SUN397Dataset(SUN397):
    def __init__(self, root, transform=None):
        super(SUN397Dataset, self).__init__(root, transform=transform)
    
    def __getitem__(self,index):
        img, target = super(SUN397Dataset, self).__getitem__(index)
        return img, target, index

class ImageNetDataset(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""
    def __init__(self, root, transform=None):
        super(ImageNetDataset, self).__init__(root, transform=transform)
        self.class_to_super_idx = {} 
        self.target_to_supertarget = {}
        self.target_transform = None
        self.supertarget_transform = None
        self.supertargets = []
        
        for (idx,(path, target)) in enumerate(self.samples):
            primary_class = path.split('/')[-2]
            if primary_class[:4] not in self.class_to_super_idx.keys():
                self.class_to_super_idx[primary_class[:4]] = len(self.class_to_super_idx) 
            supertarget = self.class_to_super_idx[primary_class[:4]]
            self.supertargets.append(supertarget)
            self.samples[idx] = (path, supertarget, target)
            self.target_to_supertarget[target] = supertarget 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, subclass target) where target is class_index of the target class.
        """
        # sample, target = super(ImageNetDataset, self).__getitem__(index)
        path, supertarget, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.supertarget_transform is not None:
            supertarget = self.supertarget_transform(supertarget)
        return sample, supertarget, target, index

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, sentiments):
        self.data = data
        self.targets = torch.tensor(targets, dtype=torch.long).cuda(0)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        
        if sentiments is not None:
            self.sentiments = torch.tensor(sentiments, dtype=torch.long).cuda(0)
        else:
            self.sentiments = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(self.data[idx], None, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_token_type_ids=True)
        data = {'input_ids':torch.tensor(inputs['input_ids'], dtype=torch.long).cuda(0),
            'attention_mask':torch.tensor(inputs['attention_mask'], dtype=torch.long).cuda(0),
            'token_type_ids':torch.tensor(inputs['token_type_ids'], dtype=torch.long).cuda(0),
            'text': self.data[idx]
            } 
        if self.sentiments is not None:
            return data, self.targets[idx], self.sentiments[idx], torch.tensor(idx).cuda(0)
        else:
            return data, self.targets[idx], torch.tensor(idx).cuda(0)

class RobertaFeatureExtractor(nn.Module):
    def __init__(self):
        super(RobertaFeatureExtractor, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.roberta.eval()
        
    def forward(self, x):
        all_layer_features = self.roberta(input_ids=x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'])
        return all_layer_features

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    
    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        return (data[0], data[1], index)

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def main():
    args = sys.argv[1:]
    
    assert args[0]=="--dataset"
    dataset = args[1]
    assert args[2]=="--arch"
    arch = args[3]

    if dataset=="amazon_reviews":
        data_dir = "/export/r36a/data/schaud35/shiftpu/amazon_reviews_tp" 
        amazon_reviews_data, all_labels = [],[]
        for label, product in enumerate(os.listdir(data_dir)):
            for data in review_parse(os.path.join(data_dir,product)):
                amazon_reviews_data.append(data)
                all_labels.append(label)

        labels, texts, sentiments = [],[], []
        for idx, (review, label) in enumerate(zip(amazon_reviews_data, all_labels)):
            if review['overall']==3.0 or 'overall' not in review.keys() or 'reviewText' not in review.keys():
                continue
            labels.append(label)
            texts.append(review['reviewText'])
            sentiments.append(int(review['overall']>3.0))
        
        labels, texts, sentiments = np.array(labels),np.array(texts), np.array(sentiments)
        neg_idxs = np.where(sentiments==0)[0]
        pos_idxs = np.where(sentiments==1)[0]
        
        selected_pos_idxs, selected_neg_idxs = [], []
        for label in np.unique(labels):
            label_neg_idxs = np.intersect1d(neg_idxs, np.where(labels==label)[0])
            label_pos_idxs = np.intersect1d(pos_idxs, np.where(labels==label)[0])
            np.random.shuffle(label_pos_idxs)
            label_pos_idxs = label_pos_idxs[:len(label_neg_idxs)]
            selected_pos_idxs.extend(label_pos_idxs)
            selected_neg_idxs.extend(label_neg_idxs)
        
        if arch=='Roberta':
            texts = list(map(fix_apostrophes, texts))
            MAX_LEN = 512

            net = RobertaFeatureExtractor()
            net = net.cuda(0)
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output[0].detach()
                return hook
            net.roberta.encoder.register_forward_hook(get_activation('encoder'))
            net.eval()

            amazon_reviews_dataset = SimpleDataset(data=texts, targets=labels, sentiments=sentiments)
            amazon_reviews_subset = Subset(amazon_reviews_dataset, torch.cat((torch.tensor(selected_pos_idxs), torch.tensor(selected_neg_idxs)),dim=0))
            dataloader = DataLoader(dataset=amazon_reviews_subset, shuffle=False, batch_size=200)
            save_features, save_input_txt, save_targets, save_sents, save_idxs = torch.tensor([]).cuda(1), torch.tensor([]).cuda(1), torch.tensor([]).cuda(1), torch.tensor([]).cuda(1), torch.tensor([]).cuda(1)
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader)):
                    
                    x, y, sent, sample_idx = batch[0], batch[1], batch[2], batch[3]
                    out = net(x)
                    
                    encoded_features, y, sent, sample_idx = activation['encoder'][:,0].cuda(1), y.cuda(1), sent.cuda(1), sample_idx.cuda(1)
                    save_features = torch.cat((save_features, encoded_features),dim=0)
                    # save_input_txt = torch.cat((save_input_txt,x),dim=0)
                    save_targets = torch.cat((save_targets, y),dim=0)
                    save_sents = torch.cat((save_sents, sent),dim=0)
                    save_idxs = torch.cat((save_idxs, sample_idx),dim=0)

                    del x
                    del y
                    del sent
                    del sample_idx
                    del out
                    torch.cuda.empty_cache()
            
            final_data = TensorDict({'features':save_features, 
                'targets':save_targets,
                'sentiments':save_sents,
                'indices':save_idxs 
                },batch_size=[])
            
            torch.save(final_data, "/export/r36a/data/schaud35/shiftpu/amazon_reviews_roberta_features.pth")
    
    elif dataset=="newsgroups":
        newsgroups_data = fetch_20newsgroups(subset='all', shuffle=False, 
                                         categories=newsgroup20_categories,remove=('header','footer',))
        
        labels=newsgroups_data.target
        texts = newsgroups_data.data
        
        if arch=='Roberta':
            texts = list(map(fix_apostrophes, texts))
            net = RobertaFeatureExtractor()
            net = net.cuda(0)
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output[0].detach()
                return hook
            net.roberta.encoder.register_forward_hook(get_activation('encoder'))
            net.eval()

            newsgroups_dataset = SimpleDataset(data=texts, targets=labels, sentiments=None)
            dataloader = DataLoader(dataset=newsgroups_dataset, shuffle=False, batch_size=200)
            save_features, save_input_txt, save_targets, save_idxs = torch.tensor([]).cuda(1), torch.tensor([]).cuda(1), torch.tensor([]).cuda(1), torch.tensor([]).cuda(1)
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader)):
                    x, y, sample_idx = batch[0], batch[1], batch[2]
                    out = net(x)
                    # import pdb; pdb.set_trace()
                    encoded_features, y, sample_idx = activation['encoder'][:,0].cuda(1), y.cuda(1), sample_idx.cuda(1)
                    save_features = torch.cat((save_features, encoded_features),dim=0)
                    save_targets = torch.cat((save_targets, y),dim=0)
                    save_idxs = torch.cat((save_idxs, sample_idx),dim=0)
                    del x
                    del y
                    del sample_idx
                    del out
                    torch.cuda.empty_cache()
            final_data = TensorDict({'features':save_features, 
                'targets':save_targets,
                'indices':save_idxs 
                },batch_size=[])
            
            torch.save(final_data, "/export/r36a/data/schaud35/shiftpu/newsgroups_roberta_features.pth")
        
    elif dataset=="cifar100":
        data_dir = "/export/r36a/data/schaud35/shiftpu/" 
        mean = (0.5074, 0.4867, 0.4411)
        std = (0.2011, 0.1987, 0.2025)

        CIFAR100withIndices = dataset_with_indices(CIFAR100)

        transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )
        
        
        if arch=='ResNet18':
            weights = ResNet18_Weights.DEFAULT
            net = resnet18(weights)
            # preprocess = weights.transforms()
            preprocess = ImageClassification(crop_size=224, resize_size=256, mean=(0.5074, 0.4867, 0.4411), std=(0.2011, 0.1987, 0.2025), interpolation=InterpolationMode.BILINEAR)
            net.cuda(0)
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            net.avgpool.register_forward_hook(get_activation('avgpool'))
            net.eval()

        train_data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=True, transform=preprocess, download=True)    
        val_data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=False, transform=preprocess, download=True)
    
        train_dataloader = DataLoader(train_data, batch_size=200, shuffle=False, num_workers=8)
        val_dataloader = DataLoader(val_data, batch_size=200, shuffle=False, num_workers=8)
        save_features, save_input_imgs, save_targets, save_idxs = torch.tensor([]).cuda(0), torch.tensor([]).cuda(0), torch.tensor([]).cuda(0), torch.tensor([]).cuda(0)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                x, y, sample_idx = batch[0].cuda(0), batch[1].cuda(0), batch[2].cuda(0)
                out = net(x)
                features, y, sample_idx = activation['avgpool'], y, sample_idx
                save_features = torch.cat((save_features, features.view(y.shape[0],-1)),dim=0)
                save_targets = torch.cat((save_targets, y),dim=0)
                save_idxs = torch.cat((save_idxs, sample_idx),dim=0)
                del x
                del y
                del sample_idx
                del out
                torch.cuda.empty_cache()

            for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                x, y, sample_idx = batch[0].cuda(0), batch[1].cuda(0), batch[2].cuda(0)
                out = net(x)
                features, y, sample_idx = activation['avgpool'].cuda(0), y.cuda(0), sample_idx.cuda(0)
                save_features = torch.cat((save_features, features.view(y.shape[0],-1)),dim=0)
                save_targets = torch.cat((save_targets, y),dim=0)
                save_idxs = torch.cat((save_idxs, sample_idx),dim=0)
                del x
                del y
                del sample_idx
                del out
                torch.cuda.empty_cache()
            
        final_data = TensorDict({'features':save_features, 
                'targets':save_targets,
                'indices':save_idxs 
                },batch_size=[])
            
        torch.save(final_data, "/export/r36a/data/schaud35/shiftpu/cifar100_ResNet18_features_imagenet_preprocessing.pth")
        import pdb; pdb.set_trace()
    
    elif dataset in ["imagenet"]:
        data_dir = "/export/r36a/data/schaud35/shiftpu/" 
        mean = (0.485, 0.456, 0.406) 
        std = (0.229, 0.224, 0.225) 
        transform = transforms.Compose(
                [transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)]
            )
        if arch.startswith('CLIP'):
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
            net, transform = clip.load(arch.split('_')[-1], device='cuda')
            # preprocess = weights.transforms()
            net.cuda(0)
            net.eval()
            activation = {}
        elif arch=="ResNet50":
            net = resnet50(ResNet50_Weights.IMAGENET1K_V1)
            net.cuda(0)
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            net.avgpool.register_forward_hook(get_activation('features'))
            net.eval()
        elif arch=="ViTL16_SWAG":
            net = torch.hub.load("facebookresearch/swag", model="vit_l16")
            net.cuda(0)
        elif arch=="ViTB16_SWAG":
            net = torch.hub.load("facebookresearch/swag", model="vit_b16")
            net.cuda(0)
        
        
        train_data = ImageNetDataset(root = f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/train", transform=transform)
        val_data = ImageNetDataset(root = f"{data_dir}/imagenet/ILSVRC/Data/CLS-LOC/val", transform=transform)

        train_dataloader = DataLoader(train_data, batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
        val_dataloader = DataLoader(val_data, batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
        save_features, save_input_imgs, save_targets, save_super_targets, save_idxs = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                x, y, y_sub, sample_idx = batch[0].cuda(0), batch[1].cuda(0), batch[2].cuda(0), batch[3].cuda(0)
                if arch.startswith("CLIP"):
                    features = net.encode_image(x)
                elif arch in ['ViTL16_SWAG', 'ViTB16_SWAG']:
                    features = net(x)
                else:
                    _ = net(x)
                    features = activation['features']
                save_features = torch.cat((save_features, features.view(y.shape[0],-1).cpu().detach()),dim=0)
                save_targets = torch.cat((save_targets, y_sub.cpu().detach()),dim=0)
                save_super_targets = torch.cat((save_super_targets, y.cpu().detach()),dim=0)
                save_idxs = torch.cat((save_idxs, sample_idx.cpu().detach()),dim=0)
                del x
                del y
                del y_sub
                del sample_idx
                del features
                torch.cuda.empty_cache()
            
            final_data = TensorDict({'features':save_features,  
                'supertargets':save_super_targets, 
                'targets':save_targets, 
                'indices':save_idxs, 
                },batch_size=[])
            torch.save(final_data, "/export/r36a/data/schaud35/shiftpu/train_"+dataset+"_"+arch+"_features_imagenet_pretraining.pth")
            del final_data
            torch.cuda.empty_cache()
            save_features, save_input_imgs, save_targets, save_super_targets, save_idxs = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

            for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                x, y, y_sub, sample_idx = batch[0].cuda(0), batch[1].cuda(0), batch[2].cuda(0), batch[3].cuda(0)
                if arch.startswith("CLIP"):
                    features = net.encode_image(x)  
                elif arch in ['ViTL16_SWAG', 'ViTB16_SWAG']:
                    features = net(x)
                else:
                    _ = net(x)
                    features = activation['features']
                save_features = torch.cat((save_features, features.view(y.shape[0],-1).cpu().detach()),dim=0)
                save_targets = torch.cat((save_targets, y_sub.cpu().detach()),dim=0)
                save_super_targets = torch.cat((save_super_targets, y.cpu().detach()),dim=0)
                save_idxs = torch.cat((save_idxs, sample_idx.cpu().detach()),dim=0)
                del x
                del y
                del y_sub
                del sample_idx
                del features
                torch.cuda.empty_cache()
            
            final_data = TensorDict({'features':save_features, 
                'supertargets':save_super_targets,
                'targets':save_targets,
                'indices':save_idxs,
                },batch_size=[])
            
            torch.save(final_data, "/export/r36a/data/schaud35/shiftpu/val_"+dataset+"_"+arch+"_features_imagenet_pretraining.pth")
        
    elif dataset in ["sun397"]:
        data_dir = "/export/r36a/data/schaud35/shiftpu/" 
        mean = (0.485, 0.456, 0.406) 
        std = (0.229, 0.224, 0.225) 
        transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)]
            )
        
        if arch=="ResNet50":
            net = resnet50(ResNet50_Weights.IMAGENET1K_V1)
            net.cuda(0)
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            net.avgpool.register_forward_hook(get_activation('features'))
            net.eval()
        elif arch.startswith('CLIP'):
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
            net, transform = clip.load(arch.split('_')[-1], device='cuda')
            # preprocess = weights.transforms()
            net.cuda(0)
            net.eval()
            activation = {}

        train_data = SUN397Dataset(root = f"{data_dir}/sun397/", transform=transform)
        # val_data = SUN397Dataset(root = f"{data_dir}/sun397/", transform=transform)
        train_dataloader = DataLoader(train_data, batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
        # val_dataloader = DataLoader(val_data, batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
        save_features, save_input_imgs, save_targets, save_super_targets, save_idxs = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                x, y, sample_idx = batch[0].cuda(0), batch[1].cuda(0), batch[2].cuda(0)
                if arch.startswith("CLIP"):
                    features = net.encode_image(x)
                elif arch in ['ViTL16_SWAG', 'ViTB16_SWAG']:
                    features = net(x)
                else:
                    _ = net(x)
                    features = activation['features']
                save_features = torch.cat((save_features, features.view(y.shape[0],-1).cpu().detach()),dim=0)
                save_targets = torch.cat((save_targets, y.cpu().detach()),dim=0)
                save_idxs = torch.cat((save_idxs, sample_idx.cpu().detach()),dim=0)
                del x
                del y
                del sample_idx
                del features
                torch.cuda.empty_cache()
            
            final_data = TensorDict({'features':save_features,  
                'targets':save_targets, 
                'indices':save_idxs, 
                },batch_size=[])
            torch.save(final_data, "/export/r36a/data/schaud35/shiftpu/train_"+dataset+"_"+arch+"_features_pretrained.pth")
            del final_data
            torch.cuda.empty_cache()
            save_features, save_input_imgs, save_targets, save_super_targets, save_idxs = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

            # for batch_idx, batch in enumerate(tqdm(val_dataloader)):
            #     x, y, sample_idx = batch[0].cuda(0), batch[1].cuda(0), batch[2].cuda(0)
            #     if arch.startswith("CLIP"):
            #         features = net.encode_image(x)  
            #     elif arch in ['ViTL16_SWAG', 'ViTB16_SWAG']:
            #         features = net(x)
            #     else:
            #         _ = net(x)
            #         features = activation['features']
            #     save_features = torch.cat((save_features, features.view(y.shape[0],-1).cpu().detach()),dim=0)
            #     save_targets = torch.cat((save_targets, y.cpu().detach()),dim=0)
            #     save_idxs = torch.cat((save_idxs, sample_idx.cpu().detach()),dim=0)
            #     del x
            #     del y
            #     del sample_idx
            #     del features
            #     torch.cuda.empty_cache()
            
            # final_data = TensorDict({'features':save_features, 
            #     'supertargets':save_super_targets,
            #     'targets':save_targets,
            #     'indices':save_idxs,
            #     },batch_size=[])
            
            # torch.save(final_data, "/export/r36a/data/schaud35/shiftpu/val_"+dataset+"_"+arch+"_features_imagenet_pretraining.pth")
        # import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
