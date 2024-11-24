import torch.nn as nn
import torch.nn.functional as F
import torchvision
import clip

import torch.optim.lr_scheduler as lr_sched
from models import * 
import logging 
import torch 

from collections import OrderedDict
from lion_pytorch import Lion
from transformers import RobertaModel, RobertaTokenizer

from src.datasets.newsgroups_utils import *
from src.datasets.amazon_reviews_utils import *

log = logging.getLogger("app")

all_classifiers = {
    "Resnet18": ResNet18,
    "Resnet50": ResNet50,
    "Resnet34": ResNet34,
    "Resnet18MultiScale": ResNet18MultiScale,
    "Densenet121": DenseNet121,
    'ViTB16_SWAG': None,
    'CLIP_RN50': None,
    'CLIP_ViT-L14': None,
}

# DIR_PATH = "path/to/models"

def full_block(in_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )


class Model_20(nn.Module):

    def __init__(self, vocab_size, dim, embeddings, num_classes, features=False):
        super(Model_20, self).__init__()
        self.vocab_size = vocab_size 
        self.dim = dim
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.convnet = nn.Sequential(OrderedDict([
            #('embed1', nn.Embedding(self.vocab_size, self.dim)),
            ('c1', nn.Conv1d(100, 128, 5)),
            ('bn1', nn.BatchNorm1d(128)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(5)),
            ('c2', nn.Conv1d(128, 128, 5)),
            ('bn2', nn.BatchNorm1d(128)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(5)),
            ('c3', nn.Conv1d(128, 128, 5)),
            ('bn3', nn.BatchNorm1d(128)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool1d(35)),
        ]))
    
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embeddings))
        #copy_((embeddings))
        self.embedding.weight.requires_grad = True
    
        if not features:
            self.fc = nn.Sequential(OrderedDict([
                ('dropout1', nn.Dropout(0.2)),
                ('f4', nn.Linear(128, 128)),
                ('relu4', nn.ReLU()),
                ('dropout2', nn.Dropout(0.2)),
                ('f5', nn.Linear(128, num_classes)),
                # ('sig5', nn.LogSoftmax(dim=-1))
            ]))
        else: 
            self.fc = nn.Sequential(OrderedDict([
                ('f4', nn.Linear(128, 128)),
                ('relu4', nn.ReLU()),
            ]))

    def forward(self, img):
        # import pdb; pdb.set_trace()
        output = self.embedding(img)
        output.transpose_(1,2)
        # print("embedding layer norm: ", output.norm())
        output = self.convnet(output)
        # print("convnet layer norm: ", output.norm())
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        # print("fc (final) layer norm: ", output.norm())
        # print("\n")
        
        return output

class FCNet(nn.Module):
    def __init__(self, x_dim, num_classes, hid_dim=64, z_dim=64, dropout=0.2, features=False):
        super(FCNet, self).__init__()

        if not features:
            self.encoder = nn.Sequential(
                full_block(x_dim, hid_dim, dropout),
                full_block(hid_dim, z_dim, dropout),
                nn.Linear(z_dim, num_classes)
            )
        else: 
            self.encoder = nn.Sequential(
                full_block(x_dim, hid_dim, dropout),
                full_block(hid_dim, z_dim, dropout),
            )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
class FCNet_SAREM(nn.Module):
    def __init__(self, x_dim, num_classes, attr, hid_dim=64, z_dim=64, dropout=0.2, features=False):
        super(FCNet_SAREM, self).__init__()

        if not features:
            self.encoder = nn.Sequential(
                full_block(x_dim, hid_dim, dropout),
                full_block(hid_dim, z_dim, dropout),
                nn.Linear(z_dim, num_classes)
            )
        else: 
            self.encoder = nn.Sequential(
                full_block(x_dim, hid_dim, dropout),
                full_block(hid_dim, z_dim, dropout),
            )
        self.attr = attr

    def forward(self, x):
        x = x[:,self.attr]
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class RobertaClassifier(nn.Module):
    def __init__(self, num_classes, max_length=512, dropout=0.2, features=False):
        super(RobertaClassifier, self).__init__()
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        for name, param in self.roberta.named_parameters():
            if name.startswith("pooler.dense"):
                param.requires_grad = True
            else:
                param.requires_grad = False
            

        self.classifier = nn.Linear(768,num_classes)
    
    def forward(self, x):
        # inputs = self.tokenizer.encode_plus(x, None, add_special_tokens=True, truncation=True, max_length=self.max_length, padding='max_length', return_token_type_ids=True)
        last_layer_features = self.roberta(input_ids=x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'])
        pooled_features = last_layer_features[1]
        out = self.classifier(pooled_features)
        return out

class FeatureClassifier(nn.Module):
    def __init__(self, feature_extractor=None, freeze_feature_extractor=True, in_features=768, hidden_size=768, num_classes=4, dropout=0.2, preprocess=None, clip=False, data_type=torch.float32):
        super(FeatureClassifier, self).__init__()
        torch.set_default_dtype(data_type)
        if feature_extractor:
            self.feature_extractor = feature_extractor
            self.activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    self.activation[name] = output.detach()
                return hook
            if not clip:
                feature_extractor.avgpool.register_forward_hook(get_activation('features'))
                d_features = getattr(feature_extractor, "fc").in_features
                self.preclassifier = nn.Linear(d_features, hidden_size)
                # self.classifier = nn.Linear(d_features, num_classes)  
            else:
                self.preclassifier = nn.Linear(in_features, hidden_size)  
            
            if freeze_feature_extractor:
                for param in feature_extractor.parameters():
                    param.requires_grad = False
        else:
            self.feature_extractor = None
            self.preclassifier = nn.Linear(in_features, hidden_size)
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.clip=clip
        if preprocess is not None:
            self.preprocess = preprocess
    def forward(self, x):
        if self.feature_extractor and not self.clip:
            _ = self.feature_extractor(x)
            features = self.activation['features'].view(x.shape[0],-1)
        elif self.clip:
            features = self.feature_extractor.encode_image(x)
        else:
            features = x
        
        out = F.relu(self.preclassifier(self.dropout1(features)))
        out = self.classifier(self.dropout2(out))
        return out 


def get_model(arch, data_dir, dataset, num_classes, pretrained, learning_rate, weight_decay, features = False,
              pretrained_model_dir= None, pretrained_model_path=None, mode="classification", device='cuda', optimizer='sgd'):
    
    if dataset.lower().startswith("cifar") and arch in all_classifiers: 
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")
        net = all_classifiers[arch](num_classes=num_classes, features=features)

        if pretrained: 
            log.debug(f"Loading " + arch + " pretrained model")
            if pretrained_model_path:
                checkpoint = torch.load(f"{pretrained_model_path}", map_location='cpu')
                state_dict = {k: v for k, v in list(checkpoint.items())[:]}
            elif pretrained_model_dir:
                checkpoint = torch.load(f"{pretrained_model_dir}/simclr/simclr_cifar-20.pth.tar", map_location='cpu')
                state_dict = {k[9:]: v for k, v in checkpoint.items()}
            net.load_state_dict(state_dict, strict=False)
            # for n,p in net.named_parameters():
            #     if n in state_dict.keys():
            #         p.requires_grad = False
        
        parameters_net = net.parameters()

        if optimizer=='sgd':
            optimizer_net = torch.optim.SGD(
                parameters_net,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        elif optimizer=='adamw':
            optimizer_net = torch.optim.AdamW(
                parameters_net,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer=='lion':
            optimizer_net = Lion(net.parameters(),
                            lr=learning_rate, 
                            weight_decay=weight_decay, 
                            use_triton=True)

        return net, optimizer_net
    
    elif dataset.lower().startswith("imagenet") and arch in all_classifiers: 
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")
        
        # feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # feature_extractor, preprocess = clip.load("RN50", device=device)
        if arch=="ViTB16_SWAG":
            net = FeatureClassifier(feature_extractor=None, freeze_feature_extractor=True, in_features=768, hidden_size=768, num_classes=num_classes, preprocess=None, clip=False)
        elif arch=="Resnet18":
            net = FeatureClassifier(feature_extractor=torchvision.models.resnet18(weights=None), freeze_feature_extractor=False, num_classes=num_classes, preprocess=None, clip=False)
        else:
            net = FeatureClassifier(feature_extractor=None, freeze_feature_extractor=True, in_features=1024, hidden_size=768, num_classes=num_classes, preprocess=None, clip=False)

        if pretrained: 
            log.debug(f"Loading " + arch + " pretrained model")
            if pretrained_model_path:
                checkpoint = torch.load(f"{pretrained_model_path}", map_location='cpu')
                state_dict = {k: v for k, v in list(checkpoint.items())[:-2]}
            elif pretrained_model_dir:
                checkpoint = torch.load(f"{pretrained_model_dir}/simclr/simclr_cifar-20.pth.tar", map_location='cpu')
                state_dict = {k[9:]: v for k, v in checkpoint.items()}
            net.load_state_dict(state_dict, strict=False)
            # for n,p in net.named_parameters():
            #     if n in state_dict.keys():
            #         p.requires_grad = False
            

        # net = torch.nn.parallel.DistributedDataParallel(net)
        
        net = torch.nn.DataParallel(net)
        parameters_net = net.parameters()

        if optimizer=='sgd':
            optimizer_net = torch.optim.SGD(
                parameters_net,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        elif optimizer=='adamw':
            optimizer_net = torch.optim.AdamW(
                net.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        # optimizer_net_2 = torch.optim.AdamW(
        #     net.parameters(),
        #     lr=learning_rate,
        #     weight_decay=weight_decay
        # )

        # optimizer_net = torch.optim.AdamW(
        #     parameters_net,
        #     lr=learning_rate,
        #     weight_decay=weight_decay
        # )
        elif optimizer=='lion':
            optimizer_net = Lion(net.parameters(),
                            lr=learning_rate, 
                            weight_decay=weight_decay, 
                            use_triton=True)
        
        return net, optimizer_net
    
    elif dataset.lower().startswith("sun397") and arch in all_classifiers: 
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")
        
        if arch=='Resnet50':
            net = FeatureClassifier(feature_extractor=None, freeze_feature_extractor=True, in_features=2048, hidden_size=1024, num_classes=num_classes, preprocess=None, clip=False)
        elif arch == 'CLIP_RN50':
            net = FeatureClassifier(feature_extractor=None, freeze_feature_extractor=True, in_features=1024, hidden_size=768, num_classes=num_classes, preprocess=None, clip=False)
        elif arch =='CLIP_ViT-L14':
            net = FeatureClassifier(feature_extractor=None, freeze_feature_extractor=True, in_features=768, hidden_size=512, num_classes=num_classes, preprocess=None, clip=False)
        elif arch in ['CLIP_ViT-B16', 'CLIP_ViT-B32']:
            net = FeatureClassifier(feature_extractor=None, freeze_feature_extractor=True, in_features=512, hidden_size=256, num_classes=num_classes, preprocess=None, clip=False)
        net = torch.nn.DataParallel(net)
        parameters_net = net.parameters()

        if optimizer=='sgd':
            optimizer_net = torch.optim.SGD(
                parameters_net,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        elif optimizer=='adamw':
            optimizer_net = torch.optim.AdamW(
                net.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer=='lion':
            optimizer_net = Lion(net.parameters(),
                            lr=learning_rate, 
                            weight_decay=weight_decay, 
                            use_triton=True)
        
        return net, optimizer_net

    elif (dataset.lower().startswith("tabula") and arch=="FCN"): 

        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        net = FCNet(2866, num_classes)

        optimizer = torch.optim.Adam(net.parameters())

        return net, optimizer
    
    elif (dataset.lower().startswith("20ng") and arch=="FCN"):
        if mode=='classification':
            net = FCNet_SAREM(112, num_classes, [k for k in range(112)])
            optimizer = torch.optim.Adam(net.parameters())
        elif mode=="propensity_estimation":
            net = FCNet_SAREM(4, num_classes, [k for k in range(111,115)])
            optimizer = torch.optim.Adam(net.parameters())
        return net, optimizer

    elif arch == "FCN" and dataset =="MNIST": 
        net = nn.Sequential(nn.Flatten(),
                nn.Linear(28*28, 5000, bias=True),
                nn.ReLU(),
                nn.Linear(5000, 5000, bias=True),
                nn.ReLU(),
                nn.Linear(5000, 50, bias=True),
                nn.ReLU(),
                nn.Linear(50, num_classes, bias=True)
            )
        return net 

    elif arch == "FCN" and  dataset.lower().startswith("cifar"): 
        net = nn.Sequential(nn.Flatten(),
                nn.Linear(32*32*3, 5000, bias=True),
                nn.ReLU(),
                nn.Linear(5000, 5000, bias=True),
                nn.ReLU(),
                nn.Linear(5000, 50, bias=True),
                nn.ReLU(),
                nn.Linear(50, num_classes, bias=True)
            )
        return net 

    
    elif dataset.lower().startswith("dermnet") and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        net = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'
        
        d_features = getattr(net, last_layer_name).in_features
        last_layer = nn.Linear(d_features, num_classes)
        net.d_out = num_classes
        setattr(net, last_layer_name, last_layer)

        optimizer = torch.optim.Adam(
            net.parameters(), 
            lr=learning_rate
        )

        return net, optimizer

    elif (dataset.lower().startswith("breakhis")  or dataset.lower().startswith("utkface")) and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        net = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'

        d_features = getattr(net, last_layer_name).in_features
        last_layer = nn.Linear(d_features, num_classes)
        net.d_out = num_classes
        setattr(net, last_layer_name, last_layer)

        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=learning_rate
        )

        return net, optimizer

    elif dataset.lower().startswith("entity30") and arch=="Resnet18":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        net = torchvision.models.resnet18(pretrained=False)

        if pretrained: 
            log.debug(f"Loading SIMCLR pretrained model")
            checkpoint = torch.load(f"{pretrained_model_dir}/simclr/pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar", map_location='cpu')
            state_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items()}
            net.load_state_dict(state_dict, strict=False)

        last_layer_name = 'fc'

        d_features = getattr(net, last_layer_name).in_features
        last_layer = nn.Linear(d_features, num_classes)
        net.d_out = num_classes
        setattr(net, last_layer_name, last_layer)

        if not pretrained:
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=learning_rate
            )

        return net, optimizer

    elif dataset.lower().startswith("newsgroups"):
        # arch = "Model_20"

        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        if arch=="Model_20":
            _, _, word_index = get_newsgroups(arch)
            embedding_matrix = glove_embeddings(f"/cis/home/schaud35/shiftpu/pretrained_models/glove_embeddings/glove.6B.100d.txt", word_index)
            EMBEDDING_DIM = 100
            net = Model_20(embedding_matrix.shape[0], EMBEDDING_DIM, embedding_matrix, num_classes)

            if pretrained: 
                log.debug(f"Loading " + arch + " pretrained model")
                if pretrained_model_path:
                    checkpoint = torch.load(f"{pretrained_model_path}", map_location='cpu')
                    state_dict = {k: v for k, v in list(checkpoint.items())[:-2]}
                elif pretrained_model_dir:
                    checkpoint = torch.load(f"{pretrained_model_dir}/simclr/simclr_cifar-20.pth.tar", map_location='cpu')
                    state_dict = {k[9:]: v for k, v in checkpoint.items()}
                net.load_state_dict(state_dict, strict=False)

        elif arch=="Roberta":
            MAX_LEN = 512
            net = RobertaClassifier(num_classes, max_length=MAX_LEN)
        elif arch=="Roberta_linear_classifier":
            net = FeatureClassifier(in_features=768, hidden_size=768, num_classes=num_classes)
        else:
            raise Exception("Not a valid architecture for newsgroups20.")

        

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)

        return net, optimizer

    elif dataset.lower().startswith("amazon_reviews"):
        if arch=="Model_20":
            arch= "Model_20"

            log.info(f"Using {arch} for {dataset} with {num_classes} classes")

            _, _, _, word_index = get_amazon_reviews(f"{data_dir}/amazon_reviews_tp", 4, arch)
            embedding_matrix = glove_embeddings(f"{pretrained_model_dir}/glove_embeddings/glove.6B.100d.txt", word_index)

            EMBEDDING_DIM = 100
            net = Model_20(embedding_matrix.shape[0], EMBEDDING_DIM, embedding_matrix, num_classes)
            if pretrained: 
                log.debug(f"Loading " + arch + " pretrained model")
                if pretrained_model_path:
                    checkpoint = torch.load(f"{pretrained_model_path}", map_location='cpu')
                    state_dict = {k: v for k, v in list(checkpoint.items())[:-2]}
                elif pretrained_model_dir:
                    checkpoint = torch.load(f"{pretrained_model_dir}/simclr/simclr_cifar-20.pth.tar", map_location='cpu')
                    state_dict = {k[9:]: v for k, v in checkpoint.items()}
                net.load_state_dict(state_dict, strict=False)
        elif arch=='Roberta':
            MAX_LEN = 512
            net = RobertaClassifier(num_classes, max_length=MAX_LEN)
        elif arch=="Roberta_linear_classifier":
            net = FeatureClassifier(in_features=768, hidden_size=768, num_classes=num_classes)
        else:
            raise Exception("Not a valid architecture for newsgroups20.")

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)

        return net, optimizer

    elif dataset.lower().startswith("rxrx1") and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        net = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'
        
        d_features = getattr(net, last_layer_name).in_features
        last_layer = nn.Linear(d_features, num_classes)
        net.d_out = num_classes
        setattr(net, last_layer_name, last_layer)

        optimizer = torch.optim.Adam(
            net.parameters(), 
            lr=learning_rate//10, 
            weight_decay=weight_decay
        )     

        return net, optimizer
    
    elif arch =="Densenet121":  
        net = torchvision.models.densenet121(pretrained=pretrained)
        last_layer_name = 'classifier'

    elif arch =="Resnet50": 
        net = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'

    else: 
        raise NotImplementedError("Net %s is not implemented" % arch)

    if arch in ('ResNet50', 'DenseNet121') :
        d_features = getattr(net, last_layer_name).in_features
        last_layer = nn.Linear(d_features, num_classes)
        net.d_out = num_classes
        setattr(net, last_layer_name, last_layer)

    return net	


def get_combined_model(arch, dataset, num_classes, pretrained, learning_rate, weight_decay, features = False, pretrained_model_dir=None): 

    if dataset.lower().startswith("cifar") and arch in all_classifiers: 
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")
        feature_extractor = all_classifiers[arch](num_classes=num_classes, features=features)

        d_features = getattr(feature_extractor, "linear").in_features

        linear_classifier = nn.Linear(d_features, num_classes)

        if pretrained: 
            log.debug(f"Loading SIMCLR pretrained model")
            checkpoint = torch.load(f"{pretrained_model_dir}/simclr/simclr_cifar-20.pth.tar", map_location='cpu')
            state_dict = {k[9:]: v for k, v in checkpoint.items()}
            feature_extractor.load_state_dict(state_dict, strict=False)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        parameters_classifier = classifier.parameters()

        optimizer_classifier = torch.optim.SGD(
            parameters_classifier,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

        parameters_domain_discriminator = domain_discriminator.parameters()

        optimizer_domain_discriminator = torch.optim.SGD(
            parameters_domain_discriminator,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

    elif dataset.lower().startswith("entity30") and arch=="Resnet18":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        feature_extractor = torchvision.models.resnet18(pretrained=False)

        if pretrained:
            log.debug(f"Loading SIMCLR pretrained model")
            checkpoint = torch.load(f"{pretrained_model_dir}/simclr/pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar", map_location='cpu')
            state_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items()}
            feature_extractor.load_state_dict(state_dict, strict=False)


        last_layer_name = 'fc'

        d_features = getattr(feature_extractor, last_layer_name).in_features
        last_layer = nn.Identity(d_features, d_features)
        feature_extractor.d_out = d_features

        setattr(feature_extractor, last_layer_name, last_layer)

        linear_classifier = nn.Linear(d_features, num_classes)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        parameters_classifier = classifier.parameters()

        if not pretrained: 
            optimizer_classifier = torch.optim.SGD(
                parameters_classifier,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            optimizer_classifier = torch.optim.Adam(
                parameters_classifier,
                lr=learning_rate)
        
        parameters_domain_discriminator = domain_discriminator.parameters()

        if not pretrained: 
            optimizer_domain_discriminator = torch.optim.SGD(
                parameters_domain_discriminator,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            optimizer_domain_discriminator = torch.optim.Adam(
                parameters_domain_discriminator,
                lr=learning_rate)

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

    elif dataset.lower().startswith("dermnet") and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        feature_extractor = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'
        
        d_features = getattr(feature_extractor, last_layer_name).in_features
        last_layer = nn.Identity(d_features, d_features)
        feature_extractor.d_out = d_features

        setattr(feature_extractor, last_layer_name, last_layer)

        linear_classifier = nn.Linear(d_features, num_classes)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        parameters_classifier = classifier.parameters()

        optimizer_classifier = torch.optim.SGD(
            parameters_classifier,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

        parameters_domain_discriminator = domain_discriminator.parameters()

        optimizer_domain_discriminator = torch.optim.SGD(
            parameters_domain_discriminator,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )     

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

    elif (dataset.lower().startswith("breakhis")  or dataset.lower().startswith("utkface")) and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        feature_extractor = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'

        d_features = getattr(feature_extractor, last_layer_name).in_features
        last_layer = nn.Identity(d_features, d_features)
        feature_extractor.d_out = d_features

        setattr(feature_extractor, last_layer_name, last_layer)

        linear_classifier = nn.Linear(d_features, num_classes)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        parameters_classifier = classifier.parameters()

        optimizer_classifier = torch.optim.SGD(
            parameters_classifier,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

        parameters_domain_discriminator = domain_discriminator.parameters()

        optimizer_domain_discriminator = torch.optim.SGD(
            parameters_domain_discriminator,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )     

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

    elif dataset.lower().startswith("newsgroups"):
        arch= "Model_20"

        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        _, _, word_index = get_newsgroups()
        embedding_matrix = glove_embeddings(f"{pretrained_model_dir}/glove_embeddings/glove.6B.100d.txt", word_index)

        EMBEDDING_DIM = 100

        feature_extractor = Model_20(embedding_matrix.shape[0], EMBEDDING_DIM, embedding_matrix, num_classes, features=features)

        d_features = 128

        linear_classifier = nn.Linear(d_features, num_classes)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        optimizer_classifier = torch.optim.Adam(\
            filter(lambda p: p.requires_grad, classifier.parameters()), \
            lr=learning_rate)

        optimizer_domain_discriminator = torch.optim.Adam(\
            filter(lambda p: p.requires_grad, domain_discriminator.parameters()), \
            lr=learning_rate)

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

    elif dataset.lower().startswith("tabula") and arch=="FCN": 

        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        feature_extractor = FCNet(2866, num_classes, features=features)
        d_features = 64

        linear_classifier = nn.Linear(d_features, num_classes)

        linear_domain_discriminator = nn.Linear(d_features, 2)

        classifier = nn.Sequential(feature_extractor, linear_classifier)

        domain_discriminator = nn.Sequential(feature_extractor, linear_domain_discriminator)

        optimizer_classifier = torch.optim.Adam(classifier.parameters())

        optimizer_domain_discriminator = torch.optim.Adam(domain_discriminator.parameters())

        return classifier, domain_discriminator, optimizer_classifier, optimizer_domain_discriminator

def update_optimizer(epoch, opt, data, lr): 

    if data.lower().startswith("cifar"): 
        if epoch>=70: 
            for g in opt.param_groups:
                g['lr'] = 0.1*lr			
        if epoch>=140: 
            for g in opt.param_groups:
                g['lr'] = 0.01*lr

    elif data.lower().startswith("entity30"): 
        if epoch>=100: 
            for g in opt.param_groups:
                g['lr'] = 0.1*lr			
        if epoch>=200: 
            for g in opt.param_groups:
                g['lr'] = 0.01*lr

    elif data.lower().startswith("breakhis") or data.lower().startswith("dermnet"):
        for g in opt.param_groups:
            g['lr'] = lr*((0.96)**(epoch))

    # elif data.lower().startswith("newsgroups"):
    #     for g in opt.param_groups:
    #         g['lr'] = lr*((0.96)**(epoch))

    elif data.lower().startswith("rxrx1"):
        if epoch <10: 
            for g in opt.param_groups:
                g['lr'] = (epoch+1)*lr / 10.0
        else: 
            for g in opt.param_groups:
                g['lr'] = max(0.0, 0.5*(1.0  + math.cos(math.pi *(epoch - 10.0/(80.0)))))*lr

    return opt
