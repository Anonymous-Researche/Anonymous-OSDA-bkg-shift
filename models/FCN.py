"""Fully connected network in PyTorch."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def full_block(in_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )

class MLP(nn.Module):
    def __init__(self, x_dim, hid_dim=64, z_dim=64, dropout=0.2, num_classes=2):
        super(FCNet, self).__init__()
        self.encoder = nn.Sequential(
            full_block(x_dim, hid_dim, dropout),
            full_block(hid_dim, z_dim, dropout),
        )
        self.final_feat_dim = z_dim
        self.classifier = nn.Linear(self.final_feat_dim, num_class)
        self.classifier.bias.data.fill_(0)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def FCN(num_classes=2, x_dim=200, features=False):
    return MLP(x_dim, [3, 4, 23, 3], num_classes=num_classes)

def test():
    net = MLP(10)
    x = torch.randn(1, 10)
    y = net(x)
    print(y)