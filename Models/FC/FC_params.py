#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn, optim


class Dense(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Dense, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True), nn.Dropout(0.5))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Cell(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim, num_features):
        super(Cell, self).__init__()
        self.layers = nn.ModuleList([Dense(in_dim, n_hidden_1, n_hidden_2, out_dim) for i in range(num_features)])
    
    def forward(self, x):
        feature_list = []
        for i, layer in enumerate(self.layers):
            feature_list.append((layer(x[:,i])))
        y = torch.cat(feature_list, dim=1)
        return y

class BrainNet(nn.Module):
    def __init__(self, in_dim, num_features):
        super(BrainNet, self).__init__()
        self.model = nn.Sequential(
            Cell(in_dim, n_hidden_1=32, n_hidden_2=16, out_dim=1, num_features=num_features),
            nn.Linear(num_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            #nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            #nn.Tanh(),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            #nn.Dropout(p=0.2),
            nn.Linear(512, 2),
#             nn.Sigmoid()
        )
    def forward(self,x:torch.tensor):
#         for layer in self.model:
#             layer = layer
        x = self.model(x)
#         print(x)
        return x
#         print(self.model.__class__.__name__,'output shape:',x.shape)
