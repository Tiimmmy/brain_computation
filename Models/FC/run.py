#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold

from FC_params import *
from train import *

import argparse


def parse_args():
    description = 'input your path'
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('-d', '--data', help = 'data path')
    parser.add_argument('-l', '--label', help = 'label path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    feature_path = args.data
    label_path = args.label
    #feature_path = '/data2/mytu/feature_generation/features/features_norm_all.npy'
    #label_path = '/data2/mytu/feature_generation/features/label_norm.npy'
    #print(feature_path)
    #dataset and label
    final_dataset = data_loader(feature_path, label_path)
    
    #model
    device =torch.device("cuda:2" if torch.cuda.is_available else "cpu")
    Net = BrainNet(in_dim=16, num_features=2000)
    Net.to(device)
    #print(Net)
    #cross validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    #hyperparameters
    weight_decay = 1e-5
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(Net.parameters(), lr=1e-3)
    optimizer = optim.SGD(Net.parameters(), lr=1e-2, momentum=0.90, weight_decay=weight_decay)
    num_epoches = 200

    #results_lists
    train_acc = []
    test_acc = []
    predicted = []
    label_list = []
    test_acc_max = []

    K_Fold(final_dataset, num_epoches, kfold, Net, train_acc, test_acc, predicted, label_list, test_acc_max, device, optimizer, weight_decay, criterion)

    #accuracy and precision
    final_acc = np.mean(test_acc_max)
    prediction = Confusion(predicted, label_list)
    precision = prediction[0] / (prediction[0] + prediction[1])

    #save results
    print('accuracy:', final_acc, 'precision:', precision)
    np.savetxt('train_acc.txt', train_acc)
    np.savetxt('test_acc.txt', test_acc)
    np.save('predicted', predicted)
    np.save('label_list', label_list)
    np.savetxt('test_acc_max', test_acc_max)
