#!/usr/bin/env python
# coding=utf-8

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold


def data_loader(feature_path, label_path):
    features = np.load(feature_path)
    label = np.load(label_path)
    X = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.array(label, dtype=np.uint8))
    final_dataset = TensorDataset(X, y.type(torch.LongTensor))
    return final_dataset

def train(train_loader, epoch, train_accu, model, device, optimizer, weight_decay, criterion):
    print('\nEpoch : %d'%epoch)
    
    model.train()

    running_loss=0
    correct=0
    total=0
    prediction=[]
    
    for data in train_loader:
        
        inputs,labels = data[0].to(device),data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        #L1 regularization
#         regularization_loss = 0
#         for param in model.parameters():
#             regularization_loss += torch.sum(abs(param))
#             regularization_loss += torch.sum(param ** 2)
        
#         classify_loss = criterion(outputs,labels)
#         loss = classify_loss + weight_decay * regularization_loss
        
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        prediction.append(predicted)
       
    train_loss = running_loss/len(train_loader)
    accu = 100.*correct/total

    train_accu.append(accu)
    #train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

def test(test_loader, epoch, test_accu, model, prediction, test_label, device, criterion):
    model.eval()

    running_loss=0
    correct=0
    total=0

    with torch.no_grad():
        for data in test_loader:
            images,labels = data[0].to(device),data[1].to(device)

            outputs = model(images)

            loss = criterion(outputs,labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
#             print(predicted)
#             prediction.append(predicted)
            prediction.append(predicted)
            test_label.append(labels)

    test_loss = running_loss/len(test_loader)
    accu = 100.*correct/total

    test_accu.append(accu)
    #test_losses.append(test_loss)

    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def K_Fold(final_dataset, num_epoches, kfold, model, train_acc, test_acc, predicted, label_list, test_acc_max, device, optimizer, weight_decay, criterion):

    for fold, (train_ids, val_ids) in enumerate(kfold.split(final_dataset)):
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        # Adding subsamples to dataloader to create batches.
        train_loader = DataLoader(final_dataset, batch_size=len(train_subsampler), sampler=train_subsampler, num_workers=16)
        test_loader = DataLoader(final_dataset, batch_size=len(val_subsampler), sampler=val_subsampler, num_workers=16)
        
        model.apply(reset_weights)
        print('Reset trainable parameters of layers')
        
        result_train = []
        result_test = []
        prediction = []
        test_label = []
        
        for epoch in range(1,num_epoches+1): 
            train(train_loader, epoch, result_train, model, device, optimizer, weight_decay, criterion)
            test(test_loader, epoch, result_test, model, prediction, test_label, device, criterion)
        
        #document train and test accu 
        train_acc.extend(result_train)
        test_acc.extend(result_test)
        
        #find the best param
        test_acc_max.append(max(result_test))
        best = result_test.index(max(result_test))
        predicted.append(prediction[best].cpu())
        label_list.append(test_label[best].cpu())
    #return test_acc_max

def Confusion(predicted, label_list):
    tp=[]
    fp=[]
    tn=[]
    fn=[]

    for i, preds in enumerate(predicted):
        tp.append(((preds == 1) & (label_list[i] == 1)).sum())
        fp.append(((preds == 1) & (label_list[i] == 0)).sum())
        tn.append(((preds == 0) & (label_list[i] == 0)).sum())
        fn.append(((preds == 0) & (label_list[i] == 1)).sum())
    return np.array(tp), np.array(fp), np.array(tn), np.array(fn)
