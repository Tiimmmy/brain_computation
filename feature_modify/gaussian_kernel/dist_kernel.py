#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import numpy as np
import pandas as pd
import argparse


#calculate Euclidean Distances between centers
def EuclideanDistances(X):
    G = np.dot(X, X.T)
    H = np.tile(np.diag(G), (G.shape[1], 1))
    D = H + H.T - G * 2
    return D

##calculate the Gaussian kernel of distance
def dist_simi(num, a):
    pointname = '/data2/mytu/codes_zebrafish/segment/tp3_' + str(num) + '.txt'
    X = np.loadtxt(pointname)
    dist = EuclideanDistances(X)
    d = np.sort(np.sqrt(dist), axis=1).T[1]
    sigma = a * d
    kernel = np.exp(-dist / (2 * sigma**2))
    kernel_at_pos = kernel / np.sum(kernel, axis=0)
    return kernel_at_pos

#generate fish mat
def mat_fish(seg, n):
    a = np.expand_dims(seg.flatten(), 0)
    b = (a).repeat([n], axis=0)
    return b

#generate sparse matrix
def sparse(seg, num):
    #generate the region index
    region = pd.DataFrame(
    {
        "region": np.arange(1,int(num)+1)
    }
    )

    a = seg.flatten()
    df = region['region'].apply(lambda x: np.where(a == x, 1, 0))
    df = np.array(df.to_list(), dtype=np.uint16)
    return df

#final process
def final(data_path, num, h):
    features_raw = np.load(data_path)
    features_merge = np.concatenate(np.concatenate(features_raw, axis=1), axis=0)  # all_fish_features_N * brain_seg_num
    kernel_at_pos = dist_simi(num, h)
    map_after_kernel = np.dot(kernel_at_pos, features_merge.T)
    feature_after_kernel = np.array_split(np.array(np.array_split(map_after_kernel.T, features_raw.shape[1], axis=0)), features_raw.shape[0], axis=1)  # return to original shape
    return np.array(feature_after_kernel)

#define the input data
def parse_args():
    description = "input information"
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('-path','--data_path', help = "Input data path", required=True)
    parser.add_argument('-H','--band_width', help = "Define the value of band_width (the final band width will times to distance)", required=True)
    parser.add_argument('-N','--num', help = "Input brain-segment-nums", required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    data_path = args.data_path
    h = float(args.band_width)
    num = int(args.num)

    features_gaussian = final(data_path, num, h)
    np.save('features_gaussian{0}_{1}.npy'.format(num, h), features_gaussian)


