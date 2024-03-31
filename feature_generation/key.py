#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import ants
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LinearRegression
from scipy import signal
from scipy.signal import find_peaks, peak_prominences
from feature_extraction import *
from ood import *
from signal_prep import *
import argparse
import pickle


def Features(mat_path, tp_path, tp, path):
    #read data
    fish_mat = np.load(mat_path)
    #transform data
    mi= ants.from_numpy(np.array(fish_mat[500].reshape(-1, 250), dtype = np.uint32), has_components=False)
    transformed = ants.apply_transforms(fixed=mi, moving=tp, transformlist=tp_path, interpolator='nearestNeighbor')
    fish_value = transformed.numpy()

    #delete fly frames
    need_move = Frame_outliers(fish_mat, fish_value)  #find outliers(frames)
    fish_mat_preped = np.delete(fish_mat, need_move, axis=0)

    #prep
    c = Sparse_tp(fish_value)
    time_range = Time_range(fish_mat_preped)

    if not os.path.exists(path[:-1]):
        os.makedirs(path[:-1])

    if Dead_fish(time_range):
        name = path + 'Dead'
        Path(name).touch()

    else:
        outliers = np.where(time_range>500)  #find outliers(points)
        fish_mat_preped.T[outliers] = 0
        #c.T[outliers] = 0
        #print(c.shape)
        #print(fish_mat_preped.shape)
        s = Norm_time(fish_mat_preped, c)

        whole_left, whole_right, whole_peaks, value_left, value_right, value_peaks = np.asarray(Find_turning_points(s))

        #features
        num_peaks = Peak_num(s)
        areas_sum, areas_mean = Areas(whole_left, whole_right, whole_peaks, value_left, value_right, value_peaks)
        spans = Span(whole_left, whole_right)
        duration_mean, dr = Duration(spans)
        duration_mean[np.isnan(duration_mean)] = 0
        height_max, height_mean = Height(value_peaks)
        ap_mean = AP(spans, value_peaks)
        interval_mean, separation_mean = Interval(whole_left, whole_right, whole_peaks)
        where_peaks = Neural_connection(s)
        peak_indexs = sorted(np.array(list(set(where_peaks))))
        connection_corr, corr_value, active_regions = Connection_corr(peak_indexs, whole_peaks, s)
        connection_corr[np.isnan(connection_corr)] = 0
        corr_value[np.isnan(corr_value)] = 0
        #return num_peaks, areas_sum, areas_mean, ap_mean, connection_corr
        #features = np.vstack((num_peaks, areas_sum, areas_mean, ap_mean, connection_corr))

        '''
        if not os.path.exists(path):
            os.makedirs(path)
        '''

        #save features
        #np.save(path, features)
        features_dict = {'num_peaks': num_peaks, 'areas_sum': areas_sum, 'areas_mean': areas_mean, 'duration_mean': duration_mean, 'dr': dr, 'height_max': height_max, 'height_mean': height_mean, 'ap_mean': ap_mean, 'interval_mean': interval_mean, 'separation_mean': separation_mean, 'connection_corr': connection_corr, 'value_corr': corr_value, 'active_regions': active_regions}
        path = path + '.pkl'
        f = open(path, wb)
        pickle.dump(features_dict, f)
        f.close()

#define the input data
def parse_args():
    description = "input your dir"
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('-a','--address', help = "The path of fish mat", required=False)
    parser.add_argument('-f','--files', help = "File paths of fish mat", required=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #read template
    arr_tp = cv2.imread('/home/mytu/zebrafish/Heatmap/tp3_100.png', 0)
    tp = ants.from_numpy(arr_tp, has_components=False)
    #arr_fi = cv2.imread('/home/mytu/zebrafish/Heatmap/template3.tif', 2)
    #fi = ants.from_numpy(arr_fi.astype(np.uint32), has_components=False)
    
    args = parse_args()
    
    if(args.address):
        mat_path = args.address
        #path
        mat_path_l = mat_path + '/l.npy'  #data path
        tp_path_l = mat_path.replace('aligned12_raw', 'tp') + '/lInverseComposite.h5'  #template path
        feature_path_l = mat_path.replace('/home/zhshen/data/zebra_fish/DATA_TEMP/mytu/ANTs/aligned12_raw', '/data2/mytu/feature_generation/features') + '/1'
        Features(mat_path_l, tp_path_l, tp, feature_path_l)

        mat_path_r = mat_path + '/r.npy'  #data path
        tp_path_r = mat_path.replace('aligned12_raw', 'tp') + '/rInverseComposite.h5'  #template path
        feature_path_r = mat_path.replace('/home/zhshen/data/zebra_fish/DATA_TEMP/mytu/ANTs/aligned12_raw', '/data2/mytu/feature_generation/features') + '/2'
        Features(mat_path_r, tp_path_r, tp, feature_path_r)

    if(args.files):
        for file_dir in open(args.files):
            mat_path = file_dir.strip()
            #path
            mat_path_l = mat_path + '/l.npy'  #data path
            tp_path_l = mat_path.replace('aligned12_raw', 'tp') + '/lInverseComposite.h5'  #template path
            feature_path_l = mat_path.replace('/home/zhshen/data/zebra_fish/DATA_TEMP/mytu/ANTs/aligned12_raw', '/data2/mytu/feature_generation/features') + '/1'
            Features(mat_path_l, tp_path_l, tp, feature_path_l)

            mat_path_r = mat_path + '/r.npy'  #data path
            tp_path_r = mat_path.replace('aligned12_raw', 'tp') + '/rInverseComposite.h5'  #template path
            feature_path_r = mat_path.replace('/home/zhshen/data/zebra_fish/DATA_TEMP/mytu/ANTs/aligned12_raw', '/data2/mytu/feature_generation/features') + '/2'
            Features(mat_path_r, tp_path_r, tp, feature_path_r)
