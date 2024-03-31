#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import ants
import numpy as np
import imutils
import pandas as pd
import time
import argparse

#prep_data: input, crop, and rotate
def prep(fi_path):
    arr_fi = cv2.imread(fi_path,2)
    #arr_mi = cv2.imread(mi_path,0)
    #arr_tp = cv2.imread(tp_path,0)

    #arr_mi_l = arr_mi[:, 0:450]
    #arr_mi_r = arr_mi[:, 650:]
    arr_fi_l = arr_fi[:, 0:600]
    arr_fi_r = arr_fi[:, 800:]

    rotated_mi_l = imutils.rotate(arr_mi_l, 26)
    rotated_mi_r = imutils.rotate(arr_mi_r, -26)
    #rotated_fi_l = imutils.rotate(arr_fi_l, 180-26).astype(np.uint32) #pay attention to the rotation angle
    #rotated_fi_r = imutils.rotate(arr_fi_r, 180+26).astype(np.uint32)

    #fi = ants.from_numpy(arr_fi, has_components=False)
    #tp = ants.from_numpy(arr_tp, has_components=False)
    #mi_l = ants.from_numpy(rotated_mi_l, has_components=False)
    #mi_r = ants.from_numpy(rotated_mi_r, has_components=False)
    fi_l = ants.from_numpy(rotated_fi_l, has_components=False)
    fi_r = ants.from_numpy(rotated_fi_r, has_components=False)

    return fi_l,fi_r

#registration with ants
def registration(fi, mi, name):
    fixed = ants.resample_image(fi,(100,80),1,0)
    moving = ants.resample_image(mi,(100,80),1,0)
    ants.registration(fixed=fixed , moving=moving, type_of_transform='ElasticSyN', outprefix=name, aff_random_sampling_rate=0.1, random_seed=1, smoothing_sigmas=(4,3,2,1), aff_shrink_factors=(8,4,2,1), write_composite_transform=True, verbose=False)
    #mywarpedimage = ants.apply_transforms(fixed=fi, moving=mi, transformlist=mytx['fwdtransforms'], interpolator='gaussian')

    #return mywarpedimage

#define the input data
def parse_args():
    description = "input your dir"
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('-a','--address', help = "The path of address", required=False)
    parser.add_argument('-f','--file', help = "The path file of addresses", required=False)
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()

    if(args.address):
        file_dir = args.address
        #new_dir = mi_dir.replace('OldDrug/Brain_gzh', 'ANTs/double')
        #if not os.path.exists(new_dir):
            #os.makedirs(new_dir)
        for root, dirs, files in os.walk(file_dir):
            if len(files)>800:
                new_dir = root.replace('/data2/zhshen/data/zebra_fish/block1', '/home/zhshen/data/zebra_fish/DATA_TEMP/mytu/ANTs/inter')
                '''
                try:
                    pub_tp = root.replace('cont', 'test')
                except:
                    pub_tp = root
                '''
                fi_path = os.path.join(root, 'img_000000500_Default_000.tif')
                fi_l,fi_r = prep(fi_path)
                #tp_choice_l = []
                #tp_choice_r = []

                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                for file in files:
                    if os.path.splitext(file)[1] == '.tif': 
                        name = os.path.splitext(file)[0]
                        namel = new_dir + '/l' + name
                        namer = new_dir + '/r' + name
                        
                        mi_path = os.path.join(root, file)
                        mi_l,mi_r = prep(mi_path)
                        
                        registration(fi_l, mi_l, namel)
                        registration(fi_r, mi_r, namer)

    if(args.file):
        for file_dir in open(args.file):
            file_dir = file_dir.strip()
            for root, dirs, files in os.walk(file_dir):
                if len(files)>800:
                    new_dir = root.replace('/data2/zhshen/data/zebra_fish/block1', '/home/zhshen/data/zebra_fish/DATA_TEMP/mytu/ANTs/inter')
                    '''
                    try:
                        pub_tp = root.replace('cont', 'test')
                    except:
                        pub_tp = root
                    '''
                    fi_path = os.path.join(root, 'img_000000500_Default_000.tif')

                    fi_l,fi_r = prep(fi_path)
                    #tp_choice_l = []
                    #tp_choice_r = []

                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    for file in files:
                        if os.path.splitext(file)[1] == '.tif': 
                            name = os.path.splitext(file)[0]
                            namel = new_dir + '/l' + name
                            namer = new_dir + '/r' + name
                            
                            mi_path = os.path.join(root, file)
                            mi_l,mi_r = prep(mi_path)
                            
                            registration(fi_l, mi_l, namel)
                            registration(fi_r, mi_r, namer)

