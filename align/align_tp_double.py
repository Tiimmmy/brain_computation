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

#rotate image
def rotate_img(img, angle):
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    #get rotation moment
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    #calculate new boundary
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    #adjust rotation moments
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))

    return rotated_img

#split image
def split_tif(img):
    h, w = img.shape[:2]
    image_left = img[0:h, 0:int(w/2)]
    image_right = img[0:h, int(w/2):int(w)]
    #image_left = img[:, 0:450]
    #image_right = img[:, 600:]
    #rotated_l = imutils.rotate(image_left, 26)
    #rotated_r = imutils.rotate(image_right, -26)
    rotated_l = rotate_img(image_left, 26)
    rotated_r = rotate_img(image_right, -26)
    return rotated_l, rotated_r

#find fish center
def find_center(img):
    pts = np.flip(np.column_stack(np.where(img > 3)), axis=1)
    center = np.mean(pts, axis=0)
    return center

#crop image
def crop_img(img, h_raw=500, w_raw=300):
    #h, w = img.shape[:2]
    img_center = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    x, y = find_center(img_center)
    cropped_img = img[int(y-0.5*h_raw):int(y+0.5*h_raw), int(x-0.5*w_raw):int(x+0.5*w_raw)]
    return cropped_img

#registration
def registration(fi, mi, name):
    fixed = ants.resample_image(fi,(100,80),1,0)
    moving = ants.resample_image(mi,(100,80),1,0)
    ants.registration(fixed=fixed , moving=moving, type_of_transform='ElasticSyN', outprefix=name, aff_random_sampling_rate=0.1, random_seed=1, smoothing_sigmas=(4,3,2,1), aff_shrink_factors=(8,4,2,1), write_composite_transform=True, verbose=False)

#save data
def save_mat(fi, img, name):
    #fi = ants.from_numpy(arr_fi.astype(np.uint32), has_components=False)
    mi = ants.from_numpy(img.astype(np.uint32), has_components=False)
    registration(fi, mi, name)

#function
def function(file_dir, fi):
    for root, dirs, files in os.walk(file_dir):
        if len(files)>800:
            tp_dir = root.replace('/home/mytu/zebrafish/Drugs_gzh/BrainTest', '/home/NVMe/mytu/zebrafish/tp')
            fish_path = os.path.join(root, 'img_000000500_Default_000.tif')

            if not os.path.exists(tp_dir):
                os.makedirs(tp_dir)

            namel = tp_dir + '/l'
            namer = tp_dir + '/r'
            image_arr = cv2.imread(fish_path, 2)
            img = split_tif(image_arr)
            img_l, img_r = crop_img(img[0]), crop_img(img[1])
            save_mat(fi, img_l, namel)
            save_mat(fi, img_r, namer)

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
    fi_path = '/home/mytu/zebrafish/Alignment/template.tif'
    arr_fi = cv2.imread(fi_path,2)
    fi = ants.from_numpy(arr_fi.astype(np.uint32), has_components=False)

    if(args.address):
        file_dir = args.address
        try:
            function(file_dir, fi)
        except Exception as e:
            print(file_dir)
            print(e)

        
    if(args.file):
        for file_dir in open(args.file):
            file_dir = file_dir.strip()
            try:
                function(file_dir, fi)
            except Exception as e:
                print(file_dir)
                print(e)






