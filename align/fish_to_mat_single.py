#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import ants
import numpy as np
import imutils
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

#prep 1000pics
def prep_mi(mi_path):
    arr_mi = cv2.imread(mi_path,2)
    h, w = arr_mi.shape[:2]
    arr_mi_l = arr_mi[0:h, 0:int(w/2)]
    arr_mi_r = arr_mi[0:h, int(w/2):int(w)]
    #arr_mi_l = arr_mi[:, 0:450]
    #arr_mi_r = arr_mi[:, 600:]
    rotated_mi_l = rotate_img(arr_mi_l, 26)
    rotated_mi_r = rotate_img(arr_mi_r, -26)
    #mi_l = ants.from_numpy(rotated_mi_l.astype(np.uint32), has_components=False)
    #mi_r = ants.from_numpy(rotated_mi_r.astype(np.uint32), has_components=False)
    return rotated_mi_l, rotated_mi_r

def load_mi(rotated_mi_l, rotated_mi_r):
    mi_l = ants.from_numpy(rotated_mi_l.astype(np.uint32), has_components=False)
    mi_r = ants.from_numpy(rotated_mi_r.astype(np.uint32), has_components=False)
    return mi_l, mi_r

#crop image
def find_center(img):
    pts = np.flip(np.column_stack(np.where(img > 3)), axis=1)
    center = np.mean(pts, axis=0)
    return center

def crop_center(img):
    img_center = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    x, y = find_center(img_center)
    return x, y

def crop_img(img, center, h_raw=500, w_raw=300):
    x, y = center
    cropped_img = img[int(y-0.5*h_raw):int(y+0.5*h_raw), int(x-0.5*w_raw):int(x+0.5*w_raw)]
    return cropped_img

#generate mat of the video
def read_value(file_dir):
    for root, dirs, files in os.walk(file_dir):
        if len(files)>800:
            
            #prep template
            tp_path = os.path.join(root, 'img_000000500_Default_000.tif')
            rotated_mi_l, rotated_mi_r = prep_mi(tp_path)
            mi_l, mi_r = load_mi(rotated_mi_l, rotated_mi_r)
            
            center_l = crop_center(rotated_mi_l)
            tp_l = crop_img(rotated_mi_l.astype(np.uint16), center_l).flatten()
            df_l = np.zeros_like([tp_l]*1000)
            #print(df_l.shape)

            center_r = crop_center(rotated_mi_r)
            tp_r = crop_img(rotated_mi_r.astype(np.uint16), center_r).flatten()
            df_r = np.zeros_like([tp_r]*1000)

            #make dirs
            mat_dir = root.replace('/home/mytu/zebrafish/Drugs_gzh/BrainTest', '/home/zhshen/data/zebra_fish/DATA_TEMP/mytu/ANTs/inter_all')
            #tp_dir = root.replace('Drugs_gzh/BrainTest', 'ANTs/fish_tp')
            value_dir = root.replace('/home/mytu/zebrafish/Drugs_gzh/BrainTest', '/home/zhshen/data/zebra_fish/DATA_TEMP/mytu/ANTs/aligned')

            if not os.path.exists(value_dir):
                os.makedirs(value_dir)

            for file in files:
                if os.path.splitext(file)[1] == '.tif':
                    fish_num = int(file.split('_')[1])
                    name = os.path.splitext(file)[0]
                    namel = mat_dir + '/l' + name + 'Composite.h5'
                    namer = mat_dir + '/r' + name + 'Composite.h5'
                    fish_path = os.path.join(root, file)
                    rotated_m_l, rotated_m_r = prep_mi(fish_path)
                    m_l, m_r = load_mi(rotated_m_l, rotated_m_r)

                    try:
                        aftertransform_l = ants.apply_transforms(fixed=mi_l, moving=m_l, transformlist=namel, interpolator='gaussian')
                        aftertransform_r = ants.apply_transforms(fixed=mi_r, moving=m_r, transformlist=namer, interpolator='gaussian')
                    except:
                        continue

                    #center_l = crop_center(rotated_mi_l)
                    map_l = crop_img(aftertransform_l.numpy().astype(np.uint16), center_l).flatten()
                    df_l[fish_num] = map_l
                    #print(map_l.shape)
                    #center_r = crop_center(rotated_mi_r)
                    map_r = crop_img(aftertransform_r.numpy().astype(np.uint16), center_r).flatten()
                    df_r[fish_num] = map_r

            value_l = value_dir + '/l' + '.npy'
            value_r = value_dir + '/r' + '.npy'
            np.save(value_l, df_l)
            np.save(value_r, df_r)
            
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

    #if the input is an address
    if(args.address):
        try:
            file_dir = args.address
            read_value(file_dir)
        except Exception as e:
            print(file_dir)
            print(e)

    #if the input is a file contains paths
    if(args.file):
        for file_dir in open(args.file):
            try:
                file_dir = file_dir.strip()
                read_value(file_dir)
            except Exception as e:
                print(file_dir)
                print(e)

