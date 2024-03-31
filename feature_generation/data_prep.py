#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np


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

#find fish center
def find_center(img):
    pts = np.flip(np.column_stack(np.where(img > 3)), axis=1)
    center = np.mean(pts, axis=0)
    return center

#crop image
def crop_img(img, h_raw=450, w_raw=250):
    #h, w = img.shape[:2]
    img_center = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    x, y = find_center(img_center)
    cropped_img = img[int(y-0.5*h_raw):int(y+0.5*h_raw), int(x-0.5*w_raw):int(x+0.5*w_raw)]
    return cropped_img

#prep 1000pics
def prep_mi(mi_path):
    arr_mi = cv2.imread(mi_path,2)
    h, w = arr_mi.shape[:2]
    arr_mi_l = arr_mi[0:h, 0:int(w/2)]
    arr_mi_r = arr_mi[0:h, int(w/2):int(w)]
    rotated_mi_l = rotate_img(arr_mi_l, 26)
    rotated_mi_r = rotate_img(arr_mi_r, -26)
#     arr_mi_l = arr_mi[:, 0:450]
#     arr_mi_r = arr_mi[:, 600:]
#     rotated_mi_l = imutils.rotate(arr_mi_l, 26)
#     rotated_mi_r = imutils.rotate(arr_mi_r, -26)
    cropped_l = crop_img(rotated_mi_l)
    cropped_r = crop_img(rotated_mi_r)

    mi_l = ants.from_numpy(cropped_l.astype(np.uint32), has_components=False)
    mi_r = ants.from_numpy(cropped_r.astype(np.uint32), has_components=False)

    return mi_l,mi_r


