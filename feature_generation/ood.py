#!/usr/bin/env python
# coding=utf-8

import numpy as np
from collections import Counter

#find fly frames
def Fly_frame(judge):
    q3, q1 = np.percentile(judge, [80, 20], axis=0)
    iqr = q3-q1
    fence_low  = q1-1.5*iqr
    return np.where(judge<fence_low)

#calculate time range
def Time_range(fish_mat_preped):
    time_max = np.max(fish_mat_preped, axis=0)
    time_min = np.min(fish_mat_preped, axis=0)
    time_range = time_max - time_min
    return time_range

#delete dead fish
def Dead_fish(time_range):
    if sum(i>1000 for i in time_range) > 10000:
        return True

#delete fly frames
def Need_move(fish_only):
    fly = Fly_frame(fish_only)
    result = Counter(fly[0])
    need_move = [k for k, v in result.items() if v > 1000]
    return need_move

#delete error points
def Outliers(time_range):
    outliers = np.where(time_range>500)
    return outliers

#extract fish points
def Fish_only(fish_mat, fish_value):
    f_value = fish_value.copy()
    f_value[f_value>0] = 1
    fish = fish_mat * f_value.flatten()
    fish_index = np.where(fish[0]==0)
    fish_only = np.delete(fish, fish_index, axis=1)
    return fish_only

#find outliers(frames)
def Frame_outliers(fish_mat, fish_value):
    fish_only = Fish_only(fish_mat, fish_value)
    fly = Fly_frame(fish_only)
    result = Counter(fly[0])
    need_move = [k for k, v in result.items() if v > 1000]
    return need_move
