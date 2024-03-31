#!/usr/bin/env python
# coding=utf-8

import numpy as np
from scipy import signal
from scipy.signal import find_peaks, peak_prominences
from sklearn.linear_model import LinearRegression
from signal_prep import Peak


#find peaks
def Find_peaks(x):
    h = Peak(x)
    peaks, _ = signal.find_peaks(x, height=h, width=5)
    return peaks

#peak nums
def Peak_num(s):
    num_peaks = []
    for i in range(len(s)):
        x = s[i]
        peaks = Find_peaks(x)
        num_peaks.append(len(peaks))
    return num_peaks

#find turning points
def Find_turning_points(s):
    whole_left = []
    whole_right = []
    whole_peaks = []
    value_left = []
    value_right = []
    value_peaks = []
    for i in range(len(s)):
        x = s[i]*10
        peaks = Find_peaks(x)
        valley = (np.diff(np.sign(np.diff(x))) > 0).nonzero()[0] + 1
        dist = valley.reshape(-1,1).repeat([len(peaks)],axis=1) - peaks
        dist = dist.T
        left = []
        right = []
        if len(peaks) != 0:
            for j in range(len(peaks)):
                try:
                    left.append(np.where(dist[j]<0)[0][-1])
                except:
                    peaks = peaks[:-1]
                    continue
                try:
                    right.append(np.where(dist[j]>0)[0][0])
                except:
                    left = left[:-1]
                    peaks = peaks[:-1]
        else:
            left.append(np.array(0))
            right.append(np.array(0))

        try:
            whole_left.append(valley[left])
            whole_right.append(valley[right])
            whole_peaks.append(list(peaks))
            value_left.append(x[valley[left]])
            value_right.append(x[valley[right]])
            value_peaks.append(x[peaks])
        except:
            blank = np.array([])
            whole_left.append(blank)
            whole_right.append(blank)
            whole_peaks.append(blank)
            value_left.append(blank)
            value_right.append(blank)
            value_peaks.append(blank)
    return whole_left, whole_right, whole_peaks, value_left, value_right, value_peaks

#calculate areas
def Areas(whole_left, whole_right, whole_peaks, value_left, value_right, value_peaks):
    #whole_left, whole_right, whole_peaks, value_left, value_right, value_peaks = np.asarray(find_turning_points(s))
    k_left = (value_peaks - value_left) / (whole_peaks - whole_left)
    k_right = (value_peaks - value_right) / (whole_peaks - whole_right)
    b_left = value_peaks - k_left * whole_peaks
    b_right = value_peaks - k_right * whole_peaks
    x_left = -b_left / k_left
    x_right = -b_right / k_right
    areas = (x_right - x_left) * value_peaks * 0.5
    areas_sum = np.zeros([100])
    areas_mean = np.zeros([100])
    for index, area in enumerate(areas):
        areas_sum[index] = np.nansum(np.array(area))
        areas_mean[index] = np.nanmean(np.array(area))
    areas_sum[np.isnan(areas_sum)] = 0
    areas_mean[np.isnan(areas_mean)] = 0
    return areas_sum, areas_mean

#calculate signal span
def Span(whole_left, whole_right):
    span = whole_right - whole_left
    return span

#calculate signal duration and Duty Ratio
def Duration(spans):
    duration_mean = np.zeros([100])
    dr = np.zeros([100])
    for index, span in enumerate(spans):
        #if len(span) != 0:
        duration_mean[index] = np.mean(span)
        dr[index] = np.sum(span) / 960
        #else:
            #duration_mean[index] = 0
            #dr[index] = 0
    return duration_mean, dr

#calculate signal height
def Height(value_peaks):
    height_max = np.zeros([100])
    height_mean = np.zeros([100])
    for index, height in enumerate(value_peaks):
        if len(height) != 0:
            height_max[index] = np.max(height)
            height_mean[index] = np.mean(height)
        else:
            height_max[index] = 0
            height_mean[index] = 0
    return height_max, height_mean

#calculate action potential
def AP(span, value_peaks):
    #span = whole_right - whole_left
    aps = value_peaks / span
    ap_mean = np.zeros([100])
    for index, ap in enumerate(aps):
        ap_mean[index] = np.mean(np.array(ap))
    ap_mean[np.isnan(ap_mean)] = 0
    return ap_mean

'''
#calculate Duty Ratio: signal span/ total time
def DR(spans, total):
    dr = np.zeros([100])
    for index, span in enumerate(spans):
        dr[index] = np.sum(span) / s.shape[1]
    return dr
'''

#calculate interval
def Interval(whole_left, whole_right, whole_peaks):
    interval_mean = np.zeros([100])
    separation_mean = np.zeros([100])
    for i in range(len(whole_peaks)):
        if len(whole_peaks[i]) > 1:
            intervals = whole_left[i][1:] - whole_right[i][:-1]
            separations = np.array(whole_peaks[i][1:]) - np.array(whole_peaks[i][:-1])
            interval_mean[i] = np.mean(intervals)
            separation_mean[i] = np.mean(separations)
        else:
            interval_mean[i] = 1000
            separation_mean[i] = 1000
    return interval_mean, separation_mean

#neural connection
def Neural_connection(s):
    where_peaks = []
    for i in range(len(s)):
        x = s[i]*10
        peaks = Find_peaks(x)
        #h = Peak(x)
        #peaks, _ = signal.find_peaks(x, height=h, width=5)
        where_peaks.extend(peaks)
    return where_peaks

#calculate corr
def Connection_corr(peak_indexs, whole_peaks, s):
    connection_mat = np.zeros([100, len(peak_indexs)])
    value_mat = np.zeros([100, len(peak_indexs)])
    brain_activ = np.zeros([100, 960])
    for index, whole_peak in enumerate(whole_peaks):
        for j, peak in enumerate(peak_indexs):
            if peak in whole_peak:
                connection_mat[index][j] = 1
                value_mat[index][j] = s[index][peak]
                brain_activ[index][peak] = 1
    corr = np.corrcoef(connection_mat)
    corr_value = np.corrcoef(value_mat)
    active_regions = np.sum(brain_activ, axis=0)
    return corr, corr_value, active_regions

'''
#calculate correlation of brain regions by normalized signal
def Region_corr(s):
    corr = np.corrcoef(s)
    corr[np.isnan(corr)] = 0
    return corr
'''
