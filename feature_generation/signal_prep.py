#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import signal


#generate sparse matrix
def Sparse_tp(tp):
    a = np.expand_dims(tp.flatten(), 0)
    b = a.repeat([100], axis=0)
    c = np.zeros_like(b)
    for i in range(1, b.shape[0]+1):
        c[i-1] = np.where(b[i-1] == i, 1, 0)
    return c

#baseline detector
class BaselineRemoval():
    '''input_array: A pandas dataframe column provided in input as dataframe['input_df_column']. It can also be a Python list object
    degree: Polynomial degree
    '''
    def __init__(self,input_array):
        self.input_array=input_array
        self.lin=LinearRegression()

    def poly(self,input_array_for_poly,degree_for_poly):
        '''qr factorization of a matrix. q` is orthonormal and `r` is upper-triangular.
        - QR decomposition is equivalent to Gram Schmidt orthogonalization, which builds a sequence of orthogonal polynomials that approximate your function with minimal least-squares error
        - in the next step, discard the first column from above matrix.
        - for each value in the range of polynomial, starting from index 0 of pollynomial range, (for k in range(p+1))
            create an array in such a way that elements of array are (original_individual_value)^polynomial_index (x**k)
        - concatenate all of these arrays created through loop, as a master array. This is done through (np.vstack)
        - transpose the master array, so that its more like a tabular form(np.transpose)'''
        input_array_for_poly = np.array(input_array_for_poly)
        X = np.transpose(np.vstack((input_array_for_poly**k for k in range(degree_for_poly+1))))
        return np.linalg.qr(X)[0][:,1:]

    def IModPoly(self,degree=2,repitition=100,gradient=0.001):
        '''IModPoly from paper: Automated Autofluorescence Background Subtraction Algorithm for Biomedical Raman Spectroscopy, by Zhao, Jianhua, Lui, Harvey, McLean, David I., Zeng, Haishan (2007)
        degree: Polynomial degree, default is 2
        repitition: How many iterations to run. Default is 100
        gradient: Gradient for polynomial loss, default is 0.001. It measures incremental gain over each iteration. If gain in any iteration is less than this, further improvement will stop
        '''

        yold=np.array(self.input_array)
        yorig=np.array(self.input_array)

        nrep=1
        ngradient=1

        polx=self.poly(list(range(1,len(yorig)+1)),degree)
        ypred=self.lin.fit(polx,yold).predict(polx)
        Previous_Dev=np.std(yorig-ypred)

        #iteration1
        yold=yold[yorig<=(ypred+Previous_Dev)]
        polx_updated=polx[yorig<=(ypred+Previous_Dev)]
        ypred=ypred[yorig<=(ypred+Previous_Dev)]

        for i in range(2,repitition+1):
            if i>2:
                Previous_Dev=DEV
            ypred=self.lin.fit(polx_updated,yold).predict(polx_updated)
            DEV=np.std(yold-ypred)

            if np.abs((DEV-Previous_Dev)/DEV) < gradient:
                break
            else:
                for i in range(len(yold)):
                    if yold[i]>=ypred[i]+DEV:
                        yold[i]=ypred[i]+DEV
        baseline=self.lin.predict(polx)
        return baseline

#find peak positions
def Peak(judge):
    q3, q1 = np.percentile(judge, [75, 25], axis=0)
    iqr = q3-q1
    fence_high = q3+2*iqr
    return fence_high

#generate time
def Time(fish_mat_preped, c):
    time = np.dot(c, fish_mat_preped.T)
    time = time.T / np.sum(c, axis=1)
    time = time.T
    return time

#filter
def Filter(time):
    b, a = signal.butter(30, 0.3, 'lowpass')
    filtedData = signal.filtfilt(b, a, time)
    return filtedData

#interplot flied frames
def Interplot(filtedData, need_move):
    if len(need_move) > 1:
        vary = np.arange(0, len(need_move))
        need_interp = need_move - vary
        interp = np.insert(filtedData, need_interp, np.nan, axis=1)
    else:
        interp = np.insert(filtedData, need_move, np.nan, axis=1)
    df = pd.DataFrame(interp)
    df.interpolate(method='linear', axis=1, inplace=True)
    interplot_signal = np.array(df)
    return interplot_signal

#generate baseline
def Baseline(filtedData):
    #filtedData = filtedData.T[10:-10].T
    baseline = np.zeros([filtedData.shape[0], filtedData.shape[1]])
    polynomial_degree=6
    for i in range(len(filtedData)):
        #print(np.isnan(filtedData[i]))
        #print(np.isinf(filtedData[i]))
        baseObj = BaselineRemoval(filtedData[i])
        baseline[i] = baseObj.IModPoly(polynomial_degree)
    return baseline

#normalize calcium signal
def Norm_time(fish_mat_preped, c, need_move):
    #get time
    time = Time(fish_mat_preped, c)
    #print(time)
    #print(time.shape)

    #filter
    filtedData = Filter(time)
    filtedData = Interplot(filtedData, need_move)
    filtedData[np.isnan(filtedData)] = 0
    filtedData[np.isinf(filtedData)] = 0
    filtedData = filtedData.T[10:-10].T
    baseline = Baseline(filtedData)
    s = (filtedData.T[10:-10].T - baseline.T[10:-10].T) / baseline.T[10:-10].T
    return s
