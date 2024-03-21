# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 18:24:58 2024

复杂预测模型的全局变量评估方法研究 
一种基于HDMR的稳健的变量全局重要性方法

@author: ling
"""
import pandas as pd
import numpy as np

class featureImportance:
    def __init__(self, X, Y, 
                 intervals = 30, seg = "quantile",  # 区间划分方法未提及
                 head_tail = (0, 1)): # 区间划分法中基于区间的

        self.X = pd.DataFrame(X)
        self.Y = Y
        self.intervals = intervals # the number of intervals
        self.numfea = X.shape[1] # the number of intervals
        self.seg = seg # 区间划分方法
        self.head_tail = head_tail # the head and tail of the samples in percentage
        #NumPy提供了您可以/应该用于子类型检查的基类，而不是Python类型。
        self.num_check = self.X.apply(lambda x: np.issubdtype(x.dtype, np.number))  # vector, index number of numeric feature 
        # self.num_check = (X.apply(lambda x: np.issubdtype(x.dtype, np.integer)) |
        #     X.apply(lambda x: np.issubdtype(x.dtype, np.inexact)))# 整数，浮点数（floating）或复数浮点数
            #X.astype({'CHAS':np.bool,'RAD':np.character})
        self.fac_check = ~self.num_check # vector, index number of factor feature
        self.__get_cutpoints()
        self.__cal_conditional_mval()

     #------------ get cutpoints ------------   
    def __get_cutpoints(self): 
        # matrix, the set of cutting points
        self.cutpoints = np.full((self.numfea, self.intervals + 1), np.nan)#np.empty((self.numfea, self.intervals + 1))
        for i in np.arange(self.numfea)[self.num_check]:
            x = self.X.iloc[:, i]
            if self.seg == "equal" :
                self.cutpoints[i,] = np.linspace(np.quantile(x, self.head_tail[0]),
                                                 np.quantile(x, self.head_tail[1]),
                                                 num = self.intervals + 1)
            else:
                self.cutpoints[i,] = np.quantile(x, np.linspace(0, 1, num = self.intervals + 1))
            self.cutpoints[i, 0] = -float('inf')
            self.cutpoints[i, self.intervals] = float('inf')
        
    #------------ calculate conditional mean and median ------------
    def __cal_conditional_mval(self):
        self.condi_mean = [np.nan] * self.numfea
        self.condi_median = [np.nan] * self.numfea
        for i in np.arange(self.numfea)[self.num_check]:
            self.condi_mean[i] = np.empty((self.intervals, 2))
            self.condi_median[i] = np.empty((self.intervals, 2))
            x = self.X.iloc[:, i]
            for j in range(self.intervals):
                indx = (x > self.cutpoints[i,j]) & (x <= self.cutpoints[i,j+1]) 
                # the frequency (weight) in each interval
                self.condi_mean[i][j, 1] = np.sum(indx)
                self.condi_median[i][j, 1] = self.condi_mean[i][j, 1]
                
                if self.condi_mean[i][j,1] > 0:
                    # conditional mean
                    self.condi_mean[i][j, 0] = np.mean(self.Y[indx])
                    # conditional median
                    self.condi_median[i][j, 0] = np.median(self.Y[indx])
                    
                else:
                    # conditional mean
                    self.condi_mean[i][j, 0] = 0
                    # conditional median
                    self.condi_median[i][j, 0] = 0
        for i in np.arange(self.numfea)[self.fac_check]:
            x = self.X.iloc[:, i]
            levels = pd.factorize(x)[1]
            
            self.condi_mean[i] = np.empty((levels.shape[0], 2))
            self.condi_median[i] = np.empty((levels.shape[0], 2))
            for j in range(levels.shape[0]):
                indx = (x == levels[j])
                
                # the frequency (weight) in each interval
                self.condi_mean[i][j, 1] = np.sum(indx)
                self.condi_median[i][j, 1] = self.condi_mean[i][j, 1]
                
                if self.condi_mean[i][j,1] > 0:
                    # conditional mean
                    self.condi_mean[i][j, 0] = np.mean(self.Y[indx])
                    # conditional median
                    self.condi_median[i][j, 0] = np.median(self.Y[indx])
                    
                else:
                    # conditional mean
                    self.condi_mean[i][j, 0] = 0
                    # conditional median
                    self.condi_median[i][j, 0] = 0

    def cal_feaimp_var_mean(self):
        feaimp = [np.nan] * self.numfea
        for i in range(self.numfea):
            c_mean = self.condi_mean[i]
            # calculate weighted mean
            m = np.sum(c_mean[:,0] * c_mean[:,1]) / np.sum(c_mean[:,1])
            # calculate variance
            feaimp[i] = np.sqrt(np.sum((c_mean[:,0] - m) ** 2 * c_mean[:,1]) / np.sum(c_mean[:,1]))
        return np.array(feaimp)


if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    dataXY = fetch_california_housing()
    X = dataXY.data
    Y = dataXY.target
    # X1 = pd.DataFrame(X,columns=load_boston().feature_names)
    X1 = pd.DataFrame(X,columns= dataXY.feature_names).astype({'HouseAge':np.str_}) 
    a  = featureImportance(X, Y, intervals = 30) 
    a1  = featureImportance(X1, Y, intervals = 30)   
    print(a.cal_feaimp_var_mean() - a1.cal_feaimp_var_mean())
            


