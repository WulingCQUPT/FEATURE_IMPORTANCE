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
        if np.issubdtype(pd.Series(Y).dtype, np.number):
            self.Y = Y # 输出变量为回归问题，正常处理
        else:
            self.Y = pd.Series(Y).factorize()[0] # 输出变量为分类问题是就因子化处理Y           
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
    
    # calculate the index of r_i^SD
    #------------ 1 calculate feature importance based on variance-mean ------------
    def cal_feaimp_var_mean(self):
        feaimp = [np.nan] * self.numfea
        for i in range(self.numfea):
            c_mean = self.condi_mean[i]
            # calculate weighted mean
            m = np.sum(c_mean[:,0] * c_mean[:,1]) / np.sum(c_mean[:,1])
            # calculate variance
            feaimp[i] = np.sqrt(np.sum((c_mean[:,0] - m) ** 2 * c_mean[:,1]) / np.sum(c_mean[:,1]))
        return np.array(feaimp)

    # calculate the index of r_i^MAD
    #------------ 21 calculate feature importance based on MeanAD_m_mean_C_Mean ------------
    def cal_feaimp_MeanAD_m_mean_C_Mean(self):
        feaimp = [np.nan] * self.numfea
        for i in range(self.numfea):
            c_mean = self.condi_mean[i]
            # calculate weighted mean
            m = np.sum(c_mean[:,0] * c_mean[:,1]) / np.sum(c_mean[:,1])
            # calculate mean of absolute deviation
            feaimp[i] = np.sum(np.abs(c_mean[:,0] - m) * c_mean[:,1]) / np.sum(c_mean[:,1])
        return np.array(feaimp)
    # calculate the index of r_i^MAD2
    #------------ 22 calculate feature importance based on MeanAD_m_mean_C_Median ------------
    def cal_feaimp_MeanAD_m_mean_C_Median(self):
        feaimp = [np.nan] * self.numfea
        for i in range(self.numfea):
            c_median = self.condi_median[i]
            # calculate weighted mean
            m = np.sum(c_median[:,0] * c_median[:,1]) / np.sum(c_median[:,1])
            # calculate mean of absolute deviation
            feaimp[i] = np.sum(np.abs(c_median[:,0] - m) * c_median[:,1]) / np.sum(c_median[:,1])
        return np.array(feaimp)


if __name__ == '__main__': 
    from sklearn.model_selection import train_test_split
    import sklearn.datasets
    import xgboost
    import sklearn
    import sklearn.linear_model
    import sklearn.ensemble
    import sklearn.neural_network
    import sklearn.discriminant_analysis
    from sklearn.preprocessing import scale
    
    # 数据准备
    ds = sklearn.datasets.load_diabetes()
    # ds = sklearn.datasets.load_boston()
    X2 = scale(ds.data) #标准化处理 （0-1区间，或正态标准化处理）
    Y2 = ds.target
    # 下面这步对本实验不是必须的，但是机器学习建模效果的必要步骤
    X_train, X_test, Y_train, Y_test = train_test_split(X2, Y2, test_size=0.1, random_state=0)
    
    # 数据建模    
    modelNames = ['Ridge', 'KNN', 'Bayesian Ridge', 'Decision Tree', 'SVM', 'Lasso',
                      'Linear Regression', 'Neural Net', 'xgboost']
    print('------- Regression Models --------')
    regressionModels = [sklearn.linear_model.Ridge(),
                            sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform'),
                            sklearn.linear_model.BayesianRidge(),
                            sklearn.tree.DecisionTreeRegressor(max_depth=5),
                            sklearn.svm.SVR(kernel="linear"),
                            sklearn.linear_model.Lasso(),
                            sklearn.linear_model.LinearRegression(),
                            sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(10,), activation='tanh',
                                                                max_iter=2000, random_state=1),
                            xgboost.XGBRegressor()]
    
    # print('------- Classification Models --------')
    # modelNames = ["Logistic", "Nearest Neighbors", "Linear SVM", "RBF SVM",
    #               "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "QDA", "xgboost"]
    # classifiers = [
    #     sklearn.linear_model.LogisticRegression(),
    #     sklearn.neighbors.KNeighborsClassifier(3),
    #     sklearn.svm.SVC(kernel="linear", C=0.025, probability=True),
    #     sklearn.svm.SVC(gamma=2, C=1, probability=True),
    #     # sklearn.gaussian_process.GaussianProcessClassifier(1.0 * sklearn.gaussian_process.kernels.RBF(1.0)),
    #     sklearn.tree.DecisionTreeClassifier(max_depth=5),
    #     sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2),
    #     sklearn.neural_network.MLPClassifier(alpha=1, hidden_layer_sizes=[20, 5, 3], activation='tanh'),
    #     sklearn.ensemble.AdaBoostClassifier(),
    #     # sklearn.naive_bayes.GaussianNB(),
    #     sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(),
    #     xgboost.XGBClassifier()]
    
    Y_fun_reg = list()
    FI_fun_reg = list()
    for i in range(len(regressionModels)):
        clf = regressionModels[i] 
        clf.fit(X_train, Y_train) #模型的训练
        Y_fun_reg.append(clf.predict) # 保存机器学习的模型
        print(modelNames[i], " : ", clf.score(X_train, Y_train))
        
        # 计算变量重要性       
        X_random =  shuffle_data(X_train)  # 此处的X_random 必须保证随机性，不能有相关性
        # X_random = np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
        Y_random = clf.predict(X_random)
        Rs = featureImportance(X_random, Y_random, intervals = 30)
        rSD = Rs.cal_feaimp_var_mean()
        rMAD = Rs.cal_feaimp_MeanAD_m_mean_C_Mean()
        rMAD2 = Rs.cal_feaimp_MeanAD_m_mean_C_Median()
       
        FI_fun_reg.append(rSD)
        FI_fun_reg.append(rMAD)
        FI_fun_reg.append(rMAD2)
        print(ds.feature_names)
        print(FI_fun_reg)
    
    FI_fun_reg_Matrix = pd.DataFrame(np.array(FI_fun_reg),
                                     columns= ds.feature_names,
                                     index = np.repeat([['SD','MAD','MAD2']],i+1,axis=0).flatten())

        
        
        
        

        
            


