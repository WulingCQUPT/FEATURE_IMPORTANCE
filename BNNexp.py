# -*- coding: utf-8 -*-

"""
Created on Fri May 10 18:34:45 2024

#贝叶斯神经网络(BNN) https://zhuanlan.zhihu.com/p/463674837
# 贝叶斯网络深度学习建模 https://github.com/Fucheng-Zhong/Bayesian_network_uncertainty/blob/master/example4.ipynb

@author: ling
"""

from Featureimportance import *
import tensorflow as tf # conda install tensorflow 或者 conda update tensorflow
import tensorflow_probability as tfp # conda install tensorflow-probability 或者 conda update tensorflow-probability
# from keras.utils.vis_utils import plot_model # 生成模型图到pdf
import os
from keras.callbacks import ModelCheckpoint # 回调（Callback）函数，它允许你在训练过程中保存模型的权重或者整个模型。这非常有用，特别是当你训练时间长、模型大或训练过程中需要保存最佳模型以供后续使用时。
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 定义贝叶斯神经网络模型，这里是4层全连接的结构，权重为高斯正态，输出也为高斯分布。
tfd = tfp.distributions # 相当于import tensorflow_probability.distributions as tfd
def Bayesian_DNN():
    # 定义 KL 散度函数
    KL = (lambda q, p, _: tfd.kl_divergence(q, p)/len(x_train))
    # 输出服从正态分布，softplus 系数用于确保标准差始终为正值
    # t 应该是一个形状为 [batch_size, 2] 的张量，其中每一行包含两个值：第一个值用于定位（loc），即正态分布的均值；第二个值经过缩放和 Softplus 激活函数处理后用于确定正态分布的标准差（scale）
    normal_sp = lambda t: tfd.Normal(loc=t[...,0],
                                     scale=1e-3+tf.math.softplus(1e-3*t[...,1]))
    # scale=1e-3+tf.math.softplus(1e-3*t[...,1])：这里首先从 t 中提取每一行的第二个值，然后将其乘以 1e-3（即 0.001）。接着，对这个乘积结果应用 tf.math.softplus 函数，这是一个平滑的激活函数，它可以将任何数值映射到 (0, +∞) 区间。Softplus 函数的输出再加上一个很小的正数 1e-3，用来确保 scale 为正值（正态分布的标准差不能为负或零）。这样做可以防止分布的标准差为零，从而避免数值问题。
    Inpt = tf.keras.layers.Input(shape=(13,))
    x = tfp.layers.DenseFlipout(32,activation='relu',
                                kernel_divergence_fn=KL,
                                bias_divergence_fn=KL)(Inpt)
    x = tfp.layers.DenseFlipout(128,activation='relu',
                                kernel_divergence_fn=KL,
                                bias_divergence_fn=KL)(x)
    x = tfp.layers.DenseFlipout(128,activation='relu',kernel_divergence_fn=KL,bias_divergence_fn=KL)(x)
    x = tfp.layers.DenseFlipout(32,activation='relu',kernel_divergence_fn=KL,bias_divergence_fn=KL)(x)
    x = tfp.layers.DenseFlipout(2,activation=None,
                                kernel_divergence_fn=KL,
                                bias_divergence_fn=KL)(x)
    dist = tfp.layers.DistributionLambda(normal_sp)(x)
    model = tf.keras.Model(inputs=Inpt, outputs=dist,name='BNN_REG')
    return model

# 提取波士顿房价的数据并建模BNN
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape,y_train.shape)
model = Bayesian_DNN() # 实例化贝叶斯神经网络模型，使用negloglik作为loss，在Flipout层中KL散度会自动加上
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) # Adam优化器，学习率0.001
negloglik = lambda y, rv_y: -rv_y.log_prob(y) # 为了使这个 lambda 函数工作，rv_y 必须是一个具有 log_prob 方法的对象，比如 tensorflow_probability 库中的 Distribution 对象。
# The Keras API will then automatically add the Kullback-Leibler divergence (contained on the individual layers of the model), to the negloglik, effectively calcuating the (negated) Evidence Lower Bound Loss (ELBO)
model.compile(optimizer,loss=negloglik,metrics='mae') # 编译模型


# 如果存在已经训练好的模型，则加载并继续训练
if os.path.exists(model.name+'.h5'):
    model.load_weights(model.name+'.h5')
else:
    epoch, batch_size = 1000, 128 # epoch被设置为 1000，意味着模型将完整遍历训练数据集 1000 次。batch_size 被设置为 128，意味着每次模型更新之前，将使用 128 个样本来进行梯度计算和参数更新。
    #  ModelCheckpoint 回调的实例
    checkpoint = ModelCheckpoint(model.name+'.h5', 
                                 monitor='mae',
                                 save_weights_only=True,
                                 verbose=1,
                                 save_best_only=True)# 仅保存最好的模型
    # 开始训练，有模型保存BNN_REG.h5可以不用训练
    history = model.fit(x_train,y_train, 
                        epochs=epoch,batch_size=batch_size, 
                        validation_data=(x_test,y_test),
                        callbacks=[checkpoint],shuffle=False)
    # 访问训练过程中的损失值和准确度
    traininghist = history.history
    print(pd.DataFrame(traininghist).tail(10))



# 对贝叶斯神经网络模型进行预测
# 数据建模    
modelNames = ['BNN']
print('------- Regression Models --------')
FI_fun_reg = list() # 特征重要性指标
   
# 计算变量重要性       
X_random =  shuffle_data(x_train,sample_size=1000)  # 此处的X_random 必须保证随机性，不能有相关性
# X_random = np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
Y_random = np.squeeze(model(X_random).mean()) # 贝叶斯模型的预测通常涉及从后验分布中采样，故其预测值为其均值
Rs = featureImportance(X_random, Y_random, intervals = 30)
rSD = Rs.cal_feaimp_var_mean()
rMAD = Rs.cal_feaimp_MeanAD_m_mean_C_Mean()
rMAD2 = Rs.cal_feaimp_MeanAD_m_mean_C_Median()
FI_fun_reg.append(rSD)
FI_fun_reg.append(rMAD)
FI_fun_reg.append(rMAD2) 
# 波士顿房价数据集的特征名称列表
boston_features = [
    'CRIM',      # 犯罪率
    'ZN',        # 住宅用地比例
    'INDUS',     # 非零售业务比例
    'CHAS',      # 查尔斯河虚拟变量
    'NOX',       # 一氧化氮浓度
    'RM',        # 平均房屋房间数
    'AGE',       # 房屋建成年份
    'DIS',       # 到波士顿五个就业中心的加权距离
    'RAD',       # 高速公路访问指数
    'TAX',       # 全职警察比例
    'PTRATIO',   # 学生与教师比例
    'B',         # 黑人人口比例
    'LSTAT'     # 低收入人群比例
]

FI_fun_reg_Matrix = pd.DataFrame(np.array(FI_fun_reg),
                                 columns= boston_features,
                                 index = np.repeat([['SD','MAD','MAD2']],1,axis=0).flatten())
print(FI_fun_reg_Matrix.head())
