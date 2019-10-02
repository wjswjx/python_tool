# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:10:20 2019

@author: Xiangli
"""
#https://dsaber.com/2014/05/28/bayesian-regression-with-pymc-a-brief-tutorial/
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
#import csv
#import theano
#from numpy.random import binomial, randn, uniform
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()


#sns.set()
#warnings.filterwarnings('ignore')


filename = 'E:/Single_station/Data/Methods/Sidaoqiao/sample_BLR.csv'
dataset = pd.read_csv(filename,header=None)
dataset.columns = ["lst", "ndvi", "rn","et"]
dataset_nonzero = dataset[np.all(dataset != 0, axis=1)]#从完全非零的所有行中选择,即删除含有0的行;

data_test = pd.read_csv('E:/Single_station/Data/Methods/Sidaoqiao/Predictors_RF_nan_0.txt', sep=" ", header=None)#BLR训练样本与RF相同；
data_test.columns = ["lst", "ndvi", "rn"]

#lst = dataset_nonzero.iloc[:,4].values.reshape(-1,1)#读取第一列(:,0)数据,自动忽略第一行标题；这种方式的提取，提取之后是float64格式，不保留DataFrame格式；可以根据数据类型自动匹配类型，比如整数则int64,小数则float64;
#rn = dataset_nonzero.iloc[:,5].values.reshape(-1,1)
#ndvi = dataset_nonzero.iloc[:,6].values.reshape(-1,1)
#et = dataset_nonzero.iloc[:,7].values.reshape(-1,1)

#Data = dataset_nonzero.ix[:,0:3]#（推荐）这种方式的提取，提取之后仍然保留DataFrame格式；
#Data = np.c_[lst,rn,ndvi,et]#四个单独的一列合成一个四列；
#x = pd.concat([LST, Rn, NDVI], axis = 1)#三个单独的一列合成一个三列；
#Data_scale1 = preprocessing.scale(Data)#对dataset的每一列进行标准化处理（均值为0，方差为1的高斯分布）；两种标准化方法的结果一样；
#Data_scale = scaler.fit_transform(Data)#对dataset的每一列进行标准化处理（均值为0，方差为1的高斯分布）；两种标准化方法的结果一样；这种方法更方便反归一化；
#Data_scale2 = pd.DataFrame({'lst':Data_scale[:,0],'rn':Data_scale[:,1],'ndvi':Data_scale[:,2],'et':Data_scale[:,3]})#对标准化后的数据变化为DataFrame格式;
# 可以查看标准化后的数据的均值与方差，已经变成0,1了
#Data_scale.mean(axis=0)
# axis=1表示对每一行去做这个操作，axis=0表示对每一列做相同的这个操作
#Data_scale.mean(axis=1)
# 同理，看一下标准差
#Data_scale.std(axis=0)

#Data.to_csv('Data1.csv',index=False,header=True)#将DataFrame保存为csv文件；index=False,则不保存index；header=False,则不保存标题;

#y = Data_scale2['et'].reshape(-1,1)
#y = Data_scale2.et

X_train_input = dataset_nonzero.iloc[:,0:3]
X_train_output = dataset_nonzero.iloc[:,3]

y_test_input = data_test

def Normalize(X, X_m, X_s):
    return (X-X_m)/(X_s)

def Denormalize(X, X_m, X_s):    
    return X_s*X + X_m

Normalize_input_data = 1
Normalize_output_data = 1
    

# Normalize Input Data
if Normalize_input_data == 1:
    X_m = np.mean(X_train_input, axis = 0)
    X_s = np.std(X_train_input, axis = 0)
    X_train_input0 = Normalize(X_train_input, X_m, X_s)
    y_test_input0 = Normalize(y_test_input, X_m, X_s)



# Normalize Output Data
if Normalize_output_data == 1:
    y_m = np.mean(X_train_output, axis = 0)
    y_s = np.std(X_train_output, axis = 0)   
    X_train_output0 = Normalize(X_train_output, y_m, y_s)


model_data = pd.concat([X_train_input0,X_train_output0], axis = 1)

#X_train1 = np.c_[LST,Rn,NDVI]
#X_train = X_train1[0:int(2/3*(len(X_train1))),:]
#X_test = X_train1[int(2/3*(len(X_train1))):len(X_train1),:]
#
#y_train1 = ET
#y_train = y_train1[0:int(2/3*(len(X_train1)))]
#y_test = y_train1[int(2/3*(len(X_train1))):len(X_train1)]

#x1 = model_data.ix[:,0]
#x2 = model_data.ix[:,1]
#x3 = model_data.ix[:,2]
#x = model_data.ix[:,0:3]
#y = model_data.ix[:,3]
#a = np.array([0,0,0]).reshape(1,3)
#sd = np.array([20,20,20]).reshape(1,3)

def linear_setup(df, ind_cols, dep_col):
    
    '''
    Inputs: pandas Data Frame, list of strings for the independent variables,
    single string for the dependent variable
    Output: PyMC Model
    '''
  
    # model our intercept and error term as above
    #  with pm.Model() as model:
    b0 = pm.Normal('b0', 0, 0.0001)
    err = pm.Uniform('err', 0, 500)
 
    # initialize a NumPy array to hold our betas
    # and our observed x values
    b = np.empty(len(ind_cols), dtype=object)
    x = np.empty(len(ind_cols), dtype=object)
 
    # loop through b, and make our ith beta
    # a normal random variable, as in the single variable case
    for i in range(len(b)):
        b[i] = pm.Normal('b' + str(i + 1), 0, 0.0001)
 
    # loop through x, and inform our model about the observed
    # x values that correspond to the ith position
    for i, col in enumerate(ind_cols):
        x[i] = pm.Normal('b' + str(i + 1), 0, 1, value=np.array(df[col]), observed=True)
 
    # as above, but use .dot() for 2D array (i.e., matrix) multiplication
    @pm.deterministic
    def y_pred(b0=b0, b=b, x=x):
        return b0 + b.dot(x)
 
    # finally, "model" our observed y values as above
    y = pm.Normal('y', y_pred, err, value=np.array(df[dep_col]), observed=True)
 
    return pm.Model([b0, pm.Container(b), err, pm.Container(x), y, y_pred])

test_model = linear_setup(model_data, ['lst','ndvi','rn'], 'et')
mcmc = pm.MCMC(test_model)
mcmc.sample(200000, 20000)#从后验分布中抽取样本10万次，但扔掉前2万个样本以确保只从稳态后验分布中抽取样本。

multifig, multiax = plt.subplots(3, 1, figsize=(10, 10))
b_nought = mcmc.trace('b0')[:]#Intercept
b_lst = mcmc.trace('b1')[:]
b_ndvi = mcmc.trace('b2')[:]
b_rn = mcmc.trace('b3')[:]
 
multiax[0].hist(b_lst)
multiax[0].set_title('lst Coefficient Probability Distribution')
multiax[1].hist(b_ndvi)
multiax[1].set_title('ndvi Coefficient Probability Distribution')
multiax[2].hist(b_rn)
multiax[2].set_title('rn Coefficient Probability Distribution')

print ('Intercept: ' + str(np.mean(b_nought)))
print ('lst Coefficient: ' + str(np.mean(b_lst)))
print ('ndvi Coefficient: ' + str(np.mean(b_ndvi)))
print ('rn Coefficient: ' + str(np.mean(b_rn)))

Intercept = np.mean(b_nought)
Coefficient_LST = np.mean(b_lst)
Coefficient_NDVI = np.mean(b_ndvi)
Coefficient_Rn = np.mean(b_rn)

#lst_test = data_test['lst']
#ndvi_test = data_test['ndvi']
#rn_test = data_test['rn']

lst_test = y_test_input0['lst']
ndvi_test = y_test_input0['ndvi']
rn_test = y_test_input0['rn']

prediciton = Intercept + Coefficient_LST*lst_test + Coefficient_NDVI*ndvi_test + Coefficient_Rn*rn_test
prediction_inverse_scale = Denormalize(prediciton,y_m,y_s)

prediction_inverse_scale.to_csv('E:/Single_station/Data/Methods/Sidaoqiao/ET_BLR_1.csv',index=False,header=False)



