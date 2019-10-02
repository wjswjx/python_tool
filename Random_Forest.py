# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:23:33 2019

@author: Xiangli
"""

#https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
#import csv
#import theano
#from numpy.random import binomial, randn, uniform
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

scaler = preprocessing.StandardScaler()
df = pd.DataFrame()

#sns.set()
#warnings.filterwarnings('ignore')


filename = 'E:/Single_station/Data/Methods/Arou/sample_RF.csv'
dataset = pd.read_csv(filename)
dataset.columns = ["lst", "ndvi", "rn","et"]
dataset_nonzero = dataset[np.all(dataset != 0, axis=1)]#从完全非零的所有行中选择,即删除含有0的行;

data_test = pd.read_csv('E:/Single_station/Data/Methods/Arou/Predictors_RF_nan_0.txt', sep=" ", header=None)
data_test.columns = ["lst", "ndvi", "rn"]



#lst = dataset_nonzero.iloc[:,4].values.reshape(-1,1)#读取第一列(:,0)数据,自动忽略第一行标题；这种方式的提取，提取之后是float64格式，不保留DataFrame格式；可以根据数据类型自动匹配类型，比如整数则int64,小数则float64;
#rn = dataset_nonzero.iloc[:,5].values.reshape(-1,1)
#ndvi = dataset_nonzero.iloc[:,6].values.reshape(-1,1)
#et = dataset_nonzero.iloc[:,7].values.reshape(-1,1)

Data = dataset_nonzero.iloc[:,:]

def Normalize(X, X_m, X_s):
    return (X-X_m)/(X_s)

def Denormalize(X, X_m, X_s):    
    return X_s*X + X_m

Normalize_input_data = 1
Normalize_output_data = 1
    

# Normalize Input Data
if Normalize_input_data == 1:
    X_m = np.mean(Data, axis = 0)
    X_s = np.std(Data, axis = 0)
    X = Normalize(Data, X_m, X_s)



# Normalize Output Data
if Normalize_output_data == 1:
    y_m = np.mean(data_test, axis = 0)
    y_s = np.std(data_test, axis = 0)   
    y = Normalize(data_test, y_m, y_s)



#Data = np.c_[lst,rn,ndvi,et]#四个单独的一列合成一个四列；
#x = pd.concat([LST, Rn, NDVI], axis = 1)#三个单独的一列合成一个三列；
#Data_scale1 = preprocessing.scale(Data)#对dataset的每一列进行标准化处理（均值为0，方差为1的高斯分布）；两种标准化方法的结果一样；
#Data_scale = scaler.fit_transform(Data)#对dataset的每一列进行标准化处理（均值为0，方差为1的高斯分布）；两种标准化方法的结果一样；这种方法更方便反归一化；
#Data_scale2 = pd.DataFrame({'lst':X[:,0],'ndvi':X[:,1],'rn':X[:,2],'et':X[:,3]})#对标准化后的数据变化为DataFrame格式;

#data_test_scale = scaler.fit_transform(data_test)
#data_test_scale2 = pd.DataFrame({'LST':y[:,0],'NDVI':y[:,1],'Rn':y[:,2]})#对标准化后的数据变化为DataFrame格式;

# 可以查看标准化后的数据的均值与方差，已经变成0,1了
#Data_scale.mean(axis=0)
# axis=1表示对每一行去做这个操作，axis=0表示对每一列做相同的这个操作
#Data_scale.mean(axis=1)
# 同理，看一下标准差
#Data_scale.std(axis=0)

#Data.to_csv('Data1.csv',index=False,header=True)#将DataFrame保存为csv文件；index=False,则不保存index；header=False,则不保存标题;

#y = Data_scale2['et'].reshape(-1,1)
#y = Data_scale2.et
#
model_data = X

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
x1 = model_data.iloc[:,0:3]
y1 = model_data.iloc[:,3]
#a = np.array([0,0,0]).reshape(1,3)

X_train = x1
y_train = y1

X_test = data_test.iloc[:,0:3]
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = RandomForestRegressor(n_estimators=500, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)

y_pred_pd = pd.DataFrame({'ET':y_pred})#对标准化后的数据变化为DataFrame格式;


#y_pred_inverse_scale = Denormalize(y_pred_pd,x_m,x_s)

y_pred_inverse_scale = Denormalize(y_pred_pd,y_m,y_s)

y_pred_inverse_scale.to_csv('E:\Single_station\Data\Methods\Arou\ET_RF_Python.csv', encoding='utf-8', index=False)

#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print('correlation coefficient:', np.corrcoef([y_test, y_pred])[0,1])

#plt.scatter(y_test, y_pred)
