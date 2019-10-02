
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:44:19 2019

@author: Thinkpad
"""

import numpy as np
import scipy.io
from sklearn import metrics
import matplotlib.pyplot as plt


data = scipy.io.loadmat('E:/GPR/software/LSTPGP/LSTData/PredictionMap_day20th_M2000_SE_Iter6000_BS2000_area2000.mat')
mean_star = data['mean_star']
#std_star = data['std_star']
data1 = scipy.io.loadmat('E:/GPR/software/LSTPGP/LSTData/TrainingData.mat')
inputdata_x = data1['Data2014_SST_Y_H']
#x_test = inputdata_x[95453:97453,0:2]
y_test = inputdata_x[95453:97453]


def Normalize(X, X_m, X_s):
    return (X-X_m)/(X_s)

x_m = np.mean(mean_star, axis = 0)
x_s = np.std(mean_star, axis = 0)
mean_star_normal = Normalize(mean_star,x_m,x_s)

y_m = np.mean(y_test, axis=0)
y_s = np.std(y_test,axis=0)
y_test_normal = Normalize(y_test,y_m,y_s)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mae = metrics.mean_absolute_error(mean_star_normal,y_test_normal)
mse = metrics.mean_squared_error(mean_star_normal,y_test_normal)
rmse = np.sqrt(metrics.mean_squared_error(mean_star_normal,y_test_normal))
R = np.corrcoef(mean_star_normal,y_test_normal,rowvar=0)[0,1]#rowvar=0,用于计算各列之间的相关系数，输出为相关系数矩阵;
mre = np.linalg.norm(mean_star_normal-y_test_normal)/np.linalg.norm(y_test_normal)*100
mape = mean_absolute_percentage_error(mean_star_normal, y_test_normal)



#MAE = np.linalg.norm(mean_star_normal-y_test_normal)
#print('Mean Absolute Error:', mae)
plt.scatter(mean_star_normal,y_test_normal)
plt.xlim(-3, 3.5)
plt.ylim(-3, 3.5)
x1 = (-3,3.5)
y1 = (-3,3.5)
plt.plot(x1,y1,color='black',markersize=0.1)
plt.gca().set_aspect('equal', adjustable='box')#xy轴长度相同；
plt.xlabel('RS_LST')
plt.ylabel('PGP_LST')
plt.title('PGP_test')
plt.text(-2.5,3,'MAE= %.2f' % (mae),fontsize=12)
plt.text(-2.5,2.75,'MRE= %.2f%%' % (mre),fontsize=12)
plt.text(-2.5,2.50,'RMSE= %.2f' % (rmse),fontsize=12)
plt.text(-2.5,2.25,'R= %.2f' % (R),fontsize=12)
plt.text(-2.5,2.0,'MAPE= %.2f%%' % (mape),fontsize=12)
#plt.text(-2,2,'RMSE= %.2f' % (rmse),fontsize=14)
#plt.show()


