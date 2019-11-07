# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:10:20 2019

@author: Thinkpad
"""
#https://dsaber.com/2014/05/28/bayesian-regression-with-pymc-a-brief-tutorial/
import numpy as np
import pandas as pd
import pymc as pm
#import seaborn as sns
#import scipy.io
import matplotlib.pyplot as plt
#import csv
#import theano
#from numpy.random import binomial, randn, uniform
#from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from osgeo import gdal, gdal_array
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import os

station = 'Arou'
#year = '2014'

'''load in-situ data for training model'''
data_set = pd.read_csv(r'E:\Single_station\Data\Measurements\{}\Traindata_{}0.csv'.format(station,station))#Default setting is reading the header;

Et = data_set['EC_obs']
#If the name of a variable is all uppercase, python does not store it.
#If the name of avariable stars with a uppercase letter and the rest is lowercase, python stores it.(recommend this)
Lst = data_set['LST_INST_obs']
Rndaily = data_set['Rndaily_obs']
Ndvi = data_set['NDVI_rs']

data_pd = pd.concat([Et,Lst,Rndaily,Ndvi],axis=1)
data = data_pd.values
#dataframe turn into array:df = df.values;
#array turn into dataframe:df = pd.DataFrame(df)
data_x = data[:,1:4]
data_y = data[:,0]
#reshape(-1) represent the result is one row;
#reshape(-1,1) represent the result is one column;

#normalize and inverse 1:
data_x_mm = MinMaxScaler().fit_transform(data_x)
#m_data_y = MinMaxScaler().fit_transform(data_y)
#origin_data = MinMaxScaler().inverse_transform(mm_data_x or mm_data_y)
#y_new_inverse = scalery.inverse_transform(y_pred)
#https://stackoverflow.com/questions/38058774/scikit-learn-how-to-scale-back-the-y-predicted-result

#normalize and inverse 2:
#m_data_x = StandardScaler().fit_transform(data_x)
#m_data_y = StandardScaler().fit_transform(data_y)
#origin_data = StandardScaler().inverse_transform(mm_data_x or mm_data_y)
#y_new_inverse = scalery.inverse_transform(y_pred)

params = {'n_estimators': 500}

x_train,x_test,y_train,y_test = train_test_split(data_x_mm,data_y,test_size=0.2,random_state=123)

'''load remote sensing data for prediction'''
Lst_rs_dir = r'E:\Single_station\Data\Remote_sensing\{}\LST\2013141516'.format(station)
Rndaily_rs_dir = r'E:\Single_station\Data\Remote_sensing\{}\Rn\2013141516'.format(station)
Ndvi_rs_dir = r'E:\Single_station\Data\Remote_sensing\{}\NDVI\2013141516'.format(station)


def load_tif_data(directory):
    name_list = os.listdir(directory)
    #Get the filename under the current directory;
    name_list.sort()
    #sort the filename;
    path_list = []
    for name in name_list:
        path_list.append(os.path.join(directory,name))
        #os.path.join(direcory, name),connect directory and name together;
    
    image_array_list = []
    image_name_list = []
    for path_1 in path_list:
        image_name = os.path.splitext(os.path.basename(path_1))[0]
        print('load image from {}'.format(image_name))
        imagedata = pd.DataFrame(gdal_array.LoadFile(path_1))#float64,(67,34)
        imagedata_new = imagedata.interpolate()
        imagedata_new = imagedata_new.fillna(0).values
#        imagedata_new = imagedata_new.fillna(method='ffill').values
#        imagedata_new = imagedata_new.values
#        print('finsh open tif:{}'.format(image_name))
#        imagedata_new = img_to_array(imagedata)
        row, col = imagedata_new.shape[0], imagedata_new.shape[1]
#        print('finsh img_to_array:{}'.format(image_name))
        imagedata_new_1 = imagedata_new.reshape(-1,1)#float64,(67*34,1)
#        img = np.repeat(imagedata_new[:,:], 3, axis=2)#repeat three times;
#        imagedata_new = np.expand_dims(imagedata, axis=2)#add the third dimension;then finish the change from single gray image (single channel) to rgb image (three channels);
#        image = Image.fromarray(imagedata)#fromarray实现array到image的转换;
        image_array_list.append(imagedata_new_1)
        image_name_list.append(image_name)
    return np.array(image_array_list), path_list, image_name_list, row, col

Lst_rs, Lst_rs_path_list, Lst_name, row, col = load_tif_data(Lst_rs_dir)
Rndaily_rs, Rndaily_rs_path_list, Rndaily_name, row, col = load_tif_data(Rndaily_rs_dir)
Ndvi_rs, Ndvi_rs_path_list, Ndvi_name, row, col = load_tif_data(Ndvi_rs_dir)

et_name = []
method = 'RF'
for j in Lst_name:
    file_name = 'ET'+ j[9:27] + method + '.tif'
    et_name.append(file_name)

ET_name = et_name
ET_name_2013 = ET_name[0:365]
ET_name_2014 = ET_name[365:365*2]
ET_name_2015 = ET_name[365*2:365*3]
ET_name_2016 = ET_name[365*3:]

#et_std_name = []
#method = 'RF'
#for j in Lst_name:
#    file_name = 'ET'+ j[9:27] + method + '_std' + '.tif'
#    et_std_name.append(file_name)
#
#ET_std_name = et_std_name
#ET_std_name_2013 = ET_std_name[0:365]
#ET_std_name_2014 = ET_std_name[365:365*2]
#ET_std_name_2015 = ET_std_name[365*2:365*3]
#ET_std_name_2016 = ET_std_name[365*3:]

Lst_rs_2013 = pd.DataFrame(Lst_rs[0:365,:,:].reshape(-1,1))
Lst_rs_2014 = pd.DataFrame(Lst_rs[365:365*2,:,:].reshape(-1,1))
Lst_rs_2015 = pd.DataFrame(Lst_rs[365*2:365*3,:,:].reshape(-1,1))
Lst_rs_2016 = pd.DataFrame(Lst_rs[365*3:1461,:,:].reshape(-1,1))

Rndaily_rs_2013 = pd.DataFrame(Rndaily_rs[0:365,:,:].reshape(-1,1))
Rndaily_rs_2014 = pd.DataFrame(Rndaily_rs[365:365*2,:,:].reshape(-1,1))
Rndaily_rs_2015 = pd.DataFrame(Rndaily_rs[365*2:365*3,:,:].reshape(-1,1))
Rndaily_rs_2016 = pd.DataFrame(Rndaily_rs[365*3:1461,:,:].reshape(-1,1))

Ndvi_rs_2013 = pd.DataFrame(Ndvi_rs[0:365,:,:].reshape(-1,1))
Ndvi_rs_2014 = pd.DataFrame(Ndvi_rs[365:365*2,:,:].reshape(-1,1))
Ndvi_rs_2015 = pd.DataFrame(Ndvi_rs[365*2:365*3,:,:].reshape(-1,1))
Ndvi_rs_2016 = pd.DataFrame(Ndvi_rs[365*3:1461,:,:].reshape(-1,1))

data_rs_x_2013 = pd.concat([Lst_rs_2013,Rndaily_rs_2013,Ndvi_rs_2013],axis=1).values
data_rs_x_2014 = pd.concat([Lst_rs_2014,Rndaily_rs_2014,Ndvi_rs_2014],axis=1).values
data_rs_x_2015 = pd.concat([Lst_rs_2015,Rndaily_rs_2015,Ndvi_rs_2015],axis=1).values
data_rs_x_2016 = pd.concat([Lst_rs_2016,Rndaily_rs_2016,Ndvi_rs_2016],axis=1).values
#How to use pd.concat:https://blog.csdn.net/stevenkwong/article/details/52528616

data_rs_x_2013_mm = MinMaxScaler().fit_transform(data_rs_x_2013)
data_rs_x_2014_mm = MinMaxScaler().fit_transform(data_rs_x_2014)
data_rs_x_2015_mm = MinMaxScaler().fit_transform(data_rs_x_2015)
data_rs_x_2016_mm = MinMaxScaler().fit_transform(data_rs_x_2016)


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

x_train_pd = pd.DataFrame(data=x_train,columns=['LST_INST_obs','Rndaily_obs','NDVI_rs'])
x_test_pd = pd.DataFrame(data=x_test,columns=['LST_INST_obs','Rndaily_obs','NDVI_rs'])
y_train_pd = pd.DataFrame(data=y_train,columns=['EC_obs'])
y_test_pd = pd.DataFrame(data=y_test,columns=['EC_obs'])

train_data = np.concatenate((x_train_pd,y_train_pd),axis=1)
train_data_pd = pd.DataFrame(train_data,columns=['LST_INST_obs','Rndaily_obs','NDVI_rs','EC_obs'])


test_model = linear_setup(train_data_pd, ['LST_INST_obs','Rndaily_obs','NDVI_rs'], 'EC_obs')
mcmc = pm.MCMC(test_model)
mcmc.sample(200000, 20000)#从后验分布中抽取样本10万次，但扔掉前2万个样本以确保只从稳态后验分布中抽取样本。

multifig, multiax = plt.subplots(3, 1, figsize=(10, 10))
b_nought = mcmc.trace('b0')[:]#Intercept
b_lst = mcmc.trace('b1')[:]
b_rn = mcmc.trace('b2')[:]
b_ndvi = mcmc.trace('b3')[:]

multiax[0].hist(b_lst)
multiax[0].set_title('lst Coefficient Probability Distribution')
multiax[1].hist(b_rn)
multiax[1].set_title('rn Coefficient Probability Distribution')
multiax[2].hist(b_ndvi)
multiax[2].set_title('ndvi Coefficient Probability Distribution')

print('\n')
print ('Intercept: ' + str(np.mean(b_nought)))
print ('lst Coefficient: ' + str(np.mean(b_lst)))
print ('rn Coefficient: ' + str(np.mean(b_rn)))
print ('ndvi Coefficient: ' + str(np.mean(b_ndvi)))

Intercept = np.mean(b_nought)
Coefficient_LST = np.mean(b_lst)
Coefficient_Rn = np.mean(b_rn)
Coefficient_NDVI = np.mean(b_ndvi)

lst_test = x_test_pd['LST_INST_obs']
rn_test = x_test_pd['Rndaily_obs']
ndvi_test = x_test_pd['NDVI_rs']

'''Modify no.1'''
year = '2013'
data_rs_x_year_mm = eval('data_rs_x_{}_mm'.format(year))
ET_name_year = eval('ET_name_{}'.format(year))
#ET_std_name_year = eval('ET_std_name_{}'.format(year))

if year == '2016':
    d_num = 366
else:
    d_num = 365

y_pred = Intercept + Coefficient_LST*lst_test + Coefficient_Rn*rn_test + Coefficient_NDVI*ndvi_test
#prediction_inverse_scale = Denormalize(prediciton,y_m,y_s)
#prediction_result = prediction.reshape(d_num,row,col)
y_pred1 = Intercept + Coefficient_LST*data_rs_x_year_mm[:,0] + Coefficient_Rn*data_rs_x_year_mm[:,1] + Coefficient_NDVI*data_rs_x_year_mm[:,2]
y_pred1_result = y_pred1.reshape(d_num,row,col)

r, p = pearsonr(y_test.flatten(), np.array(y_pred).flatten())
r2 = r2_score(y_test.flatten(), np.array(y_pred).flatten())
MAE = mean_absolute_error(y_test.flatten(), np.array(y_pred).flatten())
MSE = mean_squared_error(y_test.flatten(), np.array(y_pred).flatten())

min_vl, max_vl = -2, 8
plt.figure()
plt.title('y_test vs y_pred')
plt.xlim(min_vl,max_vl)
plt.ylim(min_vl,max_vl)
plt.scatter(y_test,y_pred)
plt.text(0,4,'r={:.3f}'.format(r))
plt.text(0,5,'r2={:.3f}'.format(r2))
plt.text(0,6,'MAE={:.3f}'.format(MAE))
plt.text(0,7,'MSE={:.3f}'.format(MSE))
plt.plot([min_vl,max_vl],[min_vl,max_vl])
plt.show()

def array2raster(f_name, np_array, driver='GTiff',
                 prototype=None,
                 xsize=None, ysize=None,
                 transform=None, projection=None,
                 dtype=None, nodata=None):
    """
    将ndarray数组写入到文件中
    :param f_name: 文件路径
    :param np_array: ndarray数组
    :param driver: 文件格式驱动
    :param prototype: 文件原型
    :param xsize: 图像的列数
    :param ysize: 图像的行数
    :param transform: GDAL中的空间转换六参数
    :param projection: 数据的投影信息
    :param dtype: 数据存储的类型
    :param nodata: NoData元数据
    """
    # 创建要写入的数据集（这里假设只有一个波段）
    # 分两种情况：一种给定了数据原型，一种没有给定，需要手动指定Transform和Projection
    driver = gdal.GetDriverByName(driver)
    if prototype:
        dataset = driver.CreateCopy(f_name, prototype)
    else:
        if dtype is None:
            dtype = gdal.GDT_Float32
        if xsize is None:
            xsize = np_array.shape[-1]  # 数组的列数
        if ysize is None:
            ysize = np_array.shape[-2]  # 数组的行数
        dataset = driver.Create(f_name, xsize, ysize, 1, dtype)  # 这里的1指的是一个波段
        dataset.SetGeoTransform(transform)
        dataset.SetProjection(projection)
    # 将array写入文件
    dataset.GetRasterBand(1).WriteArray(np_array)
    if nodata is not None:
        dataset.GetRasterBand(1).SetNoDataValue(nodata)
    dataset.FlushCache()
    return f_name

Lst_ref = gdal.Open(Lst_rs_path_list[0])
x_size = Lst_ref.RasterXSize  # 图像列数
y_size = Lst_ref.RasterYSize  # 图像行数
proj = Lst_ref.GetProjection()  # 返回的是WKT格式的字符串
trans = Lst_ref.GetGeoTransform()  # 返回的是六个参数的tuple

out_dir = r'E:\Single_station\Data\Methods\{}\BLR_new\ET\{}'.format(station,year)+'\\'
out_dir1 = r'E:\Single_station\Data\Methods\{}\BLR_new\ET\ET_ave_{}.csv'.format(station,year)

for d in range(0,d_num):
    image_name = ET_name_year[d]
#    image_std_name = ET_std_name_year[d]
    array2raster(out_dir+image_name,y_pred1_result[d,:,:],xsize=x_size, ysize=y_size,\
                 transform=trans, projection=proj,dtype=gdal.GDT_Float32)

print('finish save ET.tif')

'''save results'''
'''Modify no.5'''
start_date = '{}0101'.format(year)
end_date = '{}1231'.format(year)

ET=[]
date_range = pd.date_range(start=start_date, end=end_date)

for i in range(0,d_num):
    et_ave1 = np.mean(y_pred1_result[i,:,:])
    ET.append(et_ave1)#最后是一个list;

ET_new = pd.DataFrame(data=ET,columns=['ET_ave'])
ET_new.index=pd.Series(date_range)
ET_new.to_csv(out_dir1,index_label='date')

print('finish save ET_ave.csv')
print('All Done:{},{}'.format(station,year))




