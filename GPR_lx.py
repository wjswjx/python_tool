# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 23:47:56 2019

@author: xiangli
@email:  lixianggiser@gmail.com

"""

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import (Matern, RBF, RationalQuadratic,
                                              WhiteKernel, ConstantKernel,ExpSineSquared,DotProduct)
import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from osgeo import gdal, gdal_array
#from pathlib import Path
#from tensorflow.keras.preprocessing.image import img_to_array
import os
#from PIL import Image
#sometimes, Image.open cannot open tiff image; Using gdal_array.LoadFile replace it.
from sklearn.metrics import (explained_variance_score, mean_absolute_error, 
                             mean_squared_error, median_absolute_error, r2_score)


#matplotlib.use('tkagg')#用不用这句话都行;

station = 'Daman'
#year = '2014'

'''load in-situ data for training model'''
data_set = pd.read_csv(r'E:\Single_station\Data\Measurements\{}\Traindata_{}0.csv'.format(station,station))#Default setting is reading the header;

Et = data_set['EC_obs']
#If the name of a variable is all uppercase, python does not store it.
#If the name of avariable stars with a uppercase letter and the rest is lowercase, python stores it.(recommend this)
Lst = data_set['LST_INST_obs']
Rndaily = data_set['Rndaily_obs']
Ndvi = data_set['NDVI_rs']

data = pd.concat([Et,Lst,Rndaily,Ndvi],axis=1).values
#dataframe turn into array:df = df.values;
#array turn into dataframe:df = pd.DataFrame(df)
data_x = data[:,1:4]
data_y = data[:,0]
#reshape(-1) represent the result is one row;
#reshape(-1,1) represent the result is one column;

#normalize and inverse 1:
data_x = MinMaxScaler().fit_transform(data_x)
#m_data_y = MinMaxScaler().fit_transform(data_y)
#origin_data = MinMaxScaler().inverse_transform(mm_data_x or mm_data_y)
#y_new_inverse = scalery.inverse_transform(y_pred)
#https://stackoverflow.com/questions/38058774/scikit-learn-how-to-scale-back-the-y-predicted-result

#normalize and inverse 2:
#m_data_x = StandardScaler().fit_transform(data_x)
#m_data_y = StandardScaler().fit_transform(data_y)
#origin_data = StandardScaler().inverse_transform(mm_data_x or mm_data_y)
#y_new_inverse = scalery.inverse_transform(y_pred)

x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2,random_state=123)

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
method = 'GPR'
for j in Lst_name:
    file_name = 'ET'+ j[9:27] + method + '.tif'
    et_name.append(file_name)

ET_name = et_name
ET_name_2013 = ET_name[0:365]
ET_name_2014 = ET_name[365:365*2]
ET_name_2015 = ET_name[365*2:365*3]
ET_name_2016 = ET_name[365*3:]

et_std_name = []

for j in Lst_name:
    file_name = 'ET'+ j[9:27] + method + '_std' + '.tif'
    et_std_name.append(file_name)

ET_std_name = et_std_name
ET_std_name_2013 = ET_std_name[0:365]
ET_std_name_2014 = ET_std_name[365:365*2]
ET_std_name_2015 = ET_std_name[365*2:365*3]
ET_std_name_2016 = ET_std_name[365*3:]

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

#lst_rs = pd.DataFrame(Lst_rs.reshape(-1,1))
#rndaily_rs = pd.DataFrame(Rndaily_rs.reshape(-1,1))
#ndvi_rs = pd.DataFrame(Ndvi_rs.reshape(-1,1))

#data_rs_x = pd.concat([lst_rs,rndaily_rs,ndvi_rs],ignore_index=True).values
##axis = 0, 横向拼接；axis=1, 纵向拼接；https://blog.csdn.net/zhongjunlang/article/details/79604499
#data_rs_x_inverse = MinMaxScaler().fit_transform(data_rs_x)
#data_rs_x_2013 = data_rs_x[0:365,:,:]
#data_rs_x_2014 = data_rs_x[365+1:365*2,:,:]
#data_rs_x_2015 = data_rs_x[365*2+1:365*3,:,:]
#data_rs_x_2016 = data_rs_x[365*3+1:len(Lst_rs_path_list),:,:]

'''Gaussian process regression'''
kernel = ConstantKernel(0.1, (0.001, 10.0)) + Matern(length_scale=1.0, nu=3/2) + WhiteKernel(noise_level=1.0)
#kernels = [ConstantKernel(0.1, (0.001, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(noise_level=1.0),
#          ConstantKernel(0.1, (0.001, 10.0)) * RationalQuadratic(length_scale=1.0, alpha=0.1) + WhiteKernel(noise_level=1.0),
#          ConstantKernel(0.1, (0.001, 10.0)) * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0)) + WhiteKernel(noise_level=1.0),
#         ConstantKernel(0.1, (0.001, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2) + WhiteKernel(noise_level=1.0),
#         ConstantKernel(0.1, (0.001, 10.0)) * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=1.5) + WhiteKernel(noise_level=1.0),
#         ConstantKernel(0.1, (0.001, 10.0)) + Matern(length_scale=1.0, nu=3/2) + WhiteKernel(noise_level=1.0)]

#https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py
min_vl, max_vl = -2, 8


#for kernel in kernels:
#    #specify Gaussian Process
#    gp = gaussian_process.GaussianProcessRegressor(kernel = kernel)#n_restarts_optimizer=50
#    
#    # Plot prior
#    plt.figure(figsize=(8, 8))
#    gp.fit(x_train,y_train)
#    y_pred, y_std = gp.predict(x_test, return_std=True)
#    r, p = pearsonr(y_test.flatten(), y_pred.flatten())
#    r2 = r2_score(y_test.flatten(), y_pred.flatten())
#    MAE = mean_absolute_error(y_test.flatten(), y_pred.flatten())
#    MSE = mean_squared_error(y_test.flatten(), y_pred.flatten())
#    plt.figure()
#    plt.title('y_test vs y_pred from \n {}'.format(kernel))
#    plt.xlim(min_vl,max_vl)
#    plt.ylim(min_vl,max_vl)
#    plt.scatter(y_test,y_pred)
#    plt.text(0,4,'r={:.3f}'.format(r))
#    plt.text(0,5,'r2={:.3f}'.format(r2))
#    plt.text(0,6,'MAE={:.3f}'.format(MAE))
#    plt.text(0,7,'MSE={:.3f}'.format(MSE))
#    plt.plot([min_vl,max_vl],[min_vl,max_vl])
#    plt.show()

gp = gaussian_process.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10)
gp.fit(x_train, y_train.ravel())

##gp.kernel_
##kernel_属性将返回用于参数化GP的内核，以及相应的最优超参数值:
y_pred, y_std = gp.predict(x_test, return_std=True)

r, p = pearsonr(y_test.flatten(), y_pred.flatten())
r2 = r2_score(y_test.flatten(), y_pred.flatten())
MAE = mean_absolute_error(y_test.flatten(), y_pred.flatten())
MSE = mean_squared_error(y_test.flatten(), y_pred.flatten())

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

year = '2016'
data_rs_x_year_mm = eval('data_rs_x_{}_mm'.format(year))
ET_name_year = eval('ET_name_{}'.format(year))
ET_std_name_year = eval('ET_std_name_{}'.format(year))

#d_num = 365
d_star1 = 50
d_star2 = 100
d_star3 = 150
d_star4 = 200
d_star5 = 250
d_star6 = 300
d_star7 = 350
if year == '2016':
    d_end = 366
else:
    d_end = 365

'''Modify no.1'''
'''Modify no.2'''
for d_num in [d_star1,d_star2,d_star3,d_star4,d_star5,d_star6,d_star7,d_end]:
    if d_num == d_star1:
        y_pred1, y_std1 = gp.predict(data_rs_x_year_mm[0:row*col*d_num,:], return_std=True)
    elif d_num == d_star2:
        y_pred2, y_std2 = gp.predict(data_rs_x_year_mm[row*col*d_star1:row*col*d_star2,:], return_std=True)
    elif d_num == d_star3:
        y_pred3, y_std3 = gp.predict(data_rs_x_year_mm[row*col*d_star2:row*col*d_star3,:], return_std=True)
    elif d_num == d_star4:
        y_pred4, y_std4 = gp.predict(data_rs_x_year_mm[row*col*d_star3:row*col*d_star4,:], return_std=True)
    elif d_num == d_star5:
        y_pred5, y_std5 = gp.predict(data_rs_x_year_mm[row*col*d_star4:row*col*d_star5,:], return_std=True)
    elif d_num == d_star6:
        y_pred6, y_std6 = gp.predict(data_rs_x_year_mm[row*col*d_star5:row*col*d_star6,:], return_std=True)
    elif d_num == d_star7:
        y_pred7, y_std7 = gp.predict(data_rs_x_year_mm[row*col*d_star6:row*col*d_star7,:], return_std=True)
    elif d_num == d_end:
        y_pred8, y_std8 = gp.predict(data_rs_x_year_mm[row*col*d_star7:row*col*d_end,:], return_std=True)

y_pred1_result = y_pred1.reshape(d_star1,row,col)
y_std1_result = y_std1.reshape(d_star1,row,col)
y_pred2_result = y_pred2.reshape(d_star1,row,col)
y_std2_result = y_std2.reshape(d_star1,row,col)
y_pred3_result = y_pred3.reshape(d_star1,row,col)
y_std3_result = y_std3.reshape(d_star1,row,col)
y_pred4_result = y_pred4.reshape(d_star1,row,col)
y_std4_result = y_std4.reshape(d_star1,row,col)
y_pred5_result = y_pred5.reshape(d_star1,row,col)
y_std5_result = y_std5.reshape(d_star1,row,col)
y_pred6_result = y_pred6.reshape(d_star1,row,col)
y_std6_result = y_std6.reshape(d_star1,row,col)
y_pred7_result = y_pred7.reshape(d_star1,row,col)
y_std7_result = y_std7.reshape(d_star1,row,col)
y_pred8_result = y_pred8.reshape(d_end-d_star7,row,col)
y_std8_result = y_std8.reshape(d_end-d_star7,row,col)

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
'''Modify no.3'''
out_dir = r'E:\Single_station\Data\Methods\{}\GPR_new\ET\{}'.format(station,year)+'\\'
out_dir1 = r'E:\Single_station\Data\Methods\{}\GPR_new\ET\ET_ave_{}.csv'.format(station,year)
out_dir2 = r'E:\Single_station\Data\Methods\{}\GPR_new\ET_STD\{}'.format(station,year)+'\\'
out_dir3 = r'E:\Single_station\Data\Methods\{}\GPR_new\ET_STD\ET_std_{}.csv'.format(station,year)


'''prediction'''
'''Modify no.4'''
for d in range(0,d_end):
    if d < d_star1:
        image_name = ET_name_year[d]
        image_std_name = ET_std_name_year[d]
        array2raster(out_dir+image_name,y_pred1_result[d,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
        array2raster(out_dir2+image_std_name,y_std1_result[d,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
    elif d >= d_star1 and d < d_star2:
        image_name = ET_name_year[d]
        image_std_name = ET_std_name_year[d]
        array2raster(out_dir+image_name,y_pred2_result[d-d_star1,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
        array2raster(out_dir2+image_std_name,y_std2_result[d-d_star1,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
    elif d >= d_star2 and d < d_star3:
        image_name = ET_name_year[d]
        image_std_name = ET_std_name_year[d]
        array2raster(out_dir+image_name,y_pred3_result[d-d_star2,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
        array2raster(out_dir2+image_std_name,y_std3_result[d-d_star2,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
    elif d >= d_star3 and d < d_star4:
        image_name = ET_name_year[d]
        image_std_name = ET_std_name_year[d]
        array2raster(out_dir+image_name,y_pred4_result[d-d_star3,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
        array2raster(out_dir2+image_std_name,y_std4_result[d-d_star3,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
    elif d >= d_star4 and d < d_star5:
        image_name = ET_name_year[d]
        image_std_name = ET_std_name_year[d]
        array2raster(out_dir+image_name,y_pred5_result[d-d_star4,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
        array2raster(out_dir2+image_std_name,y_std5_result[d-d_star4,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
    elif d >= d_star5 and d < d_star6:
        image_name = ET_name_year[d]
        image_std_name = ET_std_name_year[d]
        array2raster(out_dir+image_name,y_pred6_result[d-d_star5,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
        array2raster(out_dir2+image_std_name,y_std6_result[d-d_star5,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
    elif d >= d_star6 and d < d_star7:
        image_name = ET_name_year[d]
        image_std_name = ET_std_name_year[d]
        array2raster(out_dir+image_name,y_pred7_result[d-d_star6,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
        array2raster(out_dir2+image_std_name,y_std7_result[d-d_star6,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
    else:
        image_name = ET_name_year[d]
        image_std_name = ET_std_name_year[d]
        array2raster(out_dir+image_name,y_pred8_result[d-d_star7,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
        array2raster(out_dir2+image_std_name,y_std8_result[d-d_star7,:,:],xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,dtype=gdal.GDT_Float32)
       
print('finish save ET_ave.tif and ET_std.tif')

'''save results'''
'''Modify no.5'''
start_date = '{}0101'.format(year)
end_date = '{}1231'.format(year)

ET1=[]
ET2=[]
ET3=[]
ET4=[]
ET5=[]
ET6=[]
ET7=[]
ET8=[]

date_range = pd.date_range(start=start_date, end=end_date)

for i in range(0,d_end):
    if i >= 0 and i < d_star1:
        et_ave1 = np.mean(y_pred1_result[i,:,:])
        ET1.append(et_ave1)#最后是一个list;
    elif i >= d_star1 and i < d_star2:
        et_ave2 = np.mean(y_pred2_result[i-d_star1,:,:])
        ET2.append(et_ave2)#最后是一个list;
    elif i >= d_star2 and i < d_star3:
        et_ave3 = np.mean(y_pred3_result[i-d_star2,:,:])
        ET3.append(et_ave3)#最后是一个list;
    elif i >= d_star3 and i < d_star4:
        et_ave4 = np.mean(y_pred4_result[i-d_star3,:,:])
        ET4.append(et_ave4)#最后是一个list;
    elif i >= d_star4 and i < d_star5:
        et_ave5 = np.mean(y_pred5_result[i-d_star4,:,:])
        ET5.append(et_ave5)#最后是一个list;
    elif i >= d_star5 and i < d_star6:
        et_ave6 = np.mean(y_pred6_result[i-d_star5,:,:])
        ET6.append(et_ave6)#最后是一个list;
    elif i >= d_star6 and i < d_star7:
        et_ave7 = np.mean(y_pred7_result[i-d_star6,:,:])
        ET7.append(et_ave7)#最后是一个list;
    else:
        et_ave8 = np.mean(y_pred8_result[i-d_star7,:,:])
        ET8.append(et_ave8)#最后是一个list;

ET1_new = pd.DataFrame(data=ET1,columns=['ET_ave'])
ET2_new = pd.DataFrame(data=ET2,columns=['ET_ave'])
ET3_new = pd.DataFrame(data=ET3,columns=['ET_ave'])
ET4_new = pd.DataFrame(data=ET4,columns=['ET_ave'])
ET5_new = pd.DataFrame(data=ET5,columns=['ET_ave'])
ET6_new = pd.DataFrame(data=ET6,columns=['ET_ave'])
ET7_new = pd.DataFrame(data=ET7,columns=['ET_ave'])
ET8_new = pd.DataFrame(data=ET8,columns=['ET_ave'])

ET_new = pd.concat([pd.DataFrame(ET1_new),pd.DataFrame(ET2_new),\
                    pd.DataFrame(ET3_new),pd.DataFrame(ET4_new),\
                    pd.DataFrame(ET5_new),pd.DataFrame(ET6_new),\
                    pd.DataFrame(ET7_new),pd.DataFrame(ET8_new)],ignore_index=True)
ET_new.index=pd.Series(date_range)
ET_new.to_csv(out_dir1,index_label='date')

print('finish save ET_ave.csv')

STD1=[]
STD2=[]
STD3=[]
STD4=[]
STD5=[]
STD6=[]
STD7=[]
STD8=[]

for i in range(0,d_end):
    if i >= 0 and i < d_star1:
        std_ave1 = np.mean(y_std1_result[i,:,:])
        STD1.append(std_ave1)#最后是一个list;
    elif i >= d_star1 and i < d_star2:
        std_ave2 = np.mean(y_std2_result[i-d_star1,:,:])
        STD2.append(std_ave2)#最后是一个list;
    elif i >= d_star2 and i < d_star3:
        std_ave3 = np.mean(y_std3_result[i-d_star2,:,:])
        STD3.append(std_ave3)#最后是一个list;
    elif i >= d_star3 and i < d_star4:
        std_ave4 = np.mean(y_std4_result[i-d_star3,:,:])
        STD4.append(std_ave4)#最后是一个list;
    elif i >= d_star4 and i < d_star5:
        std_ave5 = np.mean(y_std5_result[i-d_star4,:,:])
        STD5.append(std_ave5)#最后是一个list;
    elif i >= d_star5 and i < d_star6:
        std_ave6 = np.mean(y_std6_result[i-d_star5,:,:])
        STD6.append(std_ave6)#最后是一个list;
    elif i >= d_star6 and i < d_star7:
        std_ave7 = np.mean(y_std7_result[i-d_star6,:,:])
        STD7.append(std_ave7)#最后是一个list;
    else:
        std_ave8 = np.mean(y_std8_result[i-d_star7,:,:])
        STD8.append(std_ave8)#最后是一个list;


STD1_new = pd.DataFrame(data=STD1,columns=['ET_std'])
STD2_new = pd.DataFrame(data=STD2,columns=['ET_std'])
STD3_new = pd.DataFrame(data=STD3,columns=['ET_std'])
STD4_new = pd.DataFrame(data=STD4,columns=['ET_std'])
STD5_new = pd.DataFrame(data=STD5,columns=['ET_std'])
STD6_new = pd.DataFrame(data=STD6,columns=['ET_std'])
STD7_new = pd.DataFrame(data=STD7,columns=['ET_std'])
STD8_new = pd.DataFrame(data=STD8,columns=['ET_std'])

STD_new = pd.concat([pd.DataFrame(STD1_new),pd.DataFrame(STD2_new),\
                     pd.DataFrame(STD3_new),pd.DataFrame(STD4_new),\
                     pd.DataFrame(STD5_new),pd.DataFrame(STD6_new),\
                     pd.DataFrame(STD7_new),pd.DataFrame(STD8_new)],ignore_index=True)
ET_new.index=pd.Series(date_range)
ET_new.to_csv(out_dir3,index_label='date')

print('finish save ET_std.csv')
print('All Done:{},{}'.format(station,year))
