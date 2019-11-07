# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:42:40 2019

@author: xiangli
@email:  lixianggiser@gmail.com

"""

import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
#from keras.datasets import boston_housing
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import regularizers  # 正则化
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from osgeo import gdal, gdal_array
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import os

#https://github.com/ChileWang0228/DeepLearningTutorial/blob/master/MLP/MLP.ipynb
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

params = {'n_estimators': 500}

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
method = 'ANN'
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

#training
# 单CPU or GPU版本，若有GPU则自动切换
model = Sequential()  # 初始化，很重要！
model.add(Dense(units = 10,# 输出大小
                activation='relu',  # 激励函数
                input_shape=(x_train.shape[1],)  # 输入大小, 也就是列的大小
                )
    )

#model.add(Dropout(0.2))  # 丢弃神经元链接概率#使用dropout效果反而不好--lx;

model.add(Dense(units = 10,
#                kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项
#                activity_regularizer=regularizers.l1(0.01),  # 施加在输出上的正则项
                activation='relu' # 激励函数
#                bias_regularizer=keras.regularizers.l1_l2(0.01)  # 施加在偏置向量上的正则项
                )
    )

model.add(Dense(units = 10,
#                kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项
#                activity_regularizer=regularizers.l1(0.01),  # 施加在输出上的正则项
                activation='relu' # 激励函数
#                bias_regularizer=keras.regularizers.l1_l2(0.01)  # 施加在偏置向量上的正则项
                )
    )
    
model.add(Dense(units = 1,   
                activation='linear'  # 线性激励函数 回归一般在输出层用这个激励函数  
               )
         )

print(model.summary())  # 打印网络层次结构

model.compile(loss='mse',  # 损失均方误差
              optimizer='adam',  # 优化器
             )

history = model.fit(x_train, y_train,
          epochs=1000,  # 迭代次数
          batch_size=300,  # 每次用来梯度下降的批处理数据大小
          verbose=0,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
          validation_data = (x_test, y_test)  # 验证集
          )

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 保存模型
model.save('model_MLP.h5')  # creates a HDF5 file 'my_model.h5'

#模型可视化 pip install pydot
plot_model(model, to_file='model_MLP.png', show_shapes=True)

# 加载模型
model = load_model('model_MLP.h5')

# 预测
y_pred = model.predict(x_test)

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

'''Modify no.1'''
year = '2016'
data_rs_x_year_mm = eval('data_rs_x_{}_mm'.format(year))
ET_name_year = eval('ET_name_{}'.format(year))
#ET_std_name_year = eval('ET_std_name_{}'.format(year))

if year == '2016':
    d_num = 366
else:
    d_num = 365

y_pred1 = model.predict(data_rs_x_year_mm)
y_pred1_result = y_pred1.reshape(d_num,row,col)

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
out_dir = r'E:\Single_station\Data\Methods\{}\ANN_new\ET\{}'.format(station,year)+'\\'
out_dir1 = r'E:\Single_station\Data\Methods\{}\ANN_new\ET\ET_ave_{}.csv'.format(station,year)
#out_dir2 = r'E:\Single_station\Data\Methods\{}\DBN_new\ET_STD\{}'.format(station,year)+'\\'
#out_dir3 = r'E:\Single_station\Data\Methods\{}\DBN_new\ET_STD\ET_std_{}.csv'.format(station,year)

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



