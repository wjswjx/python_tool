# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:57:50 2019

@author: Xiangli
"""

from pathlib import Path
from osgeo import gdal
from osgeo import gdal_array
#from PIL import Image
import numpy as np
import pandas as pd 

input_path = Path(r'E:\Data_fusion\DATA\MODIS')
image_path = input_path / '2015' / '1000m_rename'
#path_val = Path(r'E:\Data_fusion\DATA\MODIS\2010\1000m_rename_fillmissing')

path_list=[]
ldata_list=[]
mdata_list=[]
image_ref_path = Path(r'E:\Data_fusion\DATA\MODIS\2010\1000m_rename\MOD11A1_LST2010034_bm_input.tif')

#image_ref = gdal_array.LoadFile(str(image_ref_path))

#def cal_time(path_list):
#    for i in range(1,len(path_list)+1):
#        yield i
        

#https://theonegis.gitbook.io/geopy/gdal-kong-jian-shu-ju-chu-li/gdal-shu-ju-ji-ben-cao-zuo/zha-ge-shu-ju-chuang-jian-yu-bao-cun

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

for path in Path(image_path).glob('*.tif'):
    path_list.append(path)

image_ref = gdal.Open(str(image_ref_path))

x_size = image_ref.RasterXSize
y_size = image_ref.RasterYSize

proj = image_ref.GetProjection()
trans = image_ref.GetGeoTransform()


"""Landsat scale"""   
#for path in path_list:
#    image_name = path.name
#    #read img as array, method1
#    #ldata = gdal.Open(str(path))
#    #ldata_array =  ldata.ReadAsArray()
#    #read img as array, method2
#    ldata_array = gdal_array.LoadFile(str(path))
#    # two methods to test nan values. method1:
#    # if np.isnan(ldata_array).sum>0:
#    # two methods to test nan values. method2:
#    if np.where(np.isnan(ldata_array))[0].size>0:
#        #np.where(np.isnan(i))#return row and column number, start from 0.
#        print('Exist nan value in {}, Nan numbers: {}'.format(image_name,np.where(np.isnan(ldata_array))))
#        ldata_array = pd.DataFrame(ldata_array)
##        #filling with mean value of column;https://blog.csdn.net/sinat_29957455/article/details/79017363
#        ldata_array = ldata_array.fillna(ldata_array.mean())
#        ldata_array = ldata_array.values
#        array2raster(image_name, ldata_array,
#                     xsize=x_size, ysize=y_size,
#                     transform=trans, projection=proj,
#                     dtype=gdal.GDT_Float32)
#    else:
#        print('No exist nan value in {}'.format(image_name))
#        array2raster(image_name, ldata_array,
#                     xsize=x_size, ysize=y_size,
#                     transform=trans, projection=proj,
#                     dtype=gdal.GDT_Float32)
#    ldata_list.append(ldata_array)

"""MODIS sacle"""
for path in path_list:
    image_name = path.name
    #read img as array, method1
    #ldata = gdal.Open(str(path))
    #ldata_array =  ldata.ReadAsArray()
    #read img as array, method2
    ldata_array = gdal_array.LoadFile(str(path))
    ldata_array[ldata_array==0]=np.nan
    # replace 0 with nan.
#    # two methods to test nan values. method1:
#    # if np.isnan(ldata_array).sum>0:
#    # two methods to test nan values. method2:
    if np.where(np.isnan(ldata_array))[0].size>0:
        #np.where(ldata_array==0)#return row and column number of 0 value, start from 0.
        print('Exist 0 value in {}, Nan numbers: {}'.format(image_name,np.where(np.isnan(ldata_array))))
        ldata_array = pd.DataFrame(ldata_array)
#        #filling with mean value of column;https://blog.csdn.net/sinat_29957455/article/details/79017363
        ldata_array = ldata_array.fillna(ldata_array.mean())
        ldata_array = ldata_array.values
        array2raster(image_name, ldata_array,
                     xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,
                     dtype=gdal.GDT_Float32)
    else:
        print('No exist 0 value in {}'.format(image_name))
        array2raster(image_name, ldata_array,
                     xsize=x_size, ysize=y_size,
                     transform=trans, projection=proj,
                     dtype=gdal.GDT_Float32)
    ldata_list.append(ldata_array)
