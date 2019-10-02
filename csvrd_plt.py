# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:55:15 2019

@author: Xiangli
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
#from datetime import datetime
import pandas as pd
from pandas.plotting import register_matplotlib_converters#没什么实际作用；
register_matplotlib_converters()#没什么实际作用；


filename = 'result_arou_a1.90_b0.3.csv'
time=[]#也可不要，后面重新建了time
obs=[]
pm=[]
DA=[]
uq=[]
with open(filename,'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        time += [row[0]]
        obs += [row[1]]
        pm += [row[2]]
        DA += [row[3]]
        uq += [row[4]]     
        
del(time[0])#删去第一行
del(obs[0])#删去第一行
del(pm[0])#删去第一行
del(DA[0])#删去第一行
del(uq[0])#删去第一行
obs=map(float,obs)#这两步将一个list中的str转换为list中的float格式，方便制图；
obs=list(obs)#这两步将一个list中的str转换为list中的float格式，方便制图；
pm=map(float,pm)
pm=list(pm)
DA=map(float,DA)
DA=list(DA)
uq=map(float,uq)
uq=list(uq)

##year =''.join(time)#list转str
##year1 = datetime.strptime(year, '%Y/%m/%d')
#
a = range(0,365)#这两行重新生成datetime格式的日期；
time = pd.to_datetime(a, unit='D', origin=pd.Timestamp('2015-01-01'))
##datetime.datetime.strptime(time, '%Y-%m-%d')

uq_up = list(map(lambda x: x[0] + x[1], zip(DA, uq)))#两列list数据(float类型)相加，DS+std
uq_down = list(map(lambda x: x[0] - x[1], zip(DA, uq)))#两列list数据(float类型)相减，DS-std

##
y1 = obs
y2 = pm
y3 = DA
x1 = time
##
plt.figure(dpi = 300)#adjust resolution

line1, = plt.plot(x1,y1, c = 'red',label="observation",linewidth=0.5)#线性图
#scatter1 = plt.scatter(x1,y1, color='red', marker = 'x',label="observation",s=10)#散点图
#line2, = plt.plot(x1,y2, c = 'blue',label="prediction of model",linewidth=0.5)
line3, = plt.plot(x1,y3, c = 'black',label="data assimilation",linewidth=0.5)
#scatter2 = plt.scatter(x1,y3, color='black', marker = '+',label="data assimilation",s=10)
#plt.plot(x1,uq_up,c = 'r')
#plt.plot(x1,uq_down,c = 'b')

#plt.fill_between(x1,uq_up,uq_down,facecolor = "darkgray",label="uncertainty")
plt.fill_between(x1,uq_up,uq_down,facecolor = "darkgray",label="uncertainty")
plt.xlabel('Time')
plt.ylabel('ET(mm/day)')
plt.legend()
#a = filename
plt.savefig('figure_DA_sdq_origin_a1.90_b0.3.png')
plt.show()

