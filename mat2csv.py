import pandas as pd
import scipy
import numpy as np
from scipy import io
nums=[240,76,216,47,64,6]

features_struct = scipy.io.loadmat('/home/wqy/Documents/MvADL-master/uspstraining.mat')
print(features_struct.keys())
features = features_struct['TtData']
features_train = features_struct['TrData']
label_train =  features_struct['H_train']
label_test =  features_struct['H_test']
print(label_train[0][0])
print("数据类型",type(label_train[0][0]))           #打印数组数据类型
print("数组元素数据类型：",label_train[0][0].dtype) #打印数组元素数据类型
print("数组元素总数：",label_train.size)      #打印数组尺寸，即数组元素总数
print("数组形状：",label_train.shape)         #打印数组形状
print("数组的维度数目",label_train[0][0].ndim)      #打印数组的维度数目


for i in range(0,features.size):
    appended = np.append(features[0][i], features_train[0][i], axis=1)
    print(appended.shape)
    if(i==0):
        newfeature = appended
    else:
        newfeature=np.append(newfeature, appended,axis=0)
    print('newfeture',newfeature.shape)

newlabel = np.append(label_test,label_train,axis=1)
usps_data = np.append(newfeature,newlabel,axis=0)
print('newlabel',newlabel.shape)
dfdata = pd.DataFrame(usps_data)
datapath1 = '/home/wqy/Documents/MvADL-master/usps.csv'
dfdata.to_csv(datapath1, index=False)
