# encoding=utf8

import math
import pandas as pd
import numpy as np
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MvADL(object):

    def __init__(self):
        self.learning_step = 0.000001           # 学习速率
        self.max_iteration = 1            # 最大迭代次数
        self.weight_lambda = 0.01               # 衰退权重

    def scale_cols(self,X):
        m,n = X.shape
        max_per_col = np.max(X,axis=0)
        repeat_max = np.tile(max_per_col,(m,1))
        X = X/repeat_max
        return X

    def magni_H(self,H, factor):
        magni_H = np.repeat(H,factor, axis=0)
        print(magni_H))
        print(magni_H.shape
        return magni_H

    def optimize_R(self, R, label):
        classNum = len(R)
        T = np.zeros((classNum, 1))
        V = R + 1 - np.tile(R[label], (classNum))
        #print(V)
        step = 0
        num = 0
        for i in range(0,classNum):
            if(i!=label):
                dg = V[i]
                for j in range(0,classNum):
                    if(V[i]<V[j]):
                        dg = dg + V[i] - V[j]
                if(dg>0):
                    step = step + V[i]
                    num = num+1
        step = step / (1+num)

        for i in range(1,classNum):
            if(i==label):
                T[i] = R[i] + step
            else:
                T[i] = R[i] + np.min(step-V[i],0)

        return T

    def train(self, X, H,params):
        time = 0
        magni_H = self.magni_H(H,params['magni'])
        class_id = np.argmax(H, axis=0)
        self.k = np.max(class_id)
        #print(class_id)
        X = self.scale_cols(X)
        while time < self.max_iteration:
            print('loop %d' % time)
            time += 1

            chol_in = np.dot(X,X.T)+params['lada']*params['yita']*np.eye(X.shape[0])
            L = np.linalg.cholesky(chol_in)
            svd_in = np.dot(np.dot(np.linalg.inv(L), X),H.T)
            U, S, V = np.linalg.svd(svd_in,full_matrices=False, compute_uv=True)
            Pa = np.dot(S,S.T) + 4*params['lada']*np.eye(S.shape[0])
            Pb = S + np.sqrt(Pa)
            Pc = np.dot(np.dot(np.dot(V,Pb),U.T),np.linalg.inv(L))
            P = Pc/2
            Z_co = H
            inv_temp = np.linalg.inv(np.dot(Z_co,Z_co.T)+params['lambda']*np.eye(Z_co.shape[0]))
            W = np.dot(np.dot(H ,Z_co.T),inv_temp)
            H_temp = np.dot(W, Z_co).T
            #print('H_temp',H_temp.shape)
            for ind in range(0,X.shape[1]):
                # print('indi',ind)
                # print('H_temp[ind,:]',H_temp[ind,:])
                # print('class_id[ind]',class_id[ind])
                ind_temp = self.optimize_R(H_temp[ind,:],class_id[ind])
                H[:,ind]=ind_temp.T
            ### update Z_co
            PX = np.dot(P,X)
            Z_co_inv_temp = np.linalg.inv(np.eye(W.shape[1])+ params['alpha']* np.dot(W.T,W))
            Z_co_right_part = params['alpha']* np.dot(W.T,H)+ PX
            Z_co = np.dot(Z_co_inv_temp,Z_co_right_part)
        self.W = W
        self.P = P


    def predict(self,fea, label):
        Z_temp = np.dot(self.P,fea)
        label_pred = np.dot(self.W, Z_temp)
        class_id_true = np.argmax(label, axis=0)
        class_id_pred = np.argmax(label_pred, axis=0)
        same_ids = sum(class_id_true==class_id_pred)
        acc = same_ids/len(class_id_true)
        return acc*100


if __name__ == '__main__':

    print('Start read data')

    time_1 = time.time()
    raw_data = pd.read_csv('/home/wqy/Documents/MvADL-master/usps.csv',header=0)

    data = raw_data.values.T
    m,n= data.shape
    # imgs = data[0:m-10, 0::]
    # labels = data[m-10::, 0::]
    imgs = data[0::,0:n-10]
    labels = data[0::,n-10::]
    # print(imgs.shape)
    # print(labels.shape)
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.5, random_state=23323)
    #print('test_labels',test_labels.shape)

    time_2 = time.time()
    print('read data cost '+ str(time_2 - time_1)+' second')

    print('Start training')
    p = MvADL()

    params = dict({'alpha': 1, 'lada': 2, 'yita': 3,'lambda': 4, 'magni':5})
    p.train(train_features.T, train_labels.T,params)

    time_3 = time.time()
    print('training cost '+ str(time_3 - time_2)+' second')

    print('Start predicting')
    acc = p.predict(test_features.T,test_labels.T)
    time_4 = time.time()
    print('predicting cost ' + str(time_4 - time_3) +' second')
    print("The accruacy socre is " + str(acc)+'%')
