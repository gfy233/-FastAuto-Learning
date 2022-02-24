#encoding=utf8
#stacking 集成，包含预测窗口
from cv2 import exp
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import sys
import chardet
from sklearn.datasets import make_blobs 
import random
import os
import datetime
import json
import math
import time
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from model import *
import importlib
import scipy.io as sio
import multiprocessing
import Optim

#记录训练中被淘汰的组合
bad_searchid = []
MAE_THRESHOLD = 0.15
#输入的特征文件夹
global Input_future_folder
Input_future_folder = '/data/gfy2021/gfy/KDD/process_data/fencang_feature_selected_normal_for_1214_1227_text+pic/'

#输出的测试集文件夹
global Output_folder
Output_folder = '/data/gfy2021/gfy/KDD/stacking/stacking_result/2/'

#输出的预测结果文件夹
global Output_future_folder
Output_future_folder = '/data/gfy2021/gfy/KDD/stacking/stacking_result/future_2/'

# 模型融合中的基础模型
ModelOptions=[RandomForestRegressor(random_state=0,n_estimators=500,max_depth=7,max_features=0.8,n_jobs=-1),
        xgb.XGBRegressor(random_state=0,learning_rate=0.008, n_estimators=550, max_depth=7, min_child_weight=5, seed=1024,subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=100),
        GradientBoostingRegressor(random_state=0,loss='lad',learning_rate=0.01,n_estimators=300,subsample=0.75,max_depth=5, max_features=0.75)
        ]

NameOptions=["RF","XGB","GBRT"]
#元模型排列组合选择
MeteModelOpiton=[xgb.XGBRegressor(random_state=0,learning_rate=0.008, n_estimators=550, max_depth=7, min_child_weight=5, seed=1024,subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=100),
    GradientBoostingRegressor(random_state=0,loss='lad', learning_rate=0.01, n_estimators=300, subsample=0.75, max_depth=5, max_features=0.75),
RandomForestRegressor(random_state=0,n_estimators=1000,max_depth=7,max_features=0.8,n_jobs=-1)

]   
NameMetaModel=["xgb","GradientBoostingRegressor","RF"]


#DRANN参数
# Parameters settings
parser = argparse.ArgumentParser(
    description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")

# Dataset setting
parser.add_argument('--dataroot', type=str, default="01.json", help='path to dataset')
# parser.add_argument('--dataroot', type=str, default="test.csv", help='path to dataset')
parser.add_argument('--batchsize', type=int, default=128, help='input batch size [128]')

# Encoder / Decoder parameters setting
parser.add_argument('--nhidden_encoder', type=int, default=128,
                    help='size of hidden states for the encoder m [64, 128]')
parser.add_argument('--nhidden_decoder', type=int, default=128,
                    help='size of hidden states for the decoder p [64, 128]')
parser.add_argument('--ntimestep', type=int, default=28, help='the number of time steps in the window T [10]')

# Training parameters setting
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train [10, 200, 500]')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')

# parse the arguments
args = parser.parse_args()


''' 获取测试指标值'''
def get_mse(actual, pred):
    mse = []
    rmse=0
    mse_ave=0
    for i in range(len(actual)):
        temp=round((actual[i] - pred[i]) ** 2 , 4)
        mse.append(temp)
        mse_ave+=temp
        #rmse+=np.sqrt(temp)
    rmse=np.sqrt(mse_ave/len(actual))
    return mse,rmse,mse_ave/len(actual)

def get_mae(actual, pred):
    mae = []
    for i in range(len(actual)):
        mae.append(np.absolute(pred[i]-actual[i]))
    return mae
   
def get_R_squared(actual, pred):
  
    R_squared=(1- mean_squared_error(actual,pred)/ np.var(actual))
    return R_squared


# 元模型和基模型 组合
def PermutationAndCombination(ModelOptions,NameOptions):
    clfsLists=[]
    clf=[]
    nameLists=[]
    name=[]

    #两种模型组合
    for i in range(len(ModelOptions)):
        clf.append(ModelOptions[i])
        name.append(NameOptions[i])
        for j in range(i+1,len(ModelOptions)):
            clf.append(ModelOptions[j])
            name.append(NameOptions[j])
            clfsLists.append(clf)
            nameLists.append(name)
            clf=clf[0:-1]
            name=name[0:-1]
        clf=clf[0:-1]
        name=name[0:-1]

    #三种模型组合
    for i in range(len(ModelOptions)):
        clf.append(ModelOptions[i])
        name.append(NameOptions[i])
        for j in range(i+1,len(ModelOptions)):
            clf.append(ModelOptions[j])
            name.append(NameOptions[j])
            for r in range(j+1,len(ModelOptions)):
                clf.append(ModelOptions[r])
                name.append(NameOptions[r])
                clfsLists.append(clf)
                nameLists.append(name)
                clf=clf[0:-1]
                name=name[0:-1]
            clf=clf[0:-1]
            name=name[0:-1]
        clf=clf[0:-1]
        name=name[0:-1]


    return clfsLists,nameLists

#获取watch0 预测集，即未来14天销量
def getfuturedata(datas,data_dict):
    future_dict = []
    for d in data_dict:
        if d['watch'] == 0:
            future_dict.append(d)
    
    future = datas[datas.watch==0]
    #特征窗口截止日期
    future_end_date=future['input_end_date']
    #剔除真实值、watch编号等特征
    future_x = future.drop(['label', 'watch', 'id', 'whSapName','input_start_date','input_end_date'], axis=1)
    future_x.fillna(0, inplace=True)
    #print("finish gettestdata")
    return future_x,future_dict,future_end_date,future.id,future.whSapName


#获取训练集和测试集
def getdata(datas,data_dict,watchid,bad_id):
    train_dict, test_dict = [], []
    watchid = set(watchid)

    for d in data_dict:
        if d['watch'] == 0:
            pass
        elif d['watch'] in watchid:
            test_dict.append(d)
        #排除淘汰的id
        elif d['watch']  not in bad_id:
            train_dict.append(d)

    #获取测试集
    test = datas[datas.watch.isin(watchid)]
    test_y = test.label
    test_id=test.id
   
    test_whSapName=test.whSapName
    test_id = test.id
    test_whSapName=test.whSapName
    test_x = test.drop(['label','id','watch', 'whSapName','input_start_date','input_end_date'], axis=1)

    #去除预测集
    watchid.add(0)
   
    #获取训练集
    train = datas[~datas.watch.isin(watchid) ]
    train_y = train.label
    train_x = train.drop(['label','id','watch','whSapName','input_start_date','input_end_date'], axis=1)
    
    # 缺失值补零
    train_x.fillna(0, inplace=True)
    test_x.fillna(0, inplace=True)
    
 
    return train_x,train_y,test_x,test_y,train_dict,test_dict,test_id,test_whSapName


#获取训练集和测试集
def get_train_test(dict_file,csv_file,bad_id):
    #读入json
    data_dict = []
    with open(dict_file, 'r') as fj:
        for line in fj.readlines():
            data_dict.append(json.loads(line))

    #读入csv
    datas = pd.read_csv(csv_file)

    watchlist=[]
    for i in datas.watch:
        watchlist.append(i)
    watchlist=list(set(watchlist))

    #测试集编号
    rs=[1,2,3,4,5]
    train_x,train_y,test_x,test_y,train_dict,test_dict,test_id,test_whSapName = getdata(datas,data_dict,rs,bad_id)
    future_x,future_dict,future_end_date,future_id,future_whSapName = getfuturedata(datas,data_dict)
    print("切分数据完毕")

    return data_dict,watchlist,train_x,train_y,test_x,test_y,train_dict,test_dict,test_id,test_whSapName,future_x,future_dict,future_end_date,future_id,future_whSapName

def get_i_data(i,data_dict,datas):
    train_dict, test_dict = [], []
    train_watchid = [i+5,i+6]
    train_watchid = set(train_watchid)
    test_watchid = [1,2,3,4,5]
    test_watchid = set(test_watchid)

    #获取json格式的
    for d in data_dict:
        if d['watch'] == 0:
            pass
        elif d['watch'] in train_watchid:
            train_dict.append(d)
        elif d['watch'] in test_watchid:
            test_dict.append(d)

    #获取csv格式
    test = datas[datas.watch.isin(test_watchid)]
    test_y = test.label
    test_id=test.id
   
    test_whSapName=test.whSapName
    test_id = test.id
    test_whSapName=test.whSapName
    test_x = test.drop(['label', 'id','watch','whSapName','input_start_date','input_end_date'], axis=1)

   
    #获取训练集
    train = datas[datas.watch.isin(train_watchid) ]
    train_y = train.label
    train_x = train.drop(['label','id','watch','whSapName','input_start_date','input_end_date'], axis=1)
   
    # 缺失值补零
    train_x.fillna(0, inplace=True)
    test_x.fillna(0, inplace=True)
 
    return train_x,train_y,test_x,test_y,train_dict,test_dict

#MAE阈值函数
def F_mae(i):
  
    return math.exp(1/(i*0.1+MAE_THRESHOLD))

def judege_Mae(test_id,y_predict,y_submission,i):
    #获取当前轮次的淘汰阈值
    Threshold = F_mae(i)
    print("Threshold",Threshold)
    bad_id=[]
 
    for t in range(0,len(y_predict),5):
        y_pre = y_predict[t:t+5]
        y_sub = y_submission[t:t+5]
        i_MAE = np.mean(get_mae(y_pre,y_sub) )
       
        if(i_MAE > F_mae(i)):
            test_id = np.array(test_id)
            bad_id.append(test_id[int(t/5)])
    print("badid",bad_id)
    return bad_id
        


def pipeline(whstr):

    print("当前仓库为", whstr)
    
    '''创建训练的数据集'''
    csv_file = Input_future_folder+whstr+'.csv'
    dict_file = Input_future_folder + whstr + '.json'
    data_dict,watchlist,  train_x,train_y,test_x,test_y,train_dict,test_dict,test_id,test_whSapName,future_x,future_dict,future_end_date,future_id,future_whSapName = get_train_test(dict_file,csv_file,[])
     
    data = pd.DataFrame()
    data2 = pd.DataFrame()

    #读入csv
    datas = pd.read_csv(csv_file)

    # DRANN模型
    model = DA_RNN(len(data_dict[0]['X'][0]),
                   args.ntimestep,
                   args.nhidden_encoder,
                   args.nhidden_decoder,
                   args.batchsize,
                   args.lr,
                   args.epochs
                   )

    count =0  #记录元模型编号
    countMeta=0 # 循环元模型列表

    #依次训练各种模型组合
    #循环不同的元模型
    for q in MeteModelOpiton:
        MetaModel=q
        #排列组合基模型
        clfslist,nameList=PermutationAndCombination(ModelOptions,NameOptions)
        #循环不同基模型组合
        for t_clf in range(len(clfslist)):
            
            bad_id = []
            '''添加不同轮次--------------------------'''
            for i_turn in range(1,4):
            #主要需要将trainx_x,y train_dict, test_x, test_y 
            #获取第i轮的数据
                print("i_turn",i_turn)
                if(i_turn==3):
                    data_dict,watchlist,train_xi,train_yi,test_xi,test_yi,train_dicti,test_dicti,test_id,test_whSapName,future_x,future_dict,future_end_date,future_id,future_whSapName = get_train_test(dict_file,csv_file,bad_id)
                else:
                    train_xi,train_yi,test_xi,test_yi,train_dicti,test_dicti = get_i_data(i_turn,data_dict,datas)
                #print("t_clf",t_clf)
                clfs=clfslist[t_clf]
              
                #对每个组合进行5折交叉验证
                #训练集
                
                X = np.array(train_xi)
                y = np.array(train_yi)
                X_dict = train_dicti
              
                #测试集
                X_predict = np.array(test_xi)
                y_predict = np.array(test_yi)
                X_predict_dict = test_dicti
                
                
                 
                X_future = np.array(future_x)
                X_future_dict = future_dict
            
                dataset_blend_train = np.zeros((len(train_xi), len(clfs)+1))
                dataset_blend_test = np.zeros((len(test_xi), len(clfs)+1))
                dataset_blend_future = np.zeros((len(future_x), len(clfs)+1))
              
                '''5折stacking'''
                n_folds = 5
                #skf = list(StratifiedKFold(y, n_folds))
                kf = KFold(n_splits=5, random_state=0,shuffle=True)
              
                # 训练DARNN
                dataset_blend_test_j = np.zeros((X_predict.shape[0], n_folds))
                dataset_blend_future_j = np.zeros((X_future.shape[0], n_folds))
                i = 0
                for train, test in kf.split(X, y, X_dict):
                    train_drann_dict, test_drann_dict, y_0 = [], [], []
                    for t in train:
                        train_drann_dict.append(X_dict[t])
                    model.train(train_drann_dict)

                    # 用模型预测验证集
                    for t in test:
                        test_drann_dict.append(X_dict[t])
                    test_drann_reult = model.test(test_drann_dict)
                    output = []
                    for d in test_drann_reult:
                        output.append(d['y'])
                    dataset_blend_train[test, -1] = np.array(output)

                    # 用模型预测测试集
                    predict_drann_reult = model.test(X_predict_dict)
                    output = []
                    for d in predict_drann_reult:
                        output.append(d['y'])
                    for r in range(len(output)):
                        dataset_blend_test_j[r, i] = output[r]

                    # 用模型预测预测集
                    future_drann_reult = model.test(X_future_dict)
                    output = []
                    for d in future_drann_reult:
                        output.append(d['y'])
                    for r in range(len(output)):
                        dataset_blend_future_j[r, i] = output[r]

                    i += 1
                dataset_blend_test[:, -1] = dataset_blend_test_j.mean(1)
                dataset_blend_future[:, -1] = dataset_blend_future_j.mean(1)
                print("DRANN MSE Score: %f" % mean_squared_error(y_predict, dataset_blend_test[:, -1]), "\n")
                #print('DRANN训练完毕')

                for j, clf in enumerate(clfs):
                    '''依次训练各个单模型'''
                
                    dataset_blend_test_j = np.zeros((X_predict.shape[0], n_folds))
                    dataset_blend_future_j=np.zeros((X_future.shape[0], n_folds))
                    i = 0
                   
                    for train, test in kf.split(X, y):
                    
                        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
                        #在原训练集中划分训练集和验证集
                        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
                        clf.fit(X_train, y_train)
                        #用模型预测验证集
                        y_submission = clf.predict(X_test)
                        dataset_blend_train[test, j] = y_submission

                        #用模型预测测试集
                        temp=clf.predict(X_predict)
                        for r in range(len(temp)):
                            dataset_blend_test_j[r, i] = temp[r]
                        print("2")
                        #用模型预测预测集
                        temp=clf.predict(X_future)
                        for r in range(len(temp)):
                            dataset_blend_future_j[r, i] = temp[r]
                        i = i + 1
                    
                    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
                    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
                    dataset_blend_future[:, j] = dataset_blend_future_j.mean(1)
                            
                    print("MSE Score: %f" % mean_squared_error(y_predict, dataset_blend_test[:, j]), "\n")
            
                clf = MetaModel
                clf.fit(dataset_blend_train, y)
                #y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
                y_submission = clf.predict(dataset_blend_test)

                y_future=clf.predict(dataset_blend_future)
            
                print("总模型MSE Score: %f" % (mean_squared_error(y_predict, y_submission)))

                # 调用自写函数，计算单个数据误差指标
                #print(y_predict,y_submission)
                #print(len(y_predict),len(y_submission))
                mse_stacking,rmse,mse_ave = get_mse(y_predict, y_submission)
                mae_stacking = get_mae(y_predict, y_submission)
                wholeMae=np.sum(np.absolute(y_submission-y_predict))/len(y_predict)
                R_squared=get_R_squared(y_predict, y_submission)
                #添加被淘汰的序列
                
                bad_id += judege_Mae(test_id,y_predict,y_submission,i_turn)
                
                '''添加不同轮次--------------------------'''
            
            #将结果写入csv
            #训练与测试部分
            nameMeta=NameMetaModel[countMeta]
            test_id=np.array(test_id)
            test_whSapName=np.array(test_whSapName)
            data[str(count)+'id']=test_id
            data[str(count)+'whSapName']=test_whSapName
            data[str(count)+nameMeta]=0
            namemodel=""
            
            for name in nameList[count%4]:
                namemodel+=name+"+"
            namemodel=namemodel[0:-1]
            data[str(count)+namemodel]=0
            data[str(count)+'actual']=y_predict
            data[str(count)+'pred'] = y_submission
            data[str(count)+'mse_stacking'] = mse_stacking
            data[str(count)+'mae_stacking'] = mae_stacking
            data[str(count)+'R_squared'] = R_squared
            data[str(count)+'rmse'] = rmse
            data[str(count)+'mse_ave'] = mse_ave

            print("train data into csv")
            #预测部分
            data2[str(count)+'future_id']=future_id
            data2[str(count)+'future_whSapName']=future_whSapName

            #预测窗口的起始日期在特征窗口的后一天
            print("future_end_date!!!!!!!!!!!!!",future_end_date)

            beginDate=[]
            endDate=[]
            for li,raw_date in enumerate(future_end_date):
                print("raw_date!!!!!",raw_date)
               
                temp_date = datetime.date(int(raw_date.split("-")[0]), int(raw_date.split("-")[1]), int(raw_date.split("-")[2]))
                beginDate.append( str(temp_date + datetime.timedelta(days = 1)))
                endDate.append( str(temp_date + datetime.timedelta(days = 14)))

               
            data2[str(count)+'future_begin_date']=beginDate
            data2[str(count)+'future_end_date']=endDate
            data2[str(count)+'future']=y_future
        
            print("第"+str(count)+"个组合！！！！！")
            print("MSE：",mse_ave)
            #print(data)
            count+=1
        countMeta+=1
    # 生成结果文件
    #训练和测试结果
    csv_ = Output_folder + 'result_' + whstr + '.csv'
    #预测结果
    csv_2 = Output_future_folder + 'result_' + whstr + '.csv'

    data.to_csv(csv_, index=None)
    data2.to_csv(csv_2, index=None)
    print("csv输出完成！")
   

if __name__ == '__main__':
    
    files= os.listdir(Input_future_folder)
    for file in files: 
      
        filenames=str(file).split(".")[0]
        type = str(file).split(".")[1]
        if type == 'json':
            continue

        strs=[filenames]
        #对每个仓库建立模型
        for whstr in strs:
            
            try:
                if whstr == 'bf2b729332ba6c513c7e529a879fbb7a' or whstr =='bf2b729332ba6c513c7e529a879fbb7a' or whstr =='ae9eedb427f5b2235befd37366fc09d7' or whstr == 'a8ec907a5ac10f3960aabf52e1f65f1c':
                    pipeline(whstr)
            except Exception as e:
                print(e)
                pass
            continue
            
    print("finish!")


