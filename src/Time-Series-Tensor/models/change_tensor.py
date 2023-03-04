
import sys
sys.path.append("../..")
#import utils._libs_
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from data_io import getGenerator
# models.SAmodel import SA_model
# models.coint import Johansen
import statsmodels.tsa.arima_model as arima
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
from args import Args, list_of_param_dicts
from model_runner import ModelRunner
import os
# -------------------------------------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_time_sequence_dict():
#获取每个仓库-产品组合的时间序列tensor
    product_sequence_dict = {}
    for file_name in os.listdir('./changeData_fenchanpin_fencang/'):
      
        with open("./changeData_fenchanpin_fencang/" + file_name,"r") as f_in:
            cid = file_name.split(".")[0].split("count-")[1]
            #print(cid)
            lines  = f_in.readlines()
            time_sequence = []
            for line in lines[1:]:
                time_sequence.append(int(line.strip("\n").split(",")[1]))
            product_sequence_dict[cid] = time_sequence[-16:]
            product_sequence_dict[cid] = product_sequence_dict[cid] + [0] * (16 - len(product_sequence_dict[cid]))

    #print(product_sequence_dict)
    return product_sequence_dict
     


class SDPANet(nn.Module):
    def __init__(self, args, data):
        """
        Initialization arguments:
            args   - (object)  parameters of model
            data   - (DataGenerator object) the data generator
        """
        #print("data",data
        super(SDPANet, self).__init__()
        self.use_cuda = args.cuda
        d_model=data.column_num
        freq='h'
        embed='fixed'
        enc_in=data.column_num
        self.input_T = args.input_T
        self.idim = data.column_num
        self.kernel_size = args.kernel_size
        self.hidC = args.hidCNN
        self.hidR = args.hidRNN
        self.batch_size=args.batch_size
        self.hw = args.highway_window
        self.main_taskpoint= args.main_taskpoint
        self.task_span = args.task_span
        self.dropout = nn.Dropout(p = args.dropout)
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        self.stride = 1
        self.channels = self.idim
        reduction_ratio = 4
        self.groups = 1
        self.group_channels = self.channels // self.groups
        self.conv1 = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels // reduction_ratio, # 通过reduction_ratio控制参数量
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=self.channels // reduction_ratio,
            out_channels=self.kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if self.stride > 1:
            # 如果步长大于1，则加入一个平均池化
            self.avgpool = nn.AvgPool2d(self.stride, self.stride)
        self.unfold = nn.Unfold(self.kernel_size, 1, (self.kernel_size-1)//2, self.stride)
        self.shared_lstm = nn.LSTM(self.hidC, self.hidR)
        self.target_lstm = nn.LSTM(self.hidC, self.hidR)
        self.coARIMA = nn.ModuleList([])
        self.linears = nn.ModuleList([])
        self.highways = nn.ModuleList([])
        self.fc=nn.Linear(self.hidR, self.idim)
        for i in range(self.main_taskpoint* 2 + 1):
            self.linears.append(nn.Linear(self.hidR, self.idim))
            if (self.hw > 0):
                self.highways.append(nn.Linear(self.hw * (i+1), 1))
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Forward propagation
    """   

    def skc(self,x):
        """
        Spatiotemporal Kernel-specific Convolution Component
        """
        
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x))) # 得到skc所需权重
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2) # 将权重reshape成 (B, Groups, 1, kernelsize*kernelsize, h, w)
   #     print(self.unfold(x).shape)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w) # 将输入reshape
        out = (weight * out).sum(dim=3).view(b, h*w,self.channels) # 求和，reshape回NCHW形式
        return out


x = torch.tensor([[1,1,1,1,3,3,3,3,75,2,2,2,2,4,8,7],[8,0,1,1,1,1,3,3,3,3,2,2,2,2,2,4],[1,1,1,1,3,3,3,3,75,2,2,2,2,4,8,7]],dtype=torch.float)



data = '../galanz/0e117c1684b5ebd6093fc17b468455d1.json'

param_dict = dict(
    data = ['./galanz/0e117c1684b5ebd6093fc17b468455d1.json'],
    main_taskpoint= [2],
    task_span = [5],
    train_share = [(0.9, 0.05)],
    input_T = [56],
    kernel_size = [3],
    hidCNN = [56],
    hidRNN = [128],
    dropout = [0.2],
    highway_window = [7],
    clip = [10.],
    epochs = [500],
    batch_size =[128],
    seed = [54321],
    gpu = [0],
    cuda = [True],
    optim = ['adam'],
    lr = [0.001],
    L1Loss = [False],
    skip_size=[32],
    dilation_cycles=[2],
    dilation_depth=[3]
)
params = list_of_param_dicts(param_dict)
for param in params:
    cur_args = Args(param)
    break


generator = getGenerator(data)

data_gen = generator(data,  train_share= cur_args.train_share, input_T=cur_args.input_T,

                             main_taskpoint=cur_args.main_taskpoint, task_span = cur_args.task_span,
                             cuda=cur_args.cuda)

runner = ModelRunner( cur_args,data_gen, None)
runner.model = SDPANet(cur_args,data_gen)

currentR = x.reshape(3,16,1,1)


currentR = runner.model.skc(currentR)
#print(currentR)


time_sequence_list = []
product_sequence_dict = get_time_sequence_dict()
for cid in product_sequence_dict:
    time_sequence_list.append(product_sequence_dict[cid])

time_sequence_list = torch.tensor(time_sequence_list,dtype=torch.float)

time_sequence_list = time_sequence_list.reshape(len(time_sequence_list),16,1,1)
time_sequence_list = runner.model.skc(time_sequence_list)


df = pd.DataFrame()
df['tensor'] = time_sequence_list.tolist()
df['cid'] = product_sequence_dict.keys()
df.to_csv("./time_tensor.csv")
