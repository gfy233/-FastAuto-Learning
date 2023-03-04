"""
    Define all models.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from SAmodel import SA_model
from coint import Johansen
import statsmodels.tsa.arima_model as arima
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
# -------------------------------------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    def coint(self,x):
        x_idim=[]
        cox=[]
        for i in range(x.shape[0]): 
            cox.append([])
            x_idim.append([])
            j=12
            k=0
            while j<x.shape[2]:
                temp=Johansen(x[i][:,j-12:j])
                if temp.johansen()!=None:
                    x_idim[i].append([i for i in range(j-12,j)])
                    nx=x[i,:,x_idim[i][k]].cpu().numpy()
                    x_diff=np.diff(nx, axis=0)
                    x_diff_lags = lagmat(x_diff, 1, trim='both')
                    cox[i].append(x_diff_lags)
                    j=j+12
                    if k==4:
                        break
                    k=k+1
                else:
                    j=j+1
        return cox,x_idim

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
    def forward(self, x):
        """
        Arguments:
            x   - (torch.tensor) input data
        Returns:
            res - (torch.tensor) result of prediction
        """
        regressors = []
        cox,x_idim=self.coint(x)
        cohh=[]
        result=[]
        tst=[]
        temp=0
        for i in range(x.shape[0]):
            cohh.append([])
            result.append([])
            tst.append([])
            coxx=np.array(cox[i])
            for j in range(coxx.shape[0]):
                cohh[i].append([])
                temp=temp+1
                www=coxx[j]
                coARIMA=VAR(www)
                model_fit=coARIMA.fit()
                prediction = model_fit.forecast(www,steps=1)
                pre=torch.from_numpy(prediction).to(device)
                pre1=pre.float()
                tst[i].append(pre1)
        SA_ouput=SA_model(x,self.input_T,self.hidR,self.hidR)
        hidden=SA_ouput.train_forward()
        for i in range(self.task_span):
           
            currentR=x.reshape(x.shape[0],7,8,self.channels)
            currentR=currentR.permute(0,3,1,2)
            currentR = self.skc(currentR)
            #print("currentR",currentR)
            currentR = F.leaky_relu(currentR, negative_slope=0.01)
            regressors.append(currentR)
            currentR = self.dropout(currentR)
            if i < self.task_span - 1:
                currentR = currentR.permute(0,2,1).contiguous()
                currentR = torch.unsqueeze(currentR, 1)
        shared_lstm_results = []
        target_R = None
        target_h = None
        target_c = None
        self.shared_lstm.flatten_parameters()
        for i in range(self.task_span):
            cur_R = regressors[i].permute(2,0,1).contiguous()
            _, (cur_result, cur_state) = self.shared_lstm(cur_R)
            if i == self.main_taskpoint:
                target_R = cur_R
                hidden=hidden.unsqueeze(0)
                cur_result=cur_result
                target_h = cur_result+hidden
                #print(target_h.shape)
                target_c = cur_state
            cur_result = self.dropout(torch.squeeze(cur_result, 0))
            shared_lstm_results.append(cur_result)

        self.target_lstm.flatten_parameters()
        _, (target_result, _) = self.target_lstm(target_R, (target_h, target_c))
        target_result = self.dropout(torch.squeeze(target_result, 0))
        res = None
        for i in range(self.task_span):
            if i == self.main_taskpoint:
                cur_res = self.linears[i](target_result)
                for p in range(x.shape[0]):
                    #print(tst[p])
                    #print(type(tst[p]))
                    #for j in range(np.array(tst[p]).shape[0]):
                    for j in range(np.array([item.cpu().detach().numpy() for item in tst[p]]).shape[0]):
                    #for j in range(torch.stack(tst[p], dim=0).cpu().numpy().shape[0]):
                        temm=tst[p][j].squeeze()
                        cur_res[p,x_idim[p][j]]=cur_res[p,x_idim[p][j]]+temm
            else:
                cur_res = self.linears[i](shared_lstm_results[i])
            cur_res = torch.unsqueeze(cur_res, 1)
            if res is not None:
                res = torch.cat((res, cur_res), 1)
            else:
                res = cur_res

        #highway
        if (self.hw > 0):
            highway = None
            for i in range(self.main_taskpoint* 2 + 1):
                z = x[:, -(self.hw * (i+1)):, :]
                z = z.permute(0,2,1).contiguous().view(-1, self.hw * (i+1))
                z = self.highways[i](z)
                z = z.view(-1, self.idim)
                z = torch.unsqueeze(z, 1)
                if highway is not None:
                    highway = torch.cat((highway, z), 1)
                else:
                    highway = z
            res = res + highway

        return res
