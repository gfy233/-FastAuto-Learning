# -*- coding: utf-8 -*-
"""

DARNN 模型，提供DA_RNN函数
Created on Sat Oct 24 18:51:13 2020

@author: Administrator
"""


import torch
import numpy as np

from torch import nn
from torch import optim

from torch.autograd import Variable
import torch.nn.functional as F

# DARNN 编码器
class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # LSTM作为编码器
        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,          #输入数据的特征维数，通常就是embedding_dim(词向量的维度)
            hidden_size=self.encoder_num_hidden, #LSTM中隐层的维度
            num_layers=1                         #循环神经网络的层数
                   
        )

        # 通过确定的注意力模型，构建输入注意机制
        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        # nn.Linear 用于设置网络中的全连接层
        self.encoder_attn = nn.Linear(
            #in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
            #out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]
            #相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量
            in_features=2 * self.encoder_num_hidden + self.T,
            out_features=1

           )
    
    def forward(self, X):
        """forward.

        Args:
            X: 
            X_tilde: 新输入Xt的列表

        """
        # Variable 变量类型，封装了Tensor，并整合了反向传播
        #Varibale包含三个属性：
        #data：存储了Tensor，是本体的数据
        #grad：保存了data的梯度，本事是个Variable而非Tensor，与data形状一致
        #grad_fn：指向Function对象，用于反向传播的梯度计算之用
        X_tilde = Variable(X.data.new(
            X.size(0), self.T, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # h_n, s_n: initial states with dimention hidden_size
        h_n = self._init_states(X)  #hidden state 
        s_n = self._init_states(X)  #cell state
        
        #T为特征窗口时间序列的长度 （28）
        for t in range(self.T):
            # batch_size * input_size * (2 * hidden_size + T - 1)
            # torch.cat 将两个张量（tensor）拼接在一起
            #repeat :沿着指定的维度重复tensor  (5,5,5) -> (5*input_size,5,5)
            #permute： 矩阵转置
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            '''  第一阶段注意力机制  '''
            # input attention layer
            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T))

            #softmax 转为[0-1]区间，且总和为1 
            # 通过softmax函数获取权重    alpha 为driving series 的注意力权重
            alpha = F.softmax(x.view(-1, self.input_size), dim=1)

            #torch.mul(a, b)是矩阵a和b对应位相乘
            # 获取LSTM的新输入    Xt
            # Xt = α 哈达玛积 
            x_tilde = torch.mul(alpha, X[:, t, :])
          
            '''以上为第一阶段input attention machnism,获得各driving series的注意力权重
            得到更新的输入xt
            '''
          
            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            
            #是重置参数的数据指针,为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            #unsqueeze(0) ：在第0个维度上增加一个维度
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0] #hidden state 
            s_n = final_state[1] #cell state

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n  #hidden state

        return X_tilde, X_encoded
    
    #初始化编码器的 0 hidden states 和cell states
    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        # Sequential 有序的容器
        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden +
                      encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,   #输入数据的特征维数，通常就是embedding_dim(词向量的维度)
            hidden_size=decoder_num_hidden #LSTM中隐层的维度
        
        )
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)
        self.fc.weight.data.normal_()

    def forward(self, X_encoded, y_prev):
        """forward."""
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        for t in range(self.T):

            x = torch.cat((d_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)

            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T), dim=1)

            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]
            if t < self.T:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))

                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))

                d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden

        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

        return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())


class DA_RNN(nn.Module):
    """Dual-Stage Attention-Based Recurrent Neural Network."""

    def __init__(self, m,
                 T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 parallel=False):
        #m 同仓库的商品id数量，即作为特征的时间序列数量
        #T 窗口T的时间步数量  
        #encoder_num_hidden 编码器的hidden state size
        #decoder_num_hidden 解码器的hidden state size
        """initialization."""
        super(DA_RNN, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T
        self.input_size = m
   
        #orch.device代表将torch.Tensor分配到的设备的对象。
        #torch.device包含一个设备类型（‘cpu’或‘cuda’）和可选的设备序号
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        #input_size为同仓库商品数，即driving series的数量
        self.Encoder = Encoder(input_size=m,
                               encoder_num_hidden=encoder_num_hidden,
                               T=T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T).to(self.device)

        # MSELoss均方损失函数  L1Loss : 预测值和真实值的绝对误差的平均数
        self.criterion = nn.MSELoss()
        self.MAE=nn.L1Loss()  
        if self.parallel:
            #在多个GPU上并行运行
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        #torch.optim是一个实现了多种优化算法的包
        # optimizer 优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数
        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)

        # Training set
        #self.train_timesteps = int(self.X.shape[0] * 0.7) #训练步数，这里用不上
        #self.y = self.y - np.mean(self.y[:self.train_timesteps]) #减去训练部分的y平均值？
        #self.input_size = self.X.shape[1]

    def train(self, train_dict):
        """Training process."""
        #每次迭代的mse和mae
        self.epoch_losses = np.zeros(self.epochs)
        self.epoch_mae = np.zeros(self.epochs)

        n_iter = 0
        #print("self.epochs",self.epochs)
        #迭代
        for epoch in range(self.epochs):
            idx = 0
            while (idx < len(train_dict)):
                one_batch_X, one_batch_x, one_batch_y = [], [], []

                #喂入每一批batch
                bat_i = 0
                while bat_i < self.batch_size and (idx + bat_i) < len(train_dict):
                    one_batch_X.append(train_dict[idx + bat_i]['X'])  #同仓库其他商品的时序
                    one_batch_x.append(train_dict[idx + bat_i]['x'])  #当前组合特征窗口28天销量列表
                    one_batch_y.append(train_dict[idx + bat_i]['y'])  #真实值
                    bat_i += 1
                x = np.array(one_batch_X)        #同仓库其他商品的时序
                y_prev = np.array(one_batch_x)   #特征窗口28天时间序列
                y_gt = np.array(one_batch_y)     #真实值

                loss,mae = self.train_forward(x, y_prev, y_gt)  
                self.epoch_losses[epoch] += loss
                self.epoch_mae[epoch] += mae
                idx += self.batch_size           #下一批batch 的起始下标
                n_iter += 1
                #每迭代10000次，学习率下降
                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

            #print("Epochs: ", epoch, " Loss: ", self.epoch_losses[epoch], "mae:", self.epoch_mae[epoch])

    def train_forward(self, X, y_prev, y_gt):
        """Forward pass."""
        # 每一次batch需要把梯度置零，即把loss关于weight的导数变成0
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Torch.from_numpy() 将数组转为张量，且共享内存
        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
        y_pred = self.Decoder(input_encoded, Variable(
            torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))

        y_true = Variable(torch.from_numpy(
            y_gt).type(torch.FloatTensor).to(self.device))

        #reshape 成1列，-1表示 行不确定，会根据数据变化
        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)   #MSELose
        mae=self.MAE(y_pred, y_true)    
        loss.backward()                         #反向传播，计算当前梯度

        self.encoder_optimizer.step()           #根据梯度更新网络参数
        self.decoder_optimizer.step()
      
        return loss.item(),mae.item()           #得到一个元素张量里的元素值

    def test(self, test_dict):
        """Prediction."""

        one_batch_X, one_batch_x = [], []
        for i in range(len(test_dict)):
            one_batch_X.append(test_dict[i]['X'])    #同仓库其他商品的时间序列
            one_batch_x.append(test_dict[i]['x'])    #当前组合的特征窗口时间序列
        x = np.array(one_batch_X)                    #同仓库其他商品的时间序列
        y_prev = np.array(one_batch_x)               #当前组合的特征窗口时间序列
        #特征窗口28天数据
        y_history = Variable(torch.from_numpy(
            y_prev).type(torch.FloatTensor).to(self.device))
        
        #编码，传入同仓库其他时间序列
        _, input_encoded = self.Encoder(
            Variable(torch.from_numpy(x).type(torch.FloatTensor).to(self.device)))
       
        #解码，传入 input_encoded，特征窗口28天数据，返回预测值
        y_pred = self.Decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
        for i in range(len(test_dict)):
            test_dict[i]['y'] = y_pred[i]

        return test_dict


'''
Json数据格式

{
	"watch": 7,
	"id": 1994,
	"whSapCode": "3010",
	"whSapName": "0af607c2fa9f1ba55a2f31b9f4799d25",
    X： 28行，同仓库len(idlist)列，attention机制用于X中筛选有效的时间序列
	"X": [
		[0.0, 157.0, 6.0, 0.0, 0.0, 7.0, 0.0, 13.0, 3.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0],
		[0.0, 136.0, 15.0, 0.0, 0.0, 3.0, 0.0, 22.0, 2.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0],
		[0.0, 173.0, 8.0, 0.0, 0.0, 4.0, 0.0, 9.0, 5.0, 0.0, 35.0, 0.0, 0.0, 0.0, 1.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0],
		[0.0, 161.0, 8.0, 0.0, 0.0, 9.0, 0.0, 12.0, 8.0, 0.0, 23.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0],
		[0.0, 135.0, 7.0, 0.0, 0.0, 11.0, 0.0, 13.0, 10.0, 0.0, 28.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0],
		[0.0, 72.0, 6.0, 0.0, 0.0, 12.0, 0.0, 26.0, 7.0, 0.0, 27.0, 0.0, 0.0, 1.0, 2.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
		[0.0, 9.0, 9.0, 0.0, 0.0, 10.0, 0.0, 61.0, 3.0, 0.0, 34.0, 0.0, 0.0, 0.0, 1.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
		[0.0, 1.0, 10.0, 0.0, 0.0, 10.0, 0.0, 25.0, 0.0, 0.0, 48.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
		[0.0, 5.0, 10.0, 0.0, 0.0, 14.0, 0.0, 11.0, 0.0, 0.0, 36.0, 0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 9.0, 14.0, 0.0, 0.0, 9.0, 0.0, 7.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 1.0, 23.0, 0.0, 0.0, 18.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 2.0, 17.0, 0.0, 0.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[4.0, 82.0, 5.0, 0.0, 0.0, 18.0, 0.0, 13.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[15.0, 210.0, 14.0, 0.0, 0.0, 9.0, 0.0, 26.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 23.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[11.0, 156.0, 17.0, 0.0, 0.0, 1.0, 0.0, 21.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 46.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[10.0, 147.0, 17.0, 0.0, 0.0, 1.0, 0.0, 45.0, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 76.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[11.0, 172.0, 15.0, 0.0, 0.0, 0.0, 0.0, 42.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[22.0, 137.0, 17.0, 0.0, 0.0, 1.0, 0.0, 4.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[8.0, 140.0, 15.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[13.0, 155.0, 22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[15.0, 189.0, 24.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[12.0, 186.0, 15.0, 0.0, 0.0, 0.0, 0.0, 1.0, 9.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[9.0, 129.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
		[5.0, 147.0, 20.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
		[18.0, 190.0, 22.0, 0.0, 0.0, 6.0, 0.0, 14.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 23.0],
		[10.0, 154.0, 19.0, 0.0, 0.0, 3.0, 0.0, 19.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0],
		[12.0, 142.0, 11.0, 0.0, 0.0, 4.0, 0.0, 26.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0],
		[14.0, 153.0, 16.0, 0.0, 0.0, 8.0, 0.0, 21.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
	],
	"x": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	"y": 0
}
'''
