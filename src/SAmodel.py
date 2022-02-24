import torch
import numpy as np
import csv
from torch import nn
from torch import optim
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
class SA(nn.Module):
    def __init__(self, T,
                 input_size,
                 SA_num_hidden,
                 parallel=False):
        super(SA, self).__init__()
        self.SA_num_hidden = SA_num_hidden
       # print(SA_num_hidden)
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # Fig 1. Temporal Attention Mechanism: SA is LSTM
        self.SA_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.SA_num_hidden,
            num_layers=1
        )

        # Construct Input Attention Mechanism via deterministic attention model
        self.SA_attn = nn.Linear(
            in_features=2 * self.SA_num_hidden + self.T ,
            out_features=1
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """
        X_tilde = Variable(X.data.new(
            X.size(0), self.T , self.input_size).zero_())
        
        X_encoded = Variable(X.data.new(
            X.size(0), self.T , self.SA_num_hidden).zero_())
        h_n = self._init_states(X)
        s_n = self._init_states(X)
        for t in range(self.T):
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0,2,1)), dim=2)
            x = self.SA_attn(
                x.view(-1, self.SA_num_hidden * 2 + self.T ))

            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size), dim=1)
            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])
            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.SA_lstm.flatten_parameters()

            # SA LSTM
            _, final_state = self.SA_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n
        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for SA."""
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.SA_num_hidden).zero_())


class TA(nn.Module):
    def __init__(self, T, TA_num_hidden, SA_num_hidden,X):
        super(TA, self).__init__()
        self.TA_num_hidden = TA_num_hidden
        self.SA_num_hidden = SA_num_hidden
        self.T = T
        self.dim=X.shape[2]
        self.attn_layer = nn.Sequential(
            nn.Linear(TA_num_hidden +self.dim,SA_num_hidden
                      ),
            nn.Tanh(),
            nn.Linear(SA_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=TA_num_hidden
        )
        self.fc = nn.Linear(SA_num_hidden + 1, 1)
        self.fc_final = nn.Linear(TA_num_hidden + SA_num_hidden, 1)

        self.fc.weight.data.normal_()

    def forward(self, X_encoded,X):
        """forward."""
        for t in range(self.T ):
            x = torch.cat((X,X_encoded), dim=2)
            beta = F.softmax(self.attn_layer(
                x.view(-1,  self.TA_num_hidden + X.shape[2])).view(-1, self.T ), dim=1)
            # batch_size * SA_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]
        return context

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for SA."""
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        return Variable(X.data.new(1, X.size(0), self.TA_num_hidden).zero_())


class SA_model(nn.Module):
    def __init__(self, X,T,
                 SA_num_hidden,
                 TA_num_hidden,
                 ):
        """initialization."""
        super(SA_model, self).__init__()
        self.SA_num_hidden = SA_num_hidden
        self.TA_num_hidden = TA_num_hidden
        self.X = X
        self.learning_rate=0.001
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.SA = SA(input_size=X.shape[2],
                               SA_num_hidden=SA_num_hidden,
                               T=T).to(self.device)
        self.TA = TA(SA_num_hidden=SA_num_hidden,
                               TA_num_hidden=TA_num_hidden,
                               T=T,X=X).to(self.device)
        self.SA_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.SA.parameters()),
                                            lr=self.learning_rate)
        self.TA_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.TA.parameters()),
                                            lr=self.learning_rate)
        self.dim=X.shape[2]

    def train_forward(self):
        """Forward pass."""
        input_weighted, input_encoded = self.SA(self.X)
        hidden= self.TA(input_encoded,self.X)
        self.SA_optimizer.step()
        self.TA_optimizer.step()
        return hidden

