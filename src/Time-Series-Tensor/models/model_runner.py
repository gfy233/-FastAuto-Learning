"""
    Define ModelRunner class to run the model.
"""
import csv
import math
import time
import torch 
import numpy as np
import torch.nn as nn
from data_io import DataGenerator
from optimize import Optimize
import numpy as np
import pandas as pd
import soft_dtw
import path_soft_dtw
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ModelRunner():
    def __init__(self, args, data_gen, model):
        """
        Initialization arguments:
            args       - (object)                 parameters of model
            data_gen   - (DataGenerator object)   the data generator
            model      - (torch.nn.Module object) the model to be run
        """
        self.args = args
        self.data_gen = data_gen
        self.model = model
        self.best_rmse = None
        self.best_rse = None
        self.best_mae = None
        self.running_times = []
        self.train_losses = []
        self.predictlast=[]
        self.train_mae=[]
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def huber(true, pred, delta):
        loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
        return np.sum(loss)
    """
    Train the model
    """
    def msta_loss(self,outputs, targets, alpha, gamma, device):
        # outputs, targets: shape (batch_size, N_output, idim)
        batch_size, N_output = outputs.shape[0:2]
#        print(outputs,targets)
        idim=outputs.shape[2]#循环得到每一个商品的loss
        loss_shape = 0
        loss=0
        loss_temporal=0
        softdtw_batch = soft_dtw.SoftDTWBatch.apply
        for i in range(idim):
            D = torch.zeros((batch_size, N_output,N_output )).to(device)
            for k in range(batch_size):
                Dk = soft_dtw.pairwise_distances(targets[k,:,i].view(-1,1),outputs[k,:,i].view(-1,1))
                D[k:k+1,:,:] = Dk
            loss_shape += softdtw_batch(D,gamma)

            path_dtw = path_soft_dtw.PathDTWBatch.apply
            path = path_dtw(D,gamma)
            Omega =  soft_dtw.pairwise_distances(torch.arange(1,N_output+1).view(N_output,1)).to(device)
            loss_temporal +=  torch.sum( path*Omega ) / (N_output*N_output)
            loss += alpha*loss_shape+ (1-alpha)*loss_temporal
        loss=0.01*loss
        return loss, loss_shape, loss_temporal
    def train(self):
        self.model.train()
        total_loss = 0
        n_samples = 0
        alpha=0.9
        gamma = 0.001
        for X, Y in self.data_gen.get_batches(self.data_gen.train_set[0], self.data_gen.train_set[1], self.args.batch_size, True):
           
            self.model.zero_grad()
            output= self.model(X)
            log_var_a = torch.zeros((1,), requires_grad=True).to(device) 
            log_var_b = torch.zeros((1,), requires_grad=True).to(device)
            loss,_,_ = self.msta_loss(output,Y,alpha, gamma, device)
            loss1 = torch.tensor(loss, dtype=float,requires_grad=True)
            loss2=self.criterion(Y, output)
            self.train_mae.append(loss2)
            
            loss = torch.exp(-log_var_a)*loss1 +log_var_a+ torch.exp(-log_var_b)*loss2+log_var_b

            loss.backward()
            grad_norm = self.optim.step()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * self.data_gen.column_num)

        return total_loss / n_samples

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Valid the model while training
    """
    def evaluate(self, mode='train'):
        """
        Arguments:
            mode   - (string) 'valid' or 'test'
        """
        self.model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        if mode == 'train':
            tmp_X = self.data_gen.train_set[0]
            tmp_Y = self.data_gen.train_set[1]
        elif mode == 'test':
            tmp_X = self.data_gen.test_set[0]
            tmp_Y = self.data_gen.test_set[1]
            
        else:
            raise Exception('invalid evaluation mode')

        for X, Y in self.data_gen.get_batches(tmp_X, tmp_Y, self.args.batch_size, False):
            output= self.model(X)
            L1_loss = self.evaluateL1(output, Y).item()
            L2_loss = self.evaluateL2(output, Y).item()
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict, output))
                test = torch.cat((test, Y))
            total_loss_l1 += L1_loss
            total_loss += L2_loss
            n_samples += (output.size(0) * self.data_gen.column_num)

        mse = total_loss / n_samples
        rse=0
        mae = total_loss_l1 / n_samples
        return mse, rse, mae
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def evaluate1(self):

        self.model.eval()
        predict = None
        X = self.data_gen.test_set[0]
        X=torch.tensor(X, dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X=X.to(device)
        output= self.model(X)
        print('output:',output.shape)
        predict = output[:,0:4,:]
        return predict    

    def run(self):
        use_cuda = self.args.gpu is not None
        if use_cuda:
            if type(self.args.gpu) == list:
                self.model = nn.DataParallel(self.model, device_ids=self.args.gpu)
            else:
                torch.cuda.set_device(self.args.gpu)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(self.args.seed)
        if use_cuda: self.model.cuda()

        self.nParams = sum([p.nelement() for p in self.model.parameters()])

        if self.args.L1Loss:
            self.criterion = nn.L1Loss(reduction='sum')
        else:
            self.criterion = nn.SmoothL1Loss(reduction='sum')
        self.evaluateL1 = nn.L1Loss(reduction='sum')
        self.evaluateL2 = nn.MSELoss(reduction='sum')
        if use_cuda:
            self.evaluateL1 = self.evaluateL1.cuda()
            self.evaluateL2 = self.evaluateL2.cuda()

        self.optim = Optimize(self.model.parameters(), self.args.optim, self.args.lr, self.args.clip)

        best_train_mse = float("inf")
        best_train_rse = float("inf")
        best_train_mae = float("inf")

        tmp_losses = []
        try:
            for epoch in range(1, self.args.epochs+1):
                epoch_start_time = time.time()
                train_loss = self.train()
                self.running_times.append(time.time() - epoch_start_time)
                tmp_losses.append(train_loss)
                tra_mse, tra_rse, tra_mae = self.evaluate()
                if tra_mse < best_train_mse:
                    best_train_mse = tra_mse
                    best_train_rse = tra_rse
                    best_train_mae = tra_mae

                self.optim.updateLearningRate(tra_mse, epoch)
        except KeyboardInterrupt:
            pass
        self.predictlast=self.evaluate1()
        self.train_losses.append(tmp_losses)
        print('predict:',self.predictlast.shape)
        temp=self.predictlast.cpu().detach().numpy().tolist()
        csv_fp = open('./'+self.args.data+'.csv', "w", encoding='utf-8')
        writer = csv.writer(csv_fp)
        for d in temp:
            writer.writerow(d)
        final_best_mse = best_train_mse
        final_best_rse = best_train_rse
        final_best_mae = best_train_mae

        self.best_rmse = np.sqrt(final_best_mse)
        self.best_rse = final_best_rse
        self.best_mae = final_best_mae

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Compute and output the metrics
    """
    def getMetrics(self):
        print('-' * 100)
        print()
        print('* number of parameters: %d' % self.nParams)
        for k in self.args.__dict__.keys():
            print(k, ': ', self.args.__dict__[k])
        running_times = np.array(self.running_times)
        print("time: sum {:8.7f} | mean {:8.7f}".format(np.sum(running_times), np.mean(running_times)))
        print("rmse: {:8.7f}".format(self.best_rmse))
        print("rse: {:8.7f}".format(self.best_rse))
        print("mae: {:8.7f}".format(self.best_mae))
        print()
