import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DishTS(nn.Module):
    def __init__(self, args):
        super().__init__()
        init = args.dish_init #'standard', 'avg' or 'uniform'
        activate = True
        n_series = args.n_series  # number of series
        lookback = args.lookback  # lookback length
        self.device=args.gpu
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(n_series, lookback, 2) / lookback).to(self.device)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2) / lookback).to(self.device)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2) / lookback + torch.rand(n_series, lookback, 2) / lookback).to(self.device)
        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)).to(self.device), nn.Parameter(torch.zeros(n_series)).to(self.device)
        self.activate = activate

    def preget(self, batch_x):
        x_transpose = batch_x.permute(2, 0, 1)
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1, 2, 0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:, :1, :], theta[:, 1:, :]
        self.xil = torch.sum(torch.pow(batch_x - self.phil, 2), axis=1, keepdim=True) / (batch_x.shape[1] - 1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih, 2), axis=1, keepdim=True) / (batch_x.shape[1] - 1)

    def forward_process(self, batch_input):
        # print(batch_input.shape, self.phil.shape, self.xih.shape)
        temp = (batch_input - self.phil) / torch.sqrt(self.xil + 1e-8)
        rst = (temp.mul(self.gamma) + self.beta)
        return rst
    #反向传播
    def inverse_process(self, batch_input):
        return (((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih)
    #前向传播
    def forward(self, batch_x, mode='forward', dec_inp=None):
        if mode == 'forward':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x).to(self.device)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp).to(self.device)
            return batch_x, dec_inp
        elif mode == 'inverse':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x).to(self.device)
            return batch_y

#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dish_init', type=str, default='standard')
#     #列
#     parser.add_argument('--n_series', type=int, default=3)
#     #行
#     parser.add_argument('--seq_len', type=int, default=96)
#     args = parser.parse_args()
#     model = DishTS(args)
#     # x说明：(N,seq_len,n_series)
#     x = torch.randn(5, 96,3)
#     y = model(batch_x=x)
#     print(y)
