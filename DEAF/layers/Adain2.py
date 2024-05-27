import argparse
import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from layers.Autoformer_EncDec import series_decomp


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size - 1) // 2,bias=False)
        self.act1 = nn.Sigmoid()


    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output


class adaIN(nn.Module):

    def __init__(self,configs):
        super(adaIN, self).__init__()
        self.eps = 1e-5
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        # kernel_size = configs.kernel_size
        # self.decomp = series_decomp(kernel_size)
        self.pooling = nn.AvgPool1d(3,stride=1,padding=1)
        self.pred_len= configs.pred_len
        self.label_len= configs.label_len
        # kernel_size = int(abs(math.log(configs.in_channels, 2)))
        # kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # self.conv1 = nn.Conv1d(configs.in_channels, configs.in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.relu = nn.ReLU()

    def forward(self, input):

        # mean = torch.mean(input, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(input)

        # seasonal_init, trend_init = self.decomp(input)
        # trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        # seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :],(0,0,0,self.pred_len))

        seasonal_init, trend_init=self.pooling(seasonal_init), self.pooling(trend_init)
        out_in=trend_init+seasonal_init

        out_mean, out_var = torch.mean(out_in, dim=[1], keepdim=True), torch.var(out_in, dim=[1], keepdim=True)
        out_in = (out_in - out_mean) / torch.sqrt(out_var + self.eps)+out_in
        temp=self.relu(out_in)
        out = temp+input

        return out

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--kernel_size', type=int, default=3)
#     parser.add_argument('--pred_len', type=int, default=64)
#     parser.add_argument('--label_len', type=int, default=64)
#     parser.add_argument('--in_channels', type=int, default=96)
#     args = parser.parse_args()
#
#     model = adaIN(args)
#     x = torch.randn(32, 96, 7)
#     y = model(x, 0.5)
#     print(x)
#     print(y)
#     print(y.shape)


class ResnetAdaINBlock(nn.Module):

    def __init__(self, configs):
        super(ResnetAdaINBlock, self).__init__()
        self.conv1 = nn.Conv1d(configs.seq_len, configs.seq_len*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.ECA = ECANet(configs.seq_len*2)
        self.conv2 = nn.Conv1d(configs.seq_len*2, configs.seq_len, kernel_size=3, stride=1, padding=1, bias=False)
        self.adain = adaIN(configs)
        self.relu1 = nn.ReLU()
        self.layernorm=nn.LayerNorm([configs.in_channels,configs.n_series])

    def forward(self, x):
        out = self.adain(x)
        out = self.conv1(out)
        out = self.ECA(out)
        out = self.conv2(out)
        out = self.relu1(out)
        out = self.adain(out)
        out = self.layernorm(out)
        return x+out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--pred_len', type=int, default=128)
    parser.add_argument('--label_len', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=96)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--n_series', type=int, default=7)
    args = parser.parse_args()
    model = ResnetAdaINBlock(args)
    x = torch.randn(32, 96, 7)
    y = model(x)
    print(x)
    print(y)
    print(y.shape)
