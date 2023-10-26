import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

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

    def __init__(self, eps=1e-5):
        super(adaIN, self).__init__()
        self.eps = eps

    def forward(self, input, gamma=0.5):

        in_mean, in_var = torch.mean(input, dim=[1,2], keepdim=True), torch.var(input, dim=[1,2], keepdim=True)

        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out = out_in

        out = out * gamma
        return out


class ResnetAdaINBlock(nn.Module):

    def __init__(self, enc_in):
        super(ResnetAdaINBlock, self).__init__()
        self.conv1 = nn.Conv1d(enc_in, enc_in*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.ECA = ECANet(in_channels=enc_in*4)
        self.norm1 = adaIN()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(enc_in*4, enc_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = adaIN()

    def forward(self, x, gamma):
        out = self.conv1(x)
        out = self.ECA(out)
        out = self.norm1(out, gamma)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma)
        return x+out

if __name__ == '__main__':
    model = ResnetAdaINBlock(enc_in=3)
    x = torch.randn(5, 3, 6)
    y = model(x,0.5)
    print(x)
    print(y)
    print(y.shape)
