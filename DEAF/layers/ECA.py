import argparse
import math
import torch
from torch import nn

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--gamma', type=int, default=3)

    parser.add_argument('--b', type=int, default=2)

    args = parser.parse_args()

    model = ECANet(args.in_channels,args.gamma,args.b).cuda()
    x = torch.randn(32, 96, 7).cuda()
    y = model(x)
    print(x)
    print(y)
    print(y.shape)
