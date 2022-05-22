import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlockG_GN(nn.Module): 
    def __init__(self, in_ch, mid_ch, out_ch):
        super(ConvBlockG_GN, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), 
            nn.GroupNorm(mid_ch,mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1),
            nn.GroupNorm(out_ch,out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        y = self.conv0(x)
        return self.conv(x) + y 

class ResizeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResizeConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.reflectionpad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.upsample(x)
        x = self.reflectionpad(x)
        x = self.conv(x)
        return x

class Ups(nn.Module):
    def __init__(self, nChannels, kernel_size=3, stride=2):
        super(Ups, self).__init__()
        self.deconv = nn.ConvTranspose2d(nChannels, nChannels, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(nChannels, nChannels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out

class Downs(nn.Module):
    def __init__(self, nChannels, kernel_size=3, stride=2):
        super(Downs, self).__init__()
        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(nChannels, stride*nChannels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out

class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, dilation=1, act='prelu'):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=dilation, bias=True,
                              dilation=dilation)

        if act == 'prelu':
            self.act = nn.PReLU(growthRate)
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.LeakyReLU()

    def forward(self, x):
        # (kernel_size - 1) // 2 + 1
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, dilation, act='***'):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate, dilation=dilation, act=act))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, act='relu'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        if act == 'prelu':
            self.act = nn.PReLU(growthRate)
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.LeakyReLU()

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, act='***'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, act=act))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out
