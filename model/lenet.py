import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P

from model.layers import ConvBlockG_GN, ResizeConv

class lighting_NetV2(nn.Module):
    '''
        define lighting network
    '''
    def __init__(self, ncInput, ncOutput_direction, ncOutput_colortemp, ncMiddle):
        super(lighting_NetV2, self).__init__()
        self.ncInput = ncInput
        self.ncOutput_direction = ncOutput_direction
        self.ncOutput_colortemp = ncOutput_colortemp
        self.ncOutput = ncOutput_direction + ncOutput_colortemp
        self.ncMiddle = ncMiddle

        self.predict_FC1 = nn.Conv2d(self.ncInput,  self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.predict_relu1 = nn.PReLU()
        self.predict_FC2 = nn.Conv2d(self.ncMiddle, self.ncOutput_direction, kernel_size=1, stride=1, bias=False)

        self.predict_FC1_colortemp = nn.Conv2d(self.ncInput,  self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.predict_relu1_colortemp = nn.PReLU()
        self.predict_FC2_colortemp = nn.Conv2d(self.ncMiddle, self.ncOutput_colortemp, kernel_size=1, stride=1, bias=False)

        self.post_FC1 = nn.Conv2d(self.ncOutput,  self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.post_relu1 = nn.PReLU()
        self.post_FC2 = nn.Conv2d(self.ncMiddle, self.ncInput, kernel_size=1, stride=1, bias=False)
        self.post_relu2 = nn.ReLU()  # to be consistance with the original feature

    def forward(self, innerFeat, target_light, target_light_colortemp, updata_only_lighting_estimation=False):
        x = innerFeat[:,0:self.ncInput,:,:] # lighting feature
        B, C, row, col = x.shape

        if updata_only_lighting_estimation:
            ###output lighting direction 
            feat = x.detach().mean(dim=(2,3), keepdim=True)
        else:
            feat = x.mean(dim=(2,3), keepdim=True)
        light = self.predict_relu1(self.predict_FC1(feat))
        light = self.predict_FC2(light)
        light_colortemp = self.predict_relu1_colortemp(self.predict_FC1_colortemp(feat))
        light_colortemp = self.predict_FC2_colortemp(light_colortemp)

        # get back the feature space
        label = torch.zeros(B,8).to(target_light.device)
        index = target_light.view(-1,1)
        label_colortemp = torch.zeros(B,5).to(target_light_colortemp.device)
        index_colortemp = target_light_colortemp.view(-1,1)

        label.scatter_(dim=1, index=index, value=1)
        label = label.view(B,8,1,1)
        label_colortemp.scatter_(dim=1, index=index_colortemp, value=1)
        label_colortemp = label_colortemp.view(B,5,1,1)
        label_fused = torch.cat([label,label_colortemp], dim=1)

        upFeat = self.post_relu1(self.post_FC1(label_fused))
        upFeat = self.post_relu2(self.post_FC2(upFeat))
        upFeat = upFeat.repeat((1,1,row, col))
        innerFeat[:,0:self.ncInput,:,:] = upFeat
        return innerFeat, light, light_colortemp

class light_estimation_net(nn.Module):                  
    def __init__(self):       
        super(light_estimation_net, self).__init__()
        # lenet
        ## lenet: encoder
        self.conv1 = ConvBlockG_GN(4, 12, 24)#[3,3]
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlockG_GN(24, 36, 48)#[6,6]
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlockG_GN(48, 72, 96)#[12,12]
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlockG_GN(96, 144, 192)#[24,24]
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = ConvBlockG_GN(192, 288, 384)#[48,48]

        ## lenet: light estimation block
        self.lightingV2 = lighting_NetV2(128, 8, 5, 64)

        # extral decoder for performance improvment
        self.up6 = ResizeConv(384, 192)
        self.conv6 = ConvBlockG_GN(384, 288, 192)
        self.up7 = ResizeConv(192, 96)
        self.conv7 = ConvBlockG_GN(192, 144, 96)
        self.up8 = ResizeConv(96, 48)
        self.conv8 = ConvBlockG_GN(96, 72, 48)
        self.up9 = ResizeConv(48, 24)
        self.conv9 = ConvBlockG_GN(48, 36, 24)
        self.conv10 = nn.Conv2d(24, 3, 1)

    def forward(self, x_depth, x, target_light_direction, target_light_colortemp):
        x = torch.cat([x, x_depth], dim=1)

        c1 = self.conv1(x)#[512,512]
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)#[256,256]
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)#[128,128]
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)#[64,64]
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)#[32,32]

        innerFeat, light_direc, light_colortemp = self.lightingV2(c5,target_light_direction,target_light_colortemp)

        up_6 = self.up6(innerFeat)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        return light_direc, light_colortemp, c10
        