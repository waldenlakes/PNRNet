import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.layers import ConvBlockG_GN
from model.ldtnet import light_direction_transfer_net

def init_weights(net,model_file):
    state_dict = torch.load(model_file,map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict,strict=False)

class test_PNRNet(nn.Module):
    def __init__(self):
        super(test_PNRNet, self).__init__()
        self.lenet = lighting_estimation_net()
        self.color_net = color_temp_transfer_net()
        self.shading_net = light_direction_transfer_net()

    def forward(self, source_depth, source_img, source_normal_map, src_pos_map, guide_depth, guide_img, val_iteration, pos_ablation=False, normal_ablation=False):
        depths = torch.cat([source_depth,guide_depth],dim=0)
        rgbs = torch.cat([source_img,guide_img],dim=0)

        pred_guide_light_direction, pred_guide_light_colortemp = self.lenet(depths,rgbs)
        pred_guide_light_direction = F.softmax(pred_guide_light_direction)
        pred_guide_light_direction = torch.argmax(pred_guide_light_direction, dim=1)
        pred_guide_light_colortemp = F.softmax(pred_guide_light_colortemp)
        pred_guide_light_colortemp = torch.argmax(pred_guide_light_colortemp, dim=1)
        target_light_direction = pred_guide_light_direction[1,0,:]
        target_light_color = pred_guide_light_colortemp[1,0,:]
        source_light_direction = pred_guide_light_direction[0,0,:]
        source_light_color = pred_guide_light_colortemp[0,0,:]

        pred_target_img_color = self.color_net(source_light_color, target_light_color, source_img)
        if pos_ablation:
            pred_relit_img = self.shading_net(pred_target_img_color,source_normal_map,source_light_direction,target_light_direction,pos_map=src_pos_map, pos_ablation=True)
        elif normal_ablation:
            pred_relit_img = self.shading_net(pred_target_img_color,source_normal_map,source_light_direction,target_light_direction,pos_map=src_pos_map, normal_ablation=True)
        else:
            pred_relit_img = self.shading_net(pred_target_img_color,src_pos_map,source_normal_map,source_light_direction,target_light_direction)
        
        return pred_relit_img

class lighting_NetV2(nn.Module):
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

    def forward(self, innerFeat, target_light, target_light_colortemp):
        x = innerFeat[:,0:self.ncInput,:,:] # lighting feature
        B, C, row, col = x.shape
        feat = x.mean(dim=(2,3), keepdim=True)
        light = self.predict_relu1(self.predict_FC1(feat))
        light = self.predict_FC2(light)
        light_colortemp = self.predict_FC1_colortemp(self.predict_relu1_colortemp(feat))
        light_colortemp = self.predict_FC2_colortemp(light_colortemp)

        return innerFeat, light, light_colortemp

class lighting_estimation_net(nn.Module):                  
    def __init__(self):       
        super(lighting_estimation_net, self).__init__()
        self.lightingV2 = lighting_NetV2(256, 8, 5, 128)

        self.conv1 = ConvBlockG_GN(4, 25, 48)#[3,3]
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlockG_GN(48, 72, 96)#[6,6]
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlockG_GN(96, 144, 192)#[12,12]
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlockG_GN(192, 288, 384)#[24,24]
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = ConvBlockG_GN(384, 576, 768)#[48,48]

    def forward(self, x_depth, x, target_light_direction=None, target_light_colortemp=None):
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
        innerFeat, light, light_colortemp = self.lightingV2(c5,target_light_direction,target_light_colortemp)

        return light, light_colortemp

class color_temp_transfer_net(nn.Module):
    def __init__(self, ncInput=5+5):
        super(color_temp_transfer_net, self).__init__()
        self.fc1 = nn.Conv2d(ncInput, 128, kernel_size=1, stride=1, bias=False)
        self.relu1 = nn.PReLU()
        self.fc2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.relu2 = nn.PReLU()
        self.fc3 = nn.Conv2d(256, 34*3, kernel_size=1, stride=1, bias=False)

    def polynomial_kernel_function_generation(self, rgb, order=4):
        polynomial_kernel_function = []
        r = rgb[:,0:1,...]
        g = rgb[:,1:2,...]
        b = rgb[:,2:3,...]
        polynomial_kernel_function.append(r)
        polynomial_kernel_function.append(g)
        polynomial_kernel_function.append(b)
        polynomial_kernel_function.append(r*r)
        polynomial_kernel_function.append(g*g)
        polynomial_kernel_function.append(b*b)
        polynomial_kernel_function.append(r*g)
        polynomial_kernel_function.append(g*b)
        polynomial_kernel_function.append(r*b)
        polynomial_kernel_function.append(r**3)
        polynomial_kernel_function.append(g**3)
        polynomial_kernel_function.append(b**3)
        polynomial_kernel_function.append(r*g*g)
        polynomial_kernel_function.append(g*b*b)
        polynomial_kernel_function.append(r*b*b)
        polynomial_kernel_function.append(g*r*r)
        polynomial_kernel_function.append(b*g*g)
        polynomial_kernel_function.append(b*r*r)
        polynomial_kernel_function.append(r*g*b)
        if order == 3:
            return torch.cat(polynomial_kernel_function,dim=1)
        polynomial_kernel_function.append(r**4)
        polynomial_kernel_function.append(g**4)
        polynomial_kernel_function.append(b**4)
        polynomial_kernel_function.append(r**3*g)
        polynomial_kernel_function.append(r**3*b)
        polynomial_kernel_function.append(g**3*r)
        polynomial_kernel_function.append(g**3*b)
        polynomial_kernel_function.append(b**3*r)
        polynomial_kernel_function.append(b**3*g)
        polynomial_kernel_function.append(r**2*g**2)
        polynomial_kernel_function.append(g**2*b**2)
        polynomial_kernel_function.append(r**2*b**2)
        polynomial_kernel_function.append(r**2*g*b)
        polynomial_kernel_function.append(g**2*r*b)
        polynomial_kernel_function.append(b**2*r*g)

        return torch.cat(polynomial_kernel_function,dim=1)

    def weight_estimation_net(self, source, target):
        B = source.shape[0]
        source_label = torch.zeros(B,5).to(source.device)
        source_index = source.view(-1,1)
        source_label.scatter_(dim=1, index=source_index, value=1)
        source_label = source_label.view(B,5,1,1)

        target_label = torch.zeros(B,5).to(target.device)
        target_index = target.view(-1,1)
        target_label.scatter_(dim=1, index=target_index, value=1)
        target_label = target_label.view(B,5,1,1)

        label = torch.cat([source_label,target_label], dim=1)

        feat = self.relu1(self.fc1(label))
        feat = self.relu2(self.fc2(feat))
        weight = self.fc3(feat)

        return weight.view(B,3,34)

    def forward(self, source, target, source_rgb):
        weight = self.weight_estimation_net(source, target)# b,3,34
        polynomial_kernel_feature = self.polynomial_kernel_function_generation(source_rgb) # b,34,h,w
        b,c,h,w = polynomial_kernel_feature.shape
        polynomial_kernel_feature = polynomial_kernel_feature.view(b,c,-1)
        target_rgb = torch.matmul(weight, polynomial_kernel_feature)
        target_rgb = target_rgb.view(b,3,h,w)

        return target_rgb

if __name__ == '__main__':
    net = color_temp_transfer_net()
    print(net)