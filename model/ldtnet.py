import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.layers import RDB, Ups, Downs


class light_direction_transfer_net(nn.Module):
    def __init__(self, nchns_in=11, depthRate=48, growthRate=48, nDenselayer=4, column=3, row=6, stride=2):
        super(light_direction_transfer_net, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(np.ones((column, row, 2, depthRate*stride**(column-1)))), requires_grad=True)

        self.input_block = nn.Sequential(
                nn.Conv2d(nchns_in, depthRate, kernel_size=3, padding=1),
                RDB(depthRate, nDenselayer, growthRate)
                )

        self.output_block = nn.Sequential(
                RDB(depthRate, nDenselayer, growthRate),
                nn.Conv2d(depthRate, 3, kernel_size=3, padding=1),
                nn.ReLU()
                )

        self.RDB_blocks = nn.ModuleDict({})
        nchns_rdb = depthRate
        for c in range(1, column+1):
            for r in range(1, row):
                self.RDB_blocks.update({
                        f'RDB_{c}_{r}': RDB(nchns_rdb, nDenselayer, growthRate)
                })
            nchns_rdb *= stride

        self.Downs_blocks = nn.ModuleDict({})
        nchns_down = depthRate
        for c in range(1, column):
            for r in range(1, row//2+1):
                self.Downs_blocks.update({
                        f'Downs_{c}_{r}': Downs(nchns_down)
                })
            nchns_down *= stride

        self.Ups_blocks = nn.ModuleDict({})
        nchns_up = 2 * depthRate
        for c in range(1, column):
            for r in range(1, row//2+1):
                self.Ups_blocks.update({
                        f'Ups_{c}_{r}': Ups(nchns_up)
                })
            nchns_up *= stride

    def forward(self, source_img, pos_map, normal_map, input_light_direction, target_light_direction):
        n,c,w,h = source_img.shape
        input_light_direction = input_light_direction.view(n,1,1,1).repeat(1,1,h,w)
        target_light_direction = target_light_direction.view(n,1,1,1).repeat(1,1,h,w)
        x = torch.cat([source_img,normal_map,pos_map,input_light_direction,target_light_direction],dim=1)

        feats_11 = self.input_block(x)
        feats_21 = self.Downs_blocks['Downs_1_1'](feats_11)
        feats_31 = self.Downs_blocks['Downs_2_1'](feats_21)

        feats_12 = self.RDB_blocks['RDB_1_1'](feats_11)
        feats_22 = self.attention_weights[1,1,0,:feats_21.shape[1]].view(1,-1,1,1) * self.RDB_blocks['RDB_2_1'](feats_21) + \
                        self.attention_weights[1,1,1,:feats_21.shape[1]].view(1,-1,1,1) * self.Downs_blocks['Downs_1_2'](feats_12)
        feats_32 = self.attention_weights[2,1,0,:feats_31.shape[1]].view(1,-1,1,1) * self.RDB_blocks['RDB_3_1'](feats_31) + \
                        self.attention_weights[2,1,1,:feats_31.shape[1]].view(1,-1,1,1) * self.Downs_blocks['Downs_2_2'](feats_22)

        feats_13 = self.RDB_blocks['RDB_1_2'](feats_12)
        feats_23 = self.attention_weights[1,2,0,:feats_22.shape[1]].view(1,-1,1,1) * self.RDB_blocks['RDB_2_2'](feats_22) + \
                        self.attention_weights[1,2,1,:feats_22.shape[1]].view(1,-1,1,1) * self.Downs_blocks['Downs_1_3'](feats_13)
        feats_33 = self.attention_weights[2,2,0,:feats_32.shape[1]].view(1,-1,1,1) * self.RDB_blocks['RDB_3_2'](feats_32) + \
                        self.attention_weights[2,2,1,:feats_32.shape[1]].view(1,-1,1,1) * self.Downs_blocks['Downs_2_3'](feats_23)

        feats_34 = self.RDB_blocks['RDB_3_3'](feats_33)
        feats_24 = self.attention_weights[1,3,0,:feats_23.shape[1]].view(1,-1,1,1) * self.RDB_blocks['RDB_2_3'](feats_23) + \
                        self.attention_weights[1,3,1,:feats_23.shape[1]].view(1,-1,1,1) * self.Ups_blocks['Ups_2_1'](feats_34, feats_23.size())
        feats_14 = self.attention_weights[0,3,0,:feats_13.shape[1]].view(1,-1,1,1) * self.RDB_blocks['RDB_1_3'](feats_13) + \
                        self.attention_weights[0,3,1,:feats_13.shape[1]].view(1,-1,1,1) * self.Ups_blocks['Ups_1_1'](feats_24, feats_13.size())
        
        feats_35 = self.RDB_blocks['RDB_3_4'](feats_34)
        feats_25 = self.attention_weights[1,4,0,:feats_24.shape[1]].view(1,-1,1,1) * self.RDB_blocks['RDB_2_4'](feats_24) + \
                        self.attention_weights[1,4,1,:feats_24.shape[1]].view(1,-1,1,1) * self.Ups_blocks['Ups_2_2'](feats_35, feats_24.size())
        feats_15 = self.attention_weights[0,4,0,:feats_14.shape[1]].view(1,-1,1,1) * self.RDB_blocks['RDB_1_4'](feats_14) + \
                        self.attention_weights[0,4,1,:feats_14.shape[1]].view(1,-1,1,1) * self.Ups_blocks['Ups_1_2'](feats_25, feats_14.size())

        feats_36 = self.RDB_blocks['RDB_3_5'](feats_35)
        feats_26 = self.attention_weights[1,5,0,:feats_25.shape[1]].view(1,-1,1,1) * self.RDB_blocks['RDB_2_5'](feats_25) + \
                        self.attention_weights[1,5,1,:feats_25.shape[1]].view(1,-1,1,1) * self.Ups_blocks['Ups_2_3'](feats_36, feats_25.size())
        feats_16 = self.attention_weights[0,5,0,:feats_15.shape[1]].view(1,-1,1,1) * self.RDB_blocks['RDB_1_5'](feats_15) + \
                        self.attention_weights[0,5,1,:feats_15.shape[1]].view(1,-1,1,1) * self.Ups_blocks['Ups_1_3'](feats_26, feats_15.size())

        out = self.output_block(feats_16)

        return out

if __name__ == "__main__":
    source_img = torch.randn(1,3,256,256) 
    pos_map = torch.randn(1,3,256,256) 
    normal_map = torch.randn(1,3,256,256) 
    input_light_direction = torch.randn(1,1)
    target_light_direction = torch.randn(1,1)

    ldtnet = light_direction_transfer_net()
    out = ldtnet(source_img, pos_map, normal_map, input_light_direction, target_light_direction)
    print(out.shape)
    