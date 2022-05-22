import os.path as osp
from tkinter.messagebox import NO
from PIL import Image
import collections
import numpy as np
import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class TrainDatasetFromFolderRGB_color_temp(Dataset):
    def __init__(self, dataset_dir, resize=512, train_list=None):
        super(TrainDatasetFromFolderRGB_color_temp, self).__init__()
        self.dataset_dir = dataset_dir
        if resize == 256:
            self.resize = 4
        elif resize == 512:
            self.resize = 2
        else:
            self.resize = 1

        self.train_list = np.load(train_list)

    def num2direc(self, index):
        if index == 0:
            light_direction = '_N'
        elif index == 1:
            light_direction = '_NE'
        elif index == 2:
            light_direction = '_E'
        elif index == 3:
            light_direction = '_SE'
        elif index == 4:
            light_direction = '_S'
        elif index == 5:
            light_direction = '_SW'
        elif index == 6:
            light_direction = '_W'
        elif index == 7:
            light_direction = '_NW'

        return light_direction

    def num2color(self, index):
        if index == 0:
            light_direction = '_2500'
        elif index == 1:
            light_direction = '_3500'
        elif index == 2:
            light_direction = '_4500'
        elif index == 3:
            light_direction = '_5500'
        elif index == 4:
            light_direction = '_6500'

        return light_direction

    def filename2img(self,filename,resize=None):
        img = Image.open(osp.join(self.dataset_dir, '%s.png' % filename))
        if resize:
            img = img.resize((resize,resize))
        img = np.array(img, dtype=np.float32)#(288,288,3)
        img = img/255.0
        img = img.transpose(2, 0, 1)#(W,H,C)->(C,W,H)
        return img[:3,...]

    def filename2shadow(self, filename, resize=None):
        shadow = Image.open(osp.join(self.dataset_dir, '%s.png' % filename)).convert('L')
        if resize:
            shadow = shadow.resize((resize,resize))
        shadow = np.array(shadow, dtype=np.float32)#(288,288,3)
        shadow = shadow/255.0
        shadow = shadow.reshape(1,shadow.shape[0],shadow.shape[1])
        return shadow

    def filename2depth(self, filename):
        depth_information = np.load(osp.join(self.dataset_dir, '%s.npy' % filename),allow_pickle=True)
        ref_center_dis = depth_information.item().get('ref_center_dis')
        normalized_depth = depth_information.item().get('normalized_depth')
        normalized_depth = normalized_depth.reshape(1,normalized_depth.shape[0],normalized_depth.shape[1])

        return normalized_depth

    def __getitem__(self, index):
        id_index = int(index/40)
        id_index = self.train_list[id_index]
        # print(id_index)
        light_index = index%40
        target_index = random.randint(0,40-1)

        self.filename = 'Image%03d'%id_index#'Image282_6500'

        input_depth = self.filename2depth(self.filename[:8])
        valid_mask = np.ones_like(input_depth)
        valid_mask[input_depth==0.0] = 0.0

        filename = self.filename + self.num2color(int(light_index/8)) + self.num2direc(light_index%8)
        input_img = self.filename2img(filename)

        target_filename = self.filename + self.num2color(int(target_index/8)) + self.num2direc(target_index%8)
        target_img = self.filename2img(target_filename)

        target_color_filename = self.filename + self.num2color(int(target_index/8)) + self.num2direc(light_index%8)
        target_img_color = self.filename2img(target_color_filename)

        return valid_mask[:,::self.resize,::self.resize], input_img[:,::self.resize,::self.resize], light_index%8, int(light_index/8), \
             target_img[:,::self.resize,::self.resize], target_index%8, int(target_index/8), target_img_color[:,::self.resize,::self.resize]

    def __len__(self):
        return len(self.train_list)*40

class ValDatasetFromFolderRGB_color_temp(Dataset):
    def __init__(self, dataset_dir, resize=512, val_list=None):
        super(ValDatasetFromFolderRGB_color_temp, self).__init__()
        self.dataset_dir = dataset_dir
        if resize == 256:
            self.resize = 4
        else:
            self.resize = 2

        self.val_list = np.load(val_list)

    def num2direc(self, index):
        if index == 0:
            light_direction = '_N'
        elif index == 1:
            light_direction = '_NE'
        elif index == 2:
            light_direction = '_E'
        elif index == 3:
            light_direction = '_SE'
        elif index == 4:
            light_direction = '_S'
        elif index == 5:
            light_direction = '_SW'
        elif index == 6:
            light_direction = '_W'
        elif index == 7:
            light_direction = '_NW'

        return light_direction

    def num2color(self, index):
        if index == 0:
            light_direction = '_2500'
        elif index == 1:
            light_direction = '_3500'
        elif index == 2:
            light_direction = '_4500'
        elif index == 3:
            light_direction = '_5500'
        elif index == 4:
            light_direction = '_6500'

        return light_direction

    def filename2img(self,filename,resize=None):
        img = Image.open(osp.join(self.dataset_dir, '%s.png' % filename))
        if resize:
            img = img.resize((resize,resize))
        img = np.array(img, dtype=np.float32)#(288,288,3)
        img = img/255.0
        img = img.transpose(2, 0, 1)#(W,H,C)->(C,W,H)
        return img[:3,...]

    def filename2shadow(self, filename, resize=None):
        shadow = Image.open(osp.join(self.dataset_dir, '%s.png' % filename)).convert('L')
        if resize:
            shadow = shadow.resize((resize,resize))
        shadow = np.array(shadow, dtype=np.float32)#(288,288,3)
        shadow = shadow/255.0
        shadow = shadow.reshape(1,shadow.shape[0],shadow.shape[1])
        return shadow

    def filename2depth(self, filename):
        depth_information = np.load(osp.join(self.dataset_dir, '%s.npy' % filename),allow_pickle=True)
        ref_center_dis = depth_information.item().get('ref_center_dis')
        normalized_depth = depth_information.item().get('normalized_depth')
        normalized_depth = normalized_depth.reshape(1,normalized_depth.shape[0],normalized_depth.shape[1])

        return normalized_depth

    def __getitem__(self, index):
        id_index = int(index/40)
        id_index = self.val_list[id_index]
        # print(id_index)
        light_index = index%40
        target_index = random.randint(0,40-1)

        self.filename = 'Image%03d'%id_index#'Image282_6500'

        input_depth = self.filename2depth(self.filename[:8])
        valid_mask = np.ones_like(input_depth)
        valid_mask[input_depth==0.0] = 0.0

        filename = self.filename + self.num2color(int(light_index/8)) + self.num2direc(light_index%8)
        input_img = self.filename2img(filename)

        target_filename = self.filename + self.num2color(int(target_index/8)) + self.num2direc(target_index%8)
        target_img = self.filename2img(target_filename)

        target_color_filename = self.filename + self.num2color(int(target_index/8)) + self.num2direc(light_index%8)
        target_img_color = self.filename2img(target_color_filename)

        return valid_mask[:,::self.resize,::self.resize], input_img[:,::self.resize,::self.resize], light_index%8, int(light_index/8), \
             target_img[:,::self.resize,::self.resize], target_index%8, int(target_index/8), target_img_color[:,::self.resize,::self.resize]

    def __len__(self):
        return len(self.val_list)*40
