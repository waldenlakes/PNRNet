from os import listdir
import os.path as osp
from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

def polynomial_kernel_function_generation(rgb, order=4):
    polynomial_kernel_function = []
    r = rgb[0:1,...]
    g = rgb[1:2,...]
    b = rgb[2:3,...]
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
        return np.concatenate(polynomial_kernel_function,axis=0)
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

    return np.concatenate(polynomial_kernel_function,axis=0)

class TrainDataset(Dataset):
    def __init__(self, dataset_dir, train_list_path, cttnet_weights_dir, resize=1, return_target_color=False):
        super(TrainDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.cttnet_weights_dir = cttnet_weights_dir
        self.dataset_list = np.load(train_list_path)
        self.resize = resize
        self.return_target_color = return_target_color

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

    def determined_bound(self, random_num):
        if random_num < 10:
            down,up = 2500,3500
        elif random_num < 20:
            down,up = 3500,4500
        elif random_num < 30:
            down,up = 4500,5500
        else:
            down,up = 5500,6500

        return down,up

    def transfer_color(self, weight, img):
        polynomial_kernel_feature = polynomial_kernel_function_generation(img)
        c,h,w = polynomial_kernel_feature.shape
        polynomial_kernel_feature = np.reshape(polynomial_kernel_feature,(c,h*w))
        target_rgb = np.matmul(weight, polynomial_kernel_feature)
        target_rgb =  np.reshape(target_rgb,(weight.shape[0],h,w))

        return target_rgb

    def query_weight(self, src, tgt):
        weight = np.load(osp.join(self.cttnet_weights_dir, "weight_{}_{}.npy".format(src,tgt)))

        return weight

    def random_color_transfer(self, src_img,tgt_img,src_light_colortemp):
        random_num = random.randint(0,40)
        down, up = self.determined_bound(random_num)
        target = 2500 + random_num * 100
        if self.return_target_color:
            self.target_lighting_color = random_num
        g = (1/target - 1/up) / (1/down - 1/up)
        if random_num<40:
            down,up = int(random_num/10),int(random_num/10)+1
        elif random_num==40:
            down,up = 3,4
        weight = g * self.query_weight(src_light_colortemp,down) + (1-g)* self.query_weight(src_light_colortemp,up)
        src_img = self.transfer_color(weight, src_img)
        tgt_img = self.transfer_color(weight, tgt_img)

        return src_img, tgt_img

    def __getitem__(self, index):
        id_index = int(index/40)
        id_index = self.dataset_list[id_index]
        light_index = index%40
        target_index = random.randint(0,40-1)

        self.filename = 'Image%03d'%id_index#'Image282_6500'
        self.normal_map = np.load(osp.join(self.dataset_dir, 'normal_map/{}.npy'.format(self.filename[:8]))).astype(np.float32)
        self.pos_map = np.load(osp.join(self.dataset_dir, 'xyz/{}.npy'.format(self.filename[:8])))
        self.depth = self.filename2depth(self.filename[:8])

        filename = self.filename + self.num2color(int(light_index/8)) + self.num2direc(light_index%8)
        input_img = self.filename2img(filename)

        target_filename = self.filename + self.num2color(int(target_index/8)) + self.num2direc(target_index%8)
        target_img = self.filename2img(target_filename)
        target_direc_filename = self.filename + self.num2color(int(light_index/8)) + self.num2direc(target_index%8)
        target_img_direc = self.filename2img(target_direc_filename)

        # data augmentation
        # input_img,target_img_direc = self.random_color_transfer(input_img,target_img_direc,int(light_index/8))

        if self.return_target_color:
            return self.depth[:,::self.resize,::self.resize], self.normal_map[:,::self.resize,::self.resize], \
            self.pos_map[:,::self.resize,::self.resize], input_img[:,::self.resize,::self.resize], light_index%8, self.target_lighting_color,\
             target_img[:,::self.resize,::self.resize], target_index%8, target_img_direc[:,::self.resize,::self.resize]            
        else:
            return self.depth[:,::self.resize,::self.resize], self.normal_map[:,::self.resize,::self.resize], \
            self.pos_map[:,::self.resize,::self.resize], input_img[:,::self.resize,::self.resize], light_index%8, \
             target_img[:,::self.resize,::self.resize], target_index%8, target_img_direc[:,::self.resize,::self.resize]

    def __len__(self):
        return len(self.dataset_list)*40
