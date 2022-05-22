from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import collections
import os.path as osp
import numpy as np
import random


class TrainDataset(Dataset):
    def __init__(self, dataset_dir, train_list_path, crop_size=None, resize_size=None, random_flipping = None, normalize = None):
        super(TrainDataset, self).__init__()
        self.files = collections.defaultdict(list)
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.random_flipping = random_flipping
        self.normalize = normalize

        for did in open(train_list_path):
            did = did.strip()
            self.files['train'].append({
                'filename': did,
            })

        self.len_of_dataset = len(self.files['train'])

    def filename2lightparams(self, filename):
        if filename[9:13] == '2500':
            light_colortemp = 0
        elif filename[9:13] == '3500':
            light_colortemp = 1
        elif filename[9:13] == '4500':
            light_colortemp = 2
        elif filename[9:13] == '5500':
            light_colortemp = 3
        elif filename[9:13] == '6500':
            light_colortemp = 4
        else:
            light_colortemp = 0

        if filename[14:] == 'N':
            light_direction = 0
        elif filename[14:] == 'NE':
            light_direction = 1
        elif filename[14:] == 'E':
            light_direction = 2
        elif filename[14:] == 'SE':
            light_direction = 3
        elif filename[14:] == 'S':
            light_direction = 4
        elif filename[14:] == 'SW':
            light_direction = 5
        elif filename[14:] == 'W':
            light_direction = 6
        elif filename[14:] == 'NW':
            light_direction = 7
        else:
            light_direction = 0

        return light_direction, light_colortemp

    def filename2img(self, filename):
        img = Image.open(osp.join(self.dataset_dir, '%s.png' % filename))
        img = np.array(img, dtype=np.float32)#(288,288,3)
        img = img/255.0
        img = img.transpose(2, 0, 1)#(W,H,C)->(C,W,H)
        return img[:3,...]

    def filename2depth(self, filename):
        depth_information = np.load(osp.join(self.dataset_dir, '%s.npy' % filename),allow_pickle=True)
        ref_center_dis = depth_information.item().get('ref_center_dis')
        normalized_depth = depth_information.item().get('normalized_depth')
        if self.resize_size:
            normalized_depth = np.resize(normalized_depth,(self.resize_size,self.resize_size))
        normalized_depth = normalized_depth.reshape(1,normalized_depth.shape[0],normalized_depth.shape[1])

        return normalized_depth

    def filename2shadow(self, filename):
        shadow = Image.open(osp.join(self.dataset_dir, '%s.png' % filename)).convert('L')
        shadow = np.array(shadow, dtype=np.float32)#(288,288,3)
        shadow = shadow/255.0
        shadow = shadow.reshape(1,shadow.shape[0],shadow.shape[1])
        return shadow

    def mean_subtraction(self, input_tensor):
        B,W,H = input_tensor.shape
        input_tensor[0,...] = input_tensor[0,...] - torch.mean(input_tensor[0,...])
        input_tensor[1,...] = input_tensor[1,...] - torch.mean(input_tensor[1,...])
        input_tensor[2,...] = input_tensor[2,...] - torch.mean(input_tensor[2,...])
        return input_tensor

    def transform(self, input_img, guide_img, target_first_img, target_second_img, resize_size = None, crop_size = None, random_flipping = None):
        # Resize
        if resize_size != None:
            resize = transforms.Resize(size=(resize_size, resize_size))
            input_img = resize(input_img)
            guide_img = resize(guide_img)
            target_first_img = resize(target_first_img)
            target_second_img = resize(target_second_img)

        # Random crop
        if crop_size != None:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(crop_size, crop_size))
            input_img = TF.crop(input_img, i, j, h, w)
            guide_img = TF.crop(guide_img, i, j, h, w)
            target_first_img = TF.crop(target_first_img, i, j, h, w)
            target_second_img = TF.crop(target_second_img, i, j, h, w)

        if random_flipping != None:
            # Random horizontal flipping
            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                guide_img = TF.hflip(guide_img)
                target_first_img = TF.hflip(target_first_img)
                target_second_img = TF.hflip(target_second_img)

            # Random vertical flipping
            if random.random() > 0.5:
                input_img = TF.vflip(input_img)
                guide_img = TF.vflip(guide_img)
                target_first_img = TF.vflip(target_first_img)
                target_second_img = TF.vflip(target_second_img)

        # Transform to tensor
        input_img = TF.to_tensor(input_img)
        guide_img = TF.to_tensor(guide_img)
        target_first_img = TF.to_tensor(target_first_img)
        target_second_img = TF.to_tensor(target_second_img)

        if self.normalize is not None:
            input_img = self.mean_subtraction(input_img)
            guide_img = self.mean_subtraction(guide_img)
            target_first_img = self.mean_subtraction(target_first_img)
            target_second_img = self.mean_subtraction(target_second_img)

        return input_img, guide_img, target_first_img, target_second_img

    def __getitem__(self, index):
        input_name = self.files['train'][index]['filename']

        guide_index = np.random.randint(self.len_of_dataset)
        guide_name = self.files['train'][guide_index]['filename']
        target_first_stage_name = input_name[:14] + guide_name[14:]
        target_second_stage_name = input_name[:9] + guide_name[9:]

        #load light_params
        input_light_direction, input_light_colortemp = self.filename2lightparams(input_name)
        guide_light_direction, guide_light_colortemp = self.filename2lightparams(guide_name)
        #target_first_light_direction, target_first_light_colortemp = self.filename2lightparams(target_first_stage_name)
        target_second_light_direction, target_second_light_colortemp = self.filename2lightparams(target_second_stage_name)

        #load image
        input_img = self.filename2img(input_name)
        guide_img = self.filename2img(guide_name)
        target_second_img = self.filename2img(target_second_stage_name)
        input_depth = self.filename2depth(input_name[:8])


        return input_img, input_depth, guide_img, target_second_img, input_light_direction,\
         input_light_colortemp, guide_light_direction, guide_light_colortemp,\
         target_second_light_direction, target_second_light_colortemp

    def __len__(self):
        return len(self.files['train'])
