import os
import shutil
import numpy as np
import json
import datetime
import torch
import torch.nn.functional as F

def create_folder(folder):
    daytime = datetime.datetime.now().strftime('%Y-%m-%d')
    hourtime = datetime.datetime.now().strftime("%H:%M:%S")
    #print(now_time + "\n"+daytime + "\n" + hourtime)

    pwd = "/"+daytime+"-"+hourtime

    word_name = os.path.exists(folder + pwd)
    print(folder + pwd)

    if not word_name:
        os.makedirs(folder + pwd)

    return folder + pwd

def create_or_recreate_folders(configs):
    """
    deletes existing folder if they already exist and
    recreates then. Only valid for training mode. does not work in
    resume mode
    :return:
    """

    folders = [configs['display_folder'],
               configs['summary'],
               configs['epoch_folder'],
               configs['display_val']]

    # iterate through the folders and delete them if they exist
    # then recreate them.
    # otherwise simply create them
    for i in range(len(folders)):
        folder = folders[i]
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)
        else:
            os.makedirs(folder)

def load_config(file):
    """
    takes as input a file path and returns a configuration file
    that contains relevant information to the training of the NN
    :param file:
    :return:
    """

    # load the file as a raw file
    loaded_file = open(file)

    # conversion from json file to dictionary
    config = json.load(loaded_file)

    # returning the file to the caller
    return config

def gradient_loss(gen_frames, gt_frames, alpha=1):

    def gradient(x):
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

def tv_loss(x,tv_loss_weight=2e-8):
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = tensor_size(x[:, :, 1:, :])
    count_w = tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
