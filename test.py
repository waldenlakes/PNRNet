import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
import torch.optim as optim
import torchvision.utils as utils
from torch.utils.data import DataLoader
from ignite.metrics import SSIM as ignite_ssim
from ignite.metrics import PSNR as ignite_psnr
import lpips
from PIL import Image
import numpy as np
import os.path as osp
from datetime import datetime
import os
import time

from model.test_model import test_PNRNet
from utils import create_folder, load_config
from data.test_data_utils import TesDatasetFromFolder


def filename2img(filename,dataset_dir,resize=512):
    img = Image.open(osp.join(dataset_dir, '%s.png' % filename))
    if resize != 1024:
        img = img.resize((resize,resize))
    img = np.array(img, dtype=np.float32)#(288,288,3)
    img = img/255.0
    img = img.transpose(2, 0, 1)[:3,...]#(W,H,C)->(C,W,H)

    return img.reshape(1,3,img.shape[1],img.shape[2])
    
def init_weights(net,model_file):
    state_dict = torch.load(model_file,map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict,strict=True)

def init_weights_(net,model_file):
    state_dict = torch.load(model_file,map_location='cpu')
    net.load_state_dict(state_dict,strict=True)

def heatmap(data_list, savepath, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    data = np.stack(data_list, axis=0)
    data = data.reshape((9, 10))
    f, ax = plt.subplots(figsize=(9.5, 8))
    ax = sns.heatmap(data, annot=True, cmap="YlGnBu")  # , ax=axis)
    fig = ax.get_figure()
    fig.savefig(os.path.join(savepath, "{}.png".format(filename)))

# ------------------------- configuration -------------------------
config = load_config('./configs/config_test.json')['config']
print(config)

devices = config['gpus']
os.environ["CUDA_VISIBLE_DEVICES"] = devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_dir = config['test_dir']
save_dir = config['save_dir']
pretrained_model_path = config['pretrained_model_path']
gt_dir = config['gt_dir']
save_dir = create_folder(save_dir)
# ------------------------- configuration -------------------------

loss_fn = lpips.LPIPS()#-1 and 1 -> 0.8140
if(torch.cuda.is_available()):
    loss_fn.cuda()

if __name__ == '__main__':
    val_set = TesDatasetFromFolder(test_dir,val_txt_name='val.txt',resize_size=1024,normal_map=True,depth_map=True,pos_map=True)

    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

    PNRNet = test_PNRNet()
    init_weights_(PNRNet, pretrained_model_path)
    PNRNet.to(device)

    val_iteration = 0
    mae_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    ssim = ignite_ssim(data_range=1.0)
    psnr = ignite_psnr(data_range=1.0)

    start_time = time.time()

    with torch.no_grad():
        imgs = []
        mae = []
        lpips_list = []
        mps_list_ours = []
        for source_img, source_normal_map, src_pos_map, source_depth, guide_img, guide_normal_map, guide_pos, guide_depth, source_img_name in val_loader:
            source_img = source_img.to(device)
            source_normal_map = source_normal_map.to(device)
            guide_img = guide_img.to(device)
            guide_normal_map = guide_normal_map.to(device)
            #target_img = target_img.to(device)
            source_depth = source_depth.to(device)
            guide_depth = guide_depth.to(device)
            src_pos_map = src_pos_map.to(device)

            pred_target_img = PNRNet(source_depth, source_img, source_normal_map, src_pos_map, guide_depth, guide_img, val_iteration)
            
            # print(source_img_name)
            gt = filename2img(source_img_name,dataset_dir=gt_dir,resize=1024)
            gt = torch.from_numpy(gt).to(device)

            display_data = torch.cat([pred_target_img, gt], dim=0)
            utils.save_image(display_data, save_dir + "/%s.png"%(source_img_name), nrow=pred_target_img.shape[0], padding=2, normalize=False)

            ssim.update([pred_target_img, gt])
            psnr.update([pred_target_img, gt])

            loss_lpips = loss_fn.forward(pred_target_img*2.0-1, gt*2.0-1)
            loss_lpips = loss_lpips.sum() / gt.shape[0]
            lpips_total += loss_lpips

            print(val_iteration)
            val_iteration += 1

        time_elapsed = time.time() - start_time

        ssim_index = ssim.compute()
        psnr_index = psnr.compute()
        print('PSNR: %f SSIM: %f LPIPS: %f' % (psnr_index, ssim_index, lpips_total / val_iteration))

        print('Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('runtime per image {:.3f}[s]'.format(time_elapsed / val_iteration))