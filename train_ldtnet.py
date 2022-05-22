import os
from traceback import print_tb
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
import lpips
from ignite.metrics import SSIM as ignite_ssim
from ignite.metrics import PSNR as ignite_psnr

from model.ldtnet import light_direction_transfer_net
from utils import create_or_recreate_folders, load_config
from data.ldtnet_data_utils import TrainDataset
from pytorch_ssim import SSIM

# ------------------------- configuration -------------------------
config = load_config('./configs/config_ldtnet.json')['config']
print(config)

devices = config['gpus']
os.environ["CUDA_VISIBLE_DEVICES"] = devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter(config['summary'])
display_folder = config['display_folder']
display_validation = config['display_val']
epoch_folder = config['epoch_folder']
train_mode = config['training']['mode']
NUM_EPOCHS = config['training']['epochs']
display_iter = config['training']['display']
display_iter_8lightingdirection = config['training']['display_8lightingdirection']
TRAIN_DATA_PATH = config['train_data_path']
lr = config['training']['lr']
record_train_iter_loss = config['training']['record_train_iter_loss']
# ------------------------- configuration -------------------------

# ------------------------- dataset -------------------------
train_set = TrainDataset(TRAIN_DATA_PATH, train_list_path=config['train_list'], cttnet_weights_dir=config['cttnet_weights_dir'], resize=config['train_resize'])

print(f'Dataset Train: {len(train_set)}')

train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=config['data_workers'], batch_size=config['train_batch'], shuffle=config['train_shuffle'])
# ------------------------- dataset -------------------------

# ------------------------- network setup -------------------------
ldtnet = light_direction_transfer_net()
opt = optim.Adam(ldtnet.parameters(), lr=lr, betas=(0.9, 0.999))

ldtnet = nn.DataParallel(ldtnet)
ldtnet.to(device)
# ------------------------- network setup -------------------------

# verify whether we want to continue with a training or start brand-new
if config['training']['continue']:
    # load weights
    print('------------------- Continue Training -------------------')
    weight = torch.load(f"{config['epoch_folder']}/ckpt{config['training']['epoch']}.pth")
    ldtnet.load_state_dict(weight)

    # set current epoch
    epoch = config['training']['epoch']
    count_train_iter_loss = config['training']['count_train_iter_loss']
else:
    print('------------------- Starting Training -------------------')
    create_or_recreate_folders(config)
    writer = SummaryWriter(config['summary'])
    epoch = 0
    count_train_iter_loss = 1

# # ------------------------- loss functions setup -------------------------
l1_loss = nn.L1Loss()
ssim_loss = SSIM()
# # ------------------------- loss functions setup -------------------------

if __name__ == '__main__':
    for epoch in range(1 + epoch, NUM_EPOCHS + 1):
        running_results = {'Loss_L1': 0, 'Loss_SSIM': 0}
        iteration = 0

        ldtnet.train()

        for depth_map, normal_map, pos_map, source_img, input_light_direction, target_img, target_light_direction, target_img_direc in train_loader:
            iteration += 1
            batch_size = source_img.size(0)

            if torch.cuda.is_available():
                normal_map = normal_map.to(device)
                pos_map = pos_map.to(device)
                source_img = source_img.to(device)
                input_light_direction = input_light_direction.to(device)
                target_light_direction = target_light_direction.to(device)
                target_img_direc = target_img_direc.to(device)

            ldtnet.zero_grad()

            pred_target_img = ldtnet(source_img, pos_map, normal_map, input_light_direction, target_light_direction)

            # computing the loss function
            loss_l1 = l1_loss(pred_target_img, target_img_direc)
            loss_ssim = 1 - ssim_loss(pred_target_img, target_img_direc)

            loss_total = loss_l1 + loss_ssim
            loss_total.backward()
            opt.step()

            # console print for the training in progress
            print('[%d/%d][%d] Loss_L1: %f Loss_SSIM: %f' % (epoch, NUM_EPOCHS, iteration, loss_l1.item(), loss_ssim.item()))
            
            # display
            if iteration % display_iter == 0:
                display_data = torch.cat([source_img,target_img_direc,pred_target_img],dim=0)
                utils.save_image(display_data, display_folder + "/Epoch_%d Iter_%d.jpg"%(epoch,iteration), nrow=batch_size, padding=1, normalize=False)

            if iteration % display_iter_8lightingdirection == 0:
                save_image = []
                with torch.no_grad():
                    for i in range(8):
                        display_batch_size = 1
                        guide_light_direction = i * torch.ones(display_batch_size, dtype=torch.long).to(device)

                        pred_relighting_image = ldtnet(
                            source_img[0:display_batch_size],pos_map[0:display_batch_size], normal_map[0:display_batch_size],
                            input_light_direction[0:display_batch_size],guide_light_direction[0:display_batch_size])

                        save_image.append(pred_relighting_image)

                display_data = torch.cat([source_img[0:display_batch_size, :3],
                                          target_img_direc[0:display_batch_size, ...], save_image[0], save_image[1],
                                          save_image[2], save_image[3], save_image[4], save_image[5], save_image[6],
                                          save_image[7]], dim=0)
                utils.save_image(display_data,
                                 display_folder + "/Epoch_%d Iter_%d 8 lightingdirection.jpg" % (epoch, iteration),
                                 nrow=display_batch_size, padding=2, normalize=False)
                                 
            # loss functions summary
            running_results['Loss_L1'] += loss_l1.data.item()
            running_results['Loss_SSIM'] += loss_ssim.data.item()
            
            if iteration % record_train_iter_loss == 0:
                writer.add_scalar('TrainIter/Loss_L1', loss_l1.data.item(), count_train_iter_loss)
                writer.add_scalar('TrainIter/Loss_SSIM', loss_ssim.data.item(), count_train_iter_loss)
                count_train_iter_loss += 1

        # one epoch finished, output training loss, save model
        writer.add_scalar('Train/Loss_L1', running_results['Loss_L1'] / iteration, epoch)
        writer.add_scalar('Train/Loss_SSIM', running_results['Loss_SSIM'] / iteration, epoch)

        torch.save(ldtnet.state_dict(), epoch_folder + '/ckpt{}.pth'.format(epoch))
