import os
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

from model.lenet import light_estimation_net
from utils import create_or_recreate_folders, load_config, gradient_loss
from data.lenet_data_utils import TrainDataset
from pytorch_ssim import SSIM

# ------------------------- configuration -------------------------
config = load_config('./configs/config_lenet.json')['config']
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
TRAIN_DATA_PATH = config['train_data_path']
lr = config['training']['lr']
record_train_iter_loss = config['training']['record_train_iter_loss']
# ------------------------- configuration -------------------------

# ------------------------- dataset -------------------------
train_set = TrainDataset(TRAIN_DATA_PATH, train_list_path = config['train_list'])

print(f'Dataset Train: {len(train_set)}')

train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=config['data_workers'], batch_size=config['train_batch'], shuffle=config['train_shuffle'])
# ------------------------- dataset -------------------------

# ------------------------- network setup -------------------------
lenet = light_estimation_net()
opt = optim.Adam(lenet.parameters(), lr=lr)

lenet = nn.DataParallel(lenet)
lenet.to(device)
# ------------------------- network setup -------------------------

# verify whether we want to continue with a training or start brand-new
if config['training']['continue']:
    # load weights
    print('------------------- Continue Training -------------------')
    weight = torch.load(f"{config['epoch_folder']}/ckpt{config['training']['epoch']}.pth")
    lenet.load_state_dict(weight)

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
lightdirection_loss = nn.CrossEntropyLoss()
l2_loss = nn.MSELoss()
ssim_loss = SSIM()
loss_fn = lpips.LPIPS()#-1 and 1 -> 0.8140
if(torch.cuda.is_available()):
    loss_fn.cuda()
# # ------------------------- loss functions setup -------------------------

if __name__ == '__main__':
    for epoch in range(1 + epoch, NUM_EPOCHS + 1):
        running_results = {'LossC_ILD': 0, 'LossC_ITD': 0, 'LossR_TFT': 0, 'Loss_SSIM': 0, 'Loss_LPIPS': 0, 'Loss_grad': 0}
        iteration = 0

        lenet.train()

        for input_tensor, input_depth_tensor, guide_tensor, target_second_tensor, input_light_direction,\
        input_light_colortemp, guide_light_direction, guide_light_colortemp,\
        target_second_light_direction, target_second_light_colortemp in train_loader:
            iteration += 1
            batch_size = input_tensor.size(0)

            if torch.cuda.is_available():
                input_depth_tensor = input_depth_tensor.to(device)
                input_tensor = input_tensor.to(device)
                input_light_direction = input_light_direction.to(device)
                input_light_colortemp = input_light_colortemp.to(device)
                guide_tensor = guide_tensor.to(device)
                guide_light_direction = guide_light_direction.to(device)
                guide_light_colortemp = guide_light_colortemp.to(device)
                target_second_tensor = target_second_tensor.to(device)

            lenet.zero_grad()

            pred_input_light_direction, pred_input_light_colortemp, pred_target_second_tensor = lenet(input_depth_tensor, input_tensor, guide_light_direction, guide_light_colortemp)

            # computing the loss function
            loss_input_light_direction = lightdirection_loss(pred_input_light_direction.view(batch_size,-1),input_light_direction)
            loss_input_light_colortemp = lightdirection_loss(pred_input_light_colortemp.view(batch_size,-1),input_light_colortemp)
            loss_target_second_tensor = l2_loss(pred_target_second_tensor,target_second_tensor)
            loss_ssim = 1 - ssim_loss(pred_target_second_tensor,target_second_tensor)
            loss_grad = gradient_loss(pred_target_second_tensor, target_second_tensor)
            loss_lpips = loss_fn.forward(pred_target_second_tensor*2.0-1, target_second_tensor*2.0-1)
            loss_lpips = loss_lpips.sum() / batch_size

            loss_le = loss_input_light_direction + loss_input_light_colortemp # direct supervision for lenet
            loss_auxiliary = loss_target_second_tensor + loss_grad + loss_lpips + loss_ssim # add auxiliary loss to improve performance
            loss = loss_le + loss_auxiliary
            loss.backward()
            opt.step()

            # console print for the training in progress
            print('[%d/%d][%d] LossC_ILD: %f LossC_ITD: %f LossR_TFT: %f Loss_SSIM: %f' % (epoch, NUM_EPOCHS, iteration, loss_input_light_direction.item()\
                , loss_input_light_colortemp.item(), loss_target_second_tensor.item(), loss_ssim.item()))
            
            if iteration % display_iter == 0:
                with torch.no_grad():
                    display_data = torch.cat([input_tensor, target_second_tensor, pred_target_second_tensor, guide_tensor], dim=0)
                utils.save_image(display_data, display_folder + "/Epoch_%d Iter_%d.jpg"%(epoch,iteration), nrow=batch_size, padding=2, normalize=False)

            # loss functions summary
            running_results['LossC_ILD'] += loss_input_light_direction.data.item()
            running_results['LossC_ITD'] += loss_input_light_colortemp.data.item()
            running_results['LossR_TFT'] += loss_target_second_tensor.data.item()
            running_results['Loss_SSIM'] += loss_ssim.data.item()
            running_results['Loss_grad'] += loss_grad.data.item()
            running_results['Loss_LPIPS'] += loss_lpips.data.item()
            
            if iteration % record_train_iter_loss == 0:
                writer.add_scalar('TrainIter/LossC_ILD', loss_input_light_direction.data.item(), count_train_iter_loss)
                writer.add_scalar('TrainIter/LossC_ITD', loss_input_light_colortemp.data.item(), count_train_iter_loss)
                writer.add_scalar('TrainIter/LossR_TFT', loss_target_second_tensor.data.item(), count_train_iter_loss)
                writer.add_scalar('TrainIter/Loss_SSIM', loss_ssim.data.item(), count_train_iter_loss)
                writer.add_scalar('TrainIter/Loss_grad', loss_grad.data.item(), count_train_iter_loss)
                writer.add_scalar('TrainIter/Loss_LPIPS', loss_lpips.data.item(), count_train_iter_loss)
                count_train_iter_loss += 1

        # one epoch finished, output training loss, save model
        writer.add_scalar('Train/LossC_ILD', running_results['LossC_ILD'] / iteration, epoch)
        writer.add_scalar('Train/LossC_ITD', running_results['LossC_ITD'] / iteration, epoch)
        writer.add_scalar('Train/LossR_TFT', running_results['LossR_TFT'] / iteration, epoch)
        writer.add_scalar('Train/Loss_SSIM', running_results['Loss_SSIM'] / iteration, epoch)
        writer.add_scalar('Train/Loss_grad', running_results['Loss_grad'] / iteration, epoch)
        writer.add_scalar('Train/Loss_LPIPS', running_results['Loss_LPIPS'] / iteration, epoch)

        torch.save(lenet.state_dict(), epoch_folder + '/ckpt{}.pth'.format(epoch))
