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

from model.cttnet import color_temp_tuning_net
from utils import create_or_recreate_folders, load_config
from data.cttnet_data_utils import TrainDatasetFromFolderRGB_color_temp,ValDatasetFromFolderRGB_color_temp

# ------------------------- configuration -------------------------
config = load_config('./configs/config_cttnet.json')['config']
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
TEST_DATA_PATH = config['test_data_path']
lr = config['training']['lr']
mm = config['training']['mm']
record_train_iter_loss = config['training']['record_train_iter_loss']
# ------------------------- configuration -------------------------

# ------------------------- dataset -------------------------
train_set = TrainDatasetFromFolderRGB_color_temp(TRAIN_DATA_PATH, resize=config['resize'], train_list = config['train_list'])
val_set = ValDatasetFromFolderRGB_color_temp(TEST_DATA_PATH, resize=config['resize'], val_list = config['val_list'])

print(f'Dataset Train: {len(train_set)}')
print(f'Dataset Test : {len(val_set)}')

train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=config['data_workers'], batch_size=config['train_batch'], shuffle=config['train_shuffle'], pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, num_workers=config['data_workers'], batch_size=config['val_batch'], shuffle=config['val_shuffle'])
# ------------------------- dataset -------------------------

# ------------------------- network setup -------------------------
cttnet = color_temp_tuning_net()
print('# Color Temperature Transfer Network parameters:', sum(param.numel() for param in cttnet.parameters()))
opt = optim.SGD(cttnet.parameters(),lr=lr,momentum=mm)

cttnet = nn.DataParallel(cttnet)
cttnet.to(device)
# ------------------------- network setup -------------------------

# verify whether we want to continue with a training or start brand-new
if config['training']['continue']:
    # load weights
    print('------------------- Continue Training -------------------')
    weight = torch.load(f"{config['epoch_folder']}/ckpt{config['training']['epoch']}.pth")
    cttnet.load_state_dict(weight)

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
l1_loss = nn.L1Loss(reduce = False)
def mask_l1_loss(out, target, valid_mask):
    loss = l1_loss(out, target)
    loss = (loss * valid_mask).sum() # gives \sigma_euclidean over unmasked elements
    normlized_loss = loss / valid_mask.sum()

    return normlized_loss
loss_fn = lpips.LPIPS()#-1 and 1 -> 0.8140
if(torch.cuda.is_available()):
    loss_fn.cuda()
# # ------------------------- loss functions setup -------------------------

if __name__ == '__main__':
    for epoch in range(1 + epoch, NUM_EPOCHS + 1):
        
        iteration = 0

        cttnet.train()

        for valid_mask, input_tensor, input_light_direction, input_light_colortemp, target_img, target_light_direction, guide_light_colortemp, target_second_tensor in train_loader:
            iteration += 1

            if torch.cuda.is_available():
                input_tensor = input_tensor.to(device)
                input_light_colortemp = input_light_colortemp.to(device)
                guide_light_colortemp = guide_light_colortemp.to(device)
                target_second_tensor = target_second_tensor.to(device)
                valid_mask = valid_mask.to(device)
                valid_mask = valid_mask.repeat(1,3,1,1)

            cttnet.zero_grad()

            pred_target_second_tensor = cttnet(input_light_colortemp, guide_light_colortemp, input_tensor)

            # computing the loss function
            loss_target_second_tensor = mask_l1_loss(pred_target_second_tensor,target_second_tensor, valid_mask)

            loss = loss_target_second_tensor
            loss.backward()
            opt.step()

            pred_target_second_tensor[valid_mask==0.0] = input_tensor[valid_mask==0.0]

            # console print for the training in progress
            print('[%d/%d][%d] MaskL1Loss: %f' % (epoch, NUM_EPOCHS, iteration, loss_target_second_tensor.item()))

            # display
            if iteration % display_iter == 0:
                with torch.no_grad():
                    display_batchsize = config['training']['display_batchsize']
                    display_data = torch.cat([input_tensor[:display_batchsize,...], target_second_tensor[:display_batchsize,...], pred_target_second_tensor[:display_batchsize,...]], dim=0)
                utils.save_image(display_data, display_folder + "/Epoch_%d Iter_%d.jpg"%(epoch,iteration), nrow=display_batchsize, padding=2, normalize=False)
            
            # loss functions summary
            writer.add_scalar('TrainIter/L1', loss.item(), count_train_iter_loss)
            count_train_iter_loss = count_train_iter_loss + 1

        # one epoch finished, save model
        torch.save(cttnet.state_dict(), epoch_folder + '/ckpt{}.pth'.format(epoch))

        # ------------------------------------------- Test image ----------------------------------------
        with torch.no_grad():
            l1_total = 0.0
            lpips_total = 0.0
            val_iteration = 0

            ssim = ignite_ssim(data_range=1.0)
            psnr = ignite_psnr(data_range=1.0)

            cttnet = cttnet.eval()

            for valid_mask, input_tensor, input_light_direction, input_light_colortemp, target_img, target_light_direction, guide_light_colortemp, target_second_tensor in val_loader:
                
                val_iteration += 1

                if torch.cuda.is_available():
                    input_tensor = input_tensor.to(device)
                    input_light_colortemp = input_light_colortemp.to(device)
                    guide_light_colortemp = guide_light_colortemp.to(device)
                    target_second_tensor = target_second_tensor.to(device)
                    valid_mask = valid_mask.to(device)
                    valid_mask = valid_mask.repeat(1,3,1,1)

            
                pred_target_second_tensor = cttnet(input_light_colortemp, guide_light_colortemp, input_tensor)
                pred_target_second_tensor[valid_mask==0.0] = input_tensor[valid_mask==0.0]

                # computing the loss function
                l1 = mask_l1_loss(pred_target_second_tensor,target_second_tensor, valid_mask)
                ssim.update([pred_target_second_tensor, target_second_tensor])
                psnr.update([pred_target_second_tensor, target_second_tensor])
                loss_lpips = loss_fn.forward(pred_target_second_tensor*2.0-1, target_second_tensor*2.0-1)
                loss_lpips = loss_lpips.sum() / target_second_tensor.shape[0]

                lpips_total += loss_lpips
                l1_total += l1.item()

                print(f'Test [{val_iteration}]')

            # loss functions summary
            ssim_index = ssim.compute()
            psnr_index = psnr.compute()
            print('PSNR: %f SSIM: %f LPIPS: %f' % (psnr_index, ssim_index, lpips_total / val_iteration))
            writer.add_scalar('Val/L1', l1_total / val_iteration, epoch)
            writer.add_scalar('Val/PSNR', psnr_index, epoch)
            writer.add_scalar('Val/SSIM', ssim_index, epoch)
            writer.add_scalar('Val/LPIPS', lpips_total / val_iteration, epoch)
