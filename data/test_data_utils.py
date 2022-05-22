import numpy as np
import collections
import os.path as osp
from PIL import Image
from torch.utils.data.dataset import Dataset


class TesDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, val_txt_name = 'val.txt', depth_map=None, normal_map=None, pos_map=None, resize_size = None, normalize = None):
        super(TesDatasetFromFolder, self).__init__()
        self.files = collections.defaultdict(list)
        self.dataset_dir = dataset_dir
        if resize_size == 1024:
            self.resize_size = 1
        elif resize_size == 512:
            self.resize_size = 2
        else:
            self.resize_size = 4
        self.normalize = normalize
        self.normal_map = normal_map
        self.depth_map = depth_map
        self.pos_map = pos_map

        imgsets_file = osp.join(dataset_dir, val_txt_name)
        for did in open(imgsets_file):
            did = did.strip()
            self.files['val'].append({
                'filename': did,
            })

    def filename2img(self, filename, dataset_dir):
        img = Image.open(osp.join(dataset_dir, '%s.png' % filename))
        if self.resize_size==2:
            img = img.resize((512,512))
        elif self.resize_size==4:
            img = img.resize((256,256))
        img = np.array(img, dtype=np.float32)
        img = img/255.0
        img = img.transpose(2, 0, 1)
        return img[:3,...]

    def filename2depth(self, filename, dataset_dir):
        depth_information = np.load(osp.join(dataset_dir, '%s.npy' % filename),allow_pickle=True)
        ref_center_dis = depth_information.item().get('ref_center_dis')
        normalized_depth = depth_information.item().get('normalized_depth')[::self.resize_size,::self.resize_size]
        normalized_depth = normalized_depth.reshape(1,normalized_depth.shape[0],normalized_depth.shape[1])

        return normalized_depth

    def __getitem__(self, index):
        filename = self.files['val'][index]['filename'] #'Pair000'
        input_name = 'input/' + filename #'input/Pair000'
        guide_name = 'guide/' + filename #'guide/Pair000'
        # load image
        input_img = self.filename2img(input_name,self.dataset_dir)
        guide_img = self.filename2img(guide_name,self.dataset_dir)

        if self.normal_map and not self.depth_map and not self.pos_map:
            input_normal = np.load(self.dataset_dir+'/input/normal_map/{}.npy'.format(filename)).astype(np.float32)[:,::self.resize_size,::self.resize_size]
            guide_normal = np.load(self.dataset_dir+'/guide/normal_map/{}.npy'.format(filename)).astype(np.float32)[:,::self.resize_size,::self.resize_size]
            return input_img, input_normal, guide_img, guide_normal, target_img, filename
        elif self.normal_map and self.depth_map and not self.pos_map:
            input_normal = np.load(self.dataset_dir+'/input/normal_map/{}.npy'.format(filename)).astype(np.float32)[:,::self.resize_size,::self.resize_size]
            guide_normal = np.load(self.dataset_dir+'/guide/normal_map/{}.npy'.format(filename)).astype(np.float32)[:,::self.resize_size,::self.resize_size]    
            input_depth = self.filename2depth(filename, self.dataset_dir+'/input')
            guide_depth = self.filename2depth(filename, self.dataset_dir+'/guide')
            return input_img, input_normal, input_depth, guide_img, guide_normal, guide_depth, target_img, filename
        elif self.normal_map and self.depth_map and self.pos_map:
            input_normal = np.load(self.dataset_dir+'/input/normal_map/{}.npy'.format(filename)).astype(np.float32)[:,::self.resize_size,::self.resize_size]
            guide_normal = np.load(self.dataset_dir+'/guide/normal_map/{}.npy'.format(filename)).astype(np.float32)[:,::self.resize_size,::self.resize_size]    
            input_pos = np.load(self.dataset_dir+'/input/xyz/{}.npy'.format(filename)).astype(np.float32)[:,::self.resize_size,::self.resize_size]
            guide_pos = np.load(self.dataset_dir+'/guide/xyz/{}.npy'.format(filename)).astype(np.float32)[:,::self.resize_size,::self.resize_size]            
            input_depth = self.filename2depth(filename, self.dataset_dir+'/input')
            guide_depth = self.filename2depth(filename, self.dataset_dir+'/guide')
            return input_img, input_normal, input_pos, input_depth, guide_img, guide_normal, guide_pos, guide_depth, filename            
        else:
            return input_img, guide_img, filename

    def __len__(self):
        return len(self.files['val'])
