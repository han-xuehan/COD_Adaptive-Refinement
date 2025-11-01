import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class PSTSeg(data.Dataset):

    def __init__(self, cfg, mode='trainval', do_aug=True):

        assert mode in ['train', 'val', 'trainval', 'test'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        #self.root = '/home/yclab/guangyu/Segmentation/data/PST900_RGBT_Dataset/test/'
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])


        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.4537, 44.2457, 31.6650, 46.4071, 30.1391])
            self.binary_class_weight = np.array([1.4507, 21.5033])
        else:
            raise (f"{cfg['class_weight']} not support.")
        
        img_path = os.path.join(self.root, 'rgb/')
        
        #img_list = [f for f in os.listdir(img_path) if f.endswith('.jpg') or f.endswith('.png')]
        if self.mode == 'train':
            split_file = 'train.txt'
        else: 
            split_file = 'test.txt'
            
        split_file_path = os.path.join(self.root, split_file)
        #print(split_file_path)
        with open(split_file_path, 'r') as f:
            selected_images = [line.strip() for line in f.readlines()]
        #print(len(selected_images))
        self.infos = selected_images
        
        self.images = [(os.path.join(self.root,'rgb' ,img_name + '.png')) for img_name in selected_images]
        #print(len(self.images))
        #self.thermal = [(os.path.join(self.root, 'thermal', img_name+ '.png')) for img_name in selected_images]
        self.depth = [(os.path.join(self.root, 'depth', img_name+ '.png')) for img_name in selected_images]
        self.gts = [(os.path.join(self.root, 'labels', img_name+ '.png')) for img_name in selected_images]
        #for training
        self.binary_label = [(os.path.join(self.root, 'binary_labels', img_name+ '.png')) for img_name in selected_images]
        self.edge = [(os.path.join(self.root, 'bound', img_name+ '.png')) for img_name in selected_images]

        self.images = sorted(self.images)
        #self.thermal = sorted(self.thermal)
        self.depth = sorted(self.depth)
        self.gts = sorted(self.gts)
        
        self.binary_label = sorted(self.binary_label)
        self.edge = sorted(self.edge)
       

        #with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            #self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        #image_path = self.infos[index].strip()
        #img_path = os.path.join(self.root, 'rgb/')
        
        #img_list = [f for f in os.listdir(img_path) if f.endswith('.jpg') or f.endswith('.png')]
        #print(img_list)
        '''
        image = Image.open(os.path.join(self.root, 'rgb', image_path + '.png'))
        depth = Image.open(os.path.join(self.root, 'depth', image_path + '.png')).convert('RGB')
        label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))
        bound = Image.open(os.path.join(self.root, 'bound', image_path+'.png'))
        edge = Image.open(os.path.join(self.root, 'bound', image_path+'.png'))
        binary_label = Image.open(os.path.join(self.root, 'binary_labels', image_path + '.png'))
        '''
        image = Image.open(self.images[index])
        #thermal = Image.open(self.thermal[index]).convert('RGB')
        depth = Image.open(self.depth[index]).convert('RGB')
        label = Image.open(self.gts[index])
        
        bound = Image.open(self.gts[index])
        edge = Image.open(self.gts[index])
        binary_label = Image.open(self.gts[index])

        sample = {
            'image': image,
            #'thermal': thermal,
            'depth': depth,
            'label': label,
            'bound': bound,
            'edge': edge,
            'binary_label': binary_label,
        }

        if self.mode in ['train', 'trainval'] and self.do_aug:  # 
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        #sample['thermal'] = self.dp_to_tensor(sample['thermal'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['edge'] = torch.from_numpy(np.asarray(sample['edge'], dtype=np.int64)).long() # 没有edge
        sample['bound'] = torch.from_numpy(np.asarray(sample['bound'], dtype=np.int64) / 255.).long()
        sample['binary_label'] = torch.from_numpy(np.asarray(sample['binary_label'], dtype=np.int64) / 255.).long()
        #sample['binary_label'] = sample['binary_label'].max(1)
        #print(torch.unique(sample['binary_label']))
        #print(torch.unique(sample['bound']))
        #sample['label_path'] = image_path.strip().split('/')[-1] + '.png'  # 
        sample['label_path'] = self.images[index].split('/')[-1] + '.png'
        return sample

    @property
    def cmap(self):
        return [
            [0, 0, 0], # background
            [0, 0, 255], # fire_extinguisher
            [0, 255, 0], # backpack
            [255, 0, 0], # drill
            [255, 255, 255], # survivor/rescue_randy
        ]



