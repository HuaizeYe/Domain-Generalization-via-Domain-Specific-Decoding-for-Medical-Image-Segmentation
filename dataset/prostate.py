import os
from torch.utils.data import Dataset
import torch
import numpy as np
import random


class Prostate(Dataset):
    def __init__(self, domain_idx=None, base_dir=None, split='train', num=None, transform=None):
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5', 'Domain6']
        self.domain_idx = domain_idx
        self.split = split
        self.transform = transform

        self.id_path = os.listdir(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'image'))

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))
    
    def __len__(self):
        return len(self.id_path)
    
    def __getitem__(self, index):
        id = self.id_path[index]
        img = np.load(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'image', id))
        # img = np.expand_dims(np.load(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'image', id)), axis=0)
        # img = np.repeat(np.expand_dims(img, axis=0), repeats=3, axis=0)

        if self.split == 'test':
            mask = np.load(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'mask', id))
            sample = {'img': img, 'mask': mask}
            
            if self.transform is not None:
                sample = self.transform(sample)
            img = sample['img']
            mask = sample['mask']
            img = img.transpose(2, 0, 1)

            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()
            
            if 'onehot_label' in sample.keys():
                onehot_label = sample['onehot_label']
                onehot_label = torch.from_numpy(onehot_label).long()
                return img, mask, onehot_label, id.split('/')[-1]

            return img, mask, id.split('/')[-1]
        
        else:
            mask = np.load(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'mask', id))
            sample = {'img': img, 'mask': mask}
            
            if self.transform is not None:
                sample = self.transform(sample)
            img = sample['img']
            mask = sample['mask']
            img = img.transpose(2, 0, 1)
            
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()
            
            if 'onehot_label' in sample.keys():
                onehot_label = sample['onehot_label']
                onehot_label = torch.from_numpy(onehot_label).long()
                return img, mask, onehot_label
            
            return img, mask


class Prostate_Multi(Dataset):
    def __init__(self, domain_idx_list=None, base_dir=None, split='train', num=None, transform=None):
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5', 'Domain6']
        self.domain_idx_list = domain_idx_list
        self.split = split
        self.transform = transform

        self.id_path = []
        for domain_idx in self.domain_idx_list:
            domain_list = os.listdir(os.path.join(self.base_dir, self.domain_name[domain_idx], 'image'))
            domain_list = [self.domain_name[domain_idx] + '/image/' + item for item in domain_list]
            self.id_path = self.id_path + domain_list

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))
    
    def __len__(self):
        return len(self.id_path)
    
    def __getitem__(self, index):
        id = self.id_path[index]
        img = np.load(os.path.join(self.base_dir, id))
        # img = np.expand_dims(np.load(os.path.join(self.base_dir, id)), axis=0)
        # img = np.repeat(np.expand_dims(img, axis=0), repeats=3, axis=0)

        if self.split == 'test':
            mask = np.load(os.path.join(self.base_dir, id.replace('image', 'mask')))
            sample = {'img': img, 'mask': mask}
            
            if self.transform is not None:
                sample = self.transform(sample)
            img = sample['img']
            mask = sample['mask']
            img = img.transpose(2, 0, 1)

            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()
            
            if 'onehot_label' in sample.keys():
                onehot_label = sample['onehot_label']
                onehot_label = torch.from_numpy(onehot_label).long()
                return img, mask, onehot_label, id.split('/')[-1]

            return img, mask, id.split('/')[-1]
        
        else:                
            mask = np.load(os.path.join(self.base_dir, id.replace('image', 'mask')))
            sample = {'img': img, 'mask': mask}
            
            if self.transform is not None:
                sample = self.transform(sample)
            img = sample['img']
            mask = sample['mask']
            img = img.transpose(2, 0, 1)
            
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()
            
            if 'onehot_label' in sample.keys():
                onehot_label = sample['onehot_label']
                onehot_label = torch.from_numpy(onehot_label).long()
                return img, mask, onehot_label
            
            return img, mask, id


if __name__ == '__main__':
    import transform as trans
    from torch.utils.data.dataloader import DataLoader

    base_dir = '/data/ziqi/datasets/muti_site_med/prostate'
    trainset = Prostate_Multi(base_dir=base_dir,
                          split='test',
                          domain_idx_list=[0,1,2,3,4])
    
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    for i, (img, mask, mask, id) in enumerate(trainloader):
        print(img.shape)
        print(mask.shape)
        print(id)