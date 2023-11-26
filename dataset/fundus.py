import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import dataset.transform as trans
import torch
import numpy as np
from dataset.transform import to_multilabel


class Fundus(Dataset):
    def __init__(self, domain_idx=None, base_dir=None, split='train', num=None, transform=None):
        self.transform = transform
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
        self.domain_idx = domain_idx
        self.split = split
        
        if split == 'train':
            with open(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'train.list'), 'r') as f:
                self.id_path = f.readlines()
        elif split == 'test':
            with open(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'test.list'), 'r') as f:
                self.id_path = f.readlines()
        
        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))
    
    def __len__(self):
        return len(self.id_path)
    
    def __getitem__(self, index):
        id = self.id_path[index]
        img = Image.open(os.path.join(self.base_dir, self.domain_name[self.domain_idx], id.split(' ')[0][8:]))

        if self.split == 'test':
            mask = Image.open(os.path.join(self.base_dir, self.domain_name[self.domain_idx], id.split(' ')[1][8:])).convert('L')
            sample = {'img': img, 'mask': mask}
            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()
            # crop during test #
            # mask = mask[..., 144:144+512, 144:144+512]

            if self.transform:
                sample = self.transform(sample)
            return sample['img'], sample['mask'], mask, id
        
        else:
            mask = Image.open(os.path.join(self.base_dir, self.domain_name[self.domain_idx], id.split(' ')[1][8:])).convert('L')

        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample['img'], sample['mask'], mask, id


class Fundus_Multi(Dataset):
    def __init__(self, domain_idx_list=None, base_dir=None, split='train', num=None, transform=None):
        self.transform = transform
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
        self.domain_idx_list = domain_idx_list
        self.split = split

        self.id_path = []
        if split == 'train':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join('./' + "{}_train.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()
        
        elif split == 'test':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join('./' + "{}_test.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()
        
        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))
    
    def __len__(self):
        return len(self.id_path)
    
    def __getitem__(self, index):
        id = self.id_path[index]
        img = Image.open(os.path.join(self.base_dir, id.split(' ')[0]))
        
        if self.split == 'test':
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')
            sample = {'img': img, 'mask': mask}

            __mask = np.array(mask).astype(np.uint8)
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()
            # crop during test #
            # mask = mask[..., 144:144+512, 144:144+512]

            if self.transform:
                sample = self.transform(sample)
            return sample['img'], sample['mask'], id
        
        else:
            mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1])).convert('L')

        sample = {'img': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample['img'], sample['mask'] , id


class Fundus_Multi2(Dataset):
    def __init__(self, domain_idx_list=None, base_dir=None, split='train', num=None, transform=None):
        self.transform = transform
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
        self.domain_idx_list = domain_idx_list
        self.split = split

        self.id_path = []
        if split == 'train':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join('./' + "{}_train.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()

        elif split == 'test':
            for domain_idx in self.domain_idx_list:
                with open(os.path.join('./' + "{}_test.list".format(self.domain_name[domain_idx])), 'r') as f:
                    self.id_path = self.id_path + f.readlines()

        self.id_path = [item.replace('\n', '') for item in self.id_path]

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))

    def __len__(self):
        return len(self.id_path)

    def __getitem__(self, index):
        id = self.id_path[index]
        img = Image.open(os.path.join(self.base_dir, id.split(' ')[0]))
        mask = Image.open(os.path.join(self.base_dir, id.split(' ')[1]))
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        mask = mask.transpose(2, 0, 1)
        mask = torch.from_numpy(mask).float()

        sample = {'img': img, 'mask': mask}
        return sample['img'], sample['mask'], id

if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    base_dir = '/data/ziqi/datasets/muti_site_med/Fundus'
    trainset = Fundus_Multi(base_dir=base_dir,
                          split='train',
                          domain_idx_list=[0,1,2],
                          transform=transforms.Compose([
                              trans.Resize((256, 256)),
                              trans.Hflip(0.5),
                              trans.Normalize()
                          ]))
    
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    for i, (img, mask) in enumerate(trainloader):
        print(mask.shape)