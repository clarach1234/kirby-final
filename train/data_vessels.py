from pathlib import Path
import os
from os.path import join
import re

from torch.utils.data import Dataset
import numpy as np
from skimage import io

import config



def read_image(file):
    img = io.imread(file)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img


def read_file(file):
    if '.npy' in file:
        data = np.load(file)
    elif '.npz' in file:
        data = np.load(file)['data']
    else:
        data = read_image(file)
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data /= 255.0
    return data


class Img2ImgDataset:

    def __init__(self, data):
        self.target_path = os.path.join(data['data_folder'], data['target']['path'])
        self.original_path = os.path.join(data['data_folder'], data['original']['path'])
        self.mask_path = os.path.join(data['data_folder'], data['mask']['path'])
        self.a_path = os.path.join(data['data_folder'], data['a']['path'])
        self.av_path = os.path.join(data['data_folder'], data['av']['path'])
        
        self.target_pattern = data['target']['pattern']
        self.original_pattern = data['original']['pattern']
        self.mask_pattern = data['mask']['pattern']
        self.a_pattern = data['a']['pattern']
        self.av_pattern = data['av']['pattern']

        self._make_dataset()

    def _make_dataset(self):
        target = re.compile(self.target_pattern)
        orig = re.compile(self.original_pattern)
        mask = re.compile(self.mask_pattern)
        a = re.compile(self.a_pattern)
        av = re.compile(self.av_pattern)
        number = re.compile('[0-9]+')

        self.targets = {}
        self.origs = {}
        self.masks = {}
        self.as_ = {}
        self.avs = {}

        paths = [self.target_path, self.original_path, self.mask_path, self.a_path, self.av_path]
        patterns = [target, orig, mask, a, av]
        data_dicts = [self.targets, self.origs, self.masks, self.as_, self.avs]

        for path, pattern, data_dict in zip(paths, patterns, data_dicts):
            for file_name in os.listdir(path):
                if config.dataset.startswith('RITE') or config.dataset.startswith('LES-AV'):
                    n = number.findall(file_name)
                    if pattern.match(file_name) and n:
                        n = int(n[0])
                        data_dict[n] = file_name
                else:
                    if pattern.match(file_name):
                        data_dict[Path(file_name).stem] = file_name

class VesselsDataset_in3(Dataset, Img2ImgDataset):

    def __init__(self, data, transform=None):
        Img2ImgDataset.__init__(self, data)
        self.transform = transform
        self.vessels = self.targets
        self.retinos = self.origs
        self.a_imgs = self.as_
        self.av_imgs = self.avs
        self.indices = [n for n in self.retinos.keys()]

    def __len__(self):
        return len(self.retinos)

    def __getitem__(self, index):
        _index = index
        retino = self.retinos[_index]
        vessel = self.vessels[_index]
        mask = self.masks[_index]

        r = read_file(join(self.original_path, retino))
        m = read_file(join(self.mask_path, mask))
        v = read_file(join(self.target_path, vessel))
        v = np.stack( [v[...,0]+v[...,1], v[...,0]+v[...,1]+v[...,2], v[...,2]+v[...,1]], axis=2 )


        item = [r, v, m]
        if self.transform is not None:
            item = self.transform(item)
        return [_index, item]
        


class VesselsDataset_in5(Dataset, Img2ImgDataset):

    def __init__(self, data, transform=None):
        Img2ImgDataset.__init__(self, data)
        self.transform = transform
        self.vessels = self.targets
        self.retinos = self.origs
        self.a_imgs = self.as_
        self.av_imgs = self.avs
        self.indices = [n for n in self.retinos.keys()]

    def __len__(self):
        return len(self.retinos)

    def __getitem__(self, index):
        _index = index
        retino = self.retinos[_index]
        vessel = self.vessels[_index]
        mask = self.masks[_index]
        a_name = self.a_imgs[_index]
        av_name = self.av_imgs[_index]

        r_rgb = read_file(join(self.original_path, retino)) 
        r_a = read_file(join(self.a_path, a_name)) 
        if r_a.ndim == 2: 
            r_a = r_a[:, :, np.newaxis]
        if r_a.shape[2] != 1: 
            r_a = r_a[:, :, :1]

        r_av = read_file(join(self.av_path, av_name)) 
        if r_av.ndim == 2:  
            r_av = r_av[:, :, np.newaxis]
        if r_av.shape[2] != 1: 
            r_av = r_av[:, :, :1]
            
        r = np.concatenate([r_rgb, r_a, r_av], axis=2)
        m = read_file(join(self.mask_path, mask))
        v = read_file(join(self.target_path, vessel))
        v = np.stack( [v[...,0]+v[...,1], v[...,0]+v[...,1]+v[...,2], v[...,2]+v[...,1]], axis=2 )

        item = [r, v, m]
        if self.transform is not None:
            item = self.transform(item)
        return [_index, item]

