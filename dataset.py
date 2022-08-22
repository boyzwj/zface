from configparser import Interpolation
from io import BytesIO
from re import I
import lmdb
import os
import glob
import random
import numpy as np
from pathlib import Path
from omegaconf import IntegerNode
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image, ImageFile,ImageOps
import imgaug as ia
import imgaug.augmenters as iaa
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True





class HifiFaceDataset(Dataset):
    def __init__(self, dataset_root_list, same_prob=0.5):
        super(HifiFaceDataset, self).__init__()
        self.identity = []
        self.dict = {}
        self.same_prob = same_prob
        
        for dataset_root in dataset_root_list:
            for id_root in os.listdir(dataset_root):
                id_path = f"{dataset_root}/{id_root}"
                if os.path.isdir(id_path):
                    img_paths_in_root = glob.glob(f'{id_path}/*.*g')
                    if len(img_paths_in_root) >0:
                        self.identity.append(id_path)
                        self.dict[id_path] = img_paths_in_root


        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        id_path = self.identity[idx]
        img_paths = self.dict[id_path]
        img_path = random.choice(img_paths)
        Xs = Image.open(img_path).convert("RGB")
        if random.random() < self.same_prob:
            t_image_path = random.choice(img_paths)
            Xt = Image.open(t_image_path).convert("RGB")
            same_person = 1
        else:
            t_id_path = random.choice(self.identity)
            if t_id_path == id_path:
                same_person = 1
                t_image_path = random.choice(img_paths)
                Xt = Image.open(t_image_path).convert("RGB")
            else:
                same_person = 0               
                t_img_paths = self.dict[t_id_path]
                t_image_path = random.choice(t_img_paths)
                Xt = Image.open(t_image_path).convert("RGB")
        return self.transforms(Xs), self.transforms(Xt), same_person
    

    def __len__(self):
        return len(self.identity)



def complex_imgaug(x, org_size):
    """input single RGB PIL Image instance"""
    # scale_size = np.random.randint(128,org_size)
    x = np.array(x)
    x = x[np.newaxis, :, :, :]
    aug_seq = iaa.Sequential([
            # iaa.Sometimes(0.5, iaa.OneOf([
            #     iaa.GaussianBlur((1, 3)),
            #     iaa.AverageBlur(k=(1, 3)),
            #     iaa.MedianBlur(k=(1, 3)),
            #     iaa.MotionBlur((3, 7))
            # ])),
            # iaa.Resize(scale_size, interpolation=ia.ALL),
            iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=0.2)),
            iaa.Sometimes(0.5, iaa.JpegCompression(compression=(10, 30))),
            # iaa.Resize(org_size)
        ])
    
    aug_img = aug_seq(images=x)
    return Image.fromarray(aug_img[0])


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, same_prob=0.5):
        self.path = path
        self.same_prob = same_prob
        self.resolution = resolution
        self.transform = transform
        self.blacklist = np.array([40650])
        env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not env:
            raise IOError('Cannot open lmdb dataset', self.path)
        with env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            print(f"begin load  data {self.length}")
        env.close()
    
    def open_lmdb(self):
        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', self.path)
        with self.env.begin(write=False) as txn:
            self.txn = txn




    def get_index(self, idx):
        shift = sum(self.blacklist <= idx)
        return idx + shift
    
    def check_consistency(self):
        for index in range(self.length):
            with self.env.begin(write=False) as txn:
                key = f'{self.resolution}-{str(index).zfill(7)}'.encode('utf-8')
                img_bytes = txn.get(key)

            buffer = BytesIO(img_bytes)
            try:
                img = Image.open(buffer)
            except:
                print(f'Exception at {index}')

    def __len__(self):
        return self.length
    
    def get_img(self,idx):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(idx).zfill(7)}'.encode('utf-8')
            img_bytes = txn.get(key)
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
        return img

    def __getitem__(self, idx):
        Xs = self.get_img(idx)

        l = self.__len__()
        if random.random() > self.same_prob:
            t_idx = random.randrange(l)
        else:
            t_idx = idx

        if t_idx == idx:
            same_person = 1
            Xt = Xs.copy()
        else:
            same_person = 0 
            Xt = self.get_img(t_idx)

        if random.random() > 0.5:
            Xs = complex_imgaug(Xs,self.resolution)    
        if random.random() > 0.5:
            Xt = complex_imgaug(Xt,self.resolution)
        Xs = self.transform(Xs)
        Xt = self.transform(Xt)

        return Xs, Xt, same_person