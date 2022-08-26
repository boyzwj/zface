from configparser import Interpolation
from genericpath import isfile
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


class HifiFaceDataset1(Dataset):
    def __init__(self, dataset_root_list, same_prob=0.5):
        super(HifiFaceDataset1, self).__init__()
        self.images = []
        self.image_identity = {}
        self.identity_ids = {}
        self.same_prob = same_prob
        cur_identity_id = 0
        cur_image_id = 0
        for dataset_root in dataset_root_list:
            for id_root in os.listdir(dataset_root):
                id_path = f"{dataset_root}/{id_root}"
                if os.path.isdir(id_path):
                    cur_identity_img_ids = []
                    for img_path in os.listdir(id_path):
                        if img_path.endswith("g"):
                            self.images.append(f"{id_path}/{img_path}")
                            self.image_identity[cur_image_id] = cur_identity_id
                            cur_identity_img_ids.append(cur_image_id)
                            cur_image_id += 1
                    self.identity_ids[cur_identity_id] = cur_identity_img_ids
                    cur_identity_id = cur_identity_id + 1
                elif os.path.isfile(id_path) and id_path.endswith("g"):
                    self.images.append(f"{id_path}")
                    cur_image_id += 1
                    self.identity_ids[cur_identity_id] = [cur_image_id]
                    cur_identity_id = cur_identity_id + 1
        self.identity_num = cur_identity_id
        self.image_num = cur_image_id
                    
                               
        

        self.transforms1 = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomRotation((-10,10),transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-10,10),transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def random_img_from_identity(self,identity_id):
        image_ids = self.identity_ids[identity_id]
        image_id = random.choice(image_ids)
        return image_id
      
    def get_img_by_id(self, idx):
        image_path = self.images[idx]
        img = Image.open(image_path).convert("RGB")
        return img
        
        
    def __getitem__(self, idx):
        Xs = self.get_img_by_id(idx)
        if random.random() < self.same_prob:
            identity_id = self.image_identity[idx]
            t_img_id = self.random_img_from_identity(identity_id)
            Xt = self.get_img_by_id(t_img_id)
            same_person = 1
            return self.transforms1(Xs), self.transforms1(Xt), same_person
        else:
            t_image_path = random.choice(self.images)
            Xt = Image.open(t_image_path).convert("RGB")
            same_person = 0
            return self.transforms2(Xs), self.transforms2(Xt), same_person            

    

    def __len__(self):
        return len(self.image_num)
    


class HifiFaceDataset2(Dataset):
    def __init__(self, dataset_root_list, same_prob=0.5):
        super(HifiFaceDataset2, self).__init__()
        self.images = []
        self.image_identity = {}
        self.identity_ids = {}
        self.same_prob = same_prob
        cur_identity_id = 0
        cur_image_id = 0
        for dataset_root in dataset_root_list:
            for id_root in os.listdir(dataset_root):
                id_path = f"{dataset_root}/{id_root}"
                if os.path.isdir(id_path):
                    cur_identity_img_ids = []
                    for img_path in os.listdir(id_path):
                        if img_path.endswith("g"):
                            self.images.append(f"{id_path}/{img_path}")
                            self.image_identity[cur_image_id] = cur_identity_id
                            cur_identity_img_ids.append(cur_image_id)
                            cur_image_id += 1
                    if len(cur_identity_img_ids) > 0:
                        self.identity_ids[cur_identity_id] = cur_identity_img_ids
                        cur_identity_id = cur_identity_id + 1
                elif os.path.isfile(id_path) and id_path.endswith("g"):
                    self.images.append(f"{id_path}")
                    self.identity_ids[cur_identity_id] = [cur_image_id]
                    cur_image_id += 1

                    cur_identity_id = cur_identity_id + 1
        self.identity_num = cur_identity_id
        self.image_num = cur_image_id
                    
                               
        self.transforms1 = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomRotation((-5,5),transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-5,5),transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def random_img_from_identity(self,identity_id):
        image_ids = self.identity_ids[identity_id]
        image_id = random.choice(image_ids)
        return image_id
      
    def get_img_by_id(self, idx):
        image_path = self.images[idx]
        img = Image.open(image_path).convert("RGB")
        return img
        
        
    def __getitem__(self, idx):
        image_ids = self.identity_ids[idx]
        img_id = random.choice(image_ids)
        Xs = self.get_img_by_id(img_id)
        if random.random() < self.same_prob or len(image_ids)>1:
            t_img_id = self.random_img_from_identity(idx)
            Xt = self.get_img_by_id(t_img_id)
            same_person = 1
        else:
            t_idx = random.randint(0,self.identity_num - 1)
            t_img_id = self.random_img_from_identity(t_idx)
            Xt = self.get_img_by_id(t_img_id)
            if t_idx == idx:
                same_person = 1
            else:
                same_person = 0
        if same_person == 1:                
            return self.transforms1(Xs), self.transforms1(Xt), same_person
        else:
            return self.transforms2(Xs), self.transforms2(Xt), same_person        

    

    def __len__(self):
        return self.identity_num
    




class MultiResolutionDataset(Dataset):
    def __init__(self, path, resolution=256, same_prob=0.5):
        self.path = path
        self.same_prob = same_prob
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomRotation((-10,10),transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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

        # if random.random() > 0.5:
        #     Xs = complex_imgaug(Xs,self.resolution)    
        # if random.random() > 0.5:
        #     Xt = complex_imgaug(Xt,self.resolution)
        Xs = self.transform(Xs)
        Xt = self.transform(Xt)

        return Xs, Xt, same_person