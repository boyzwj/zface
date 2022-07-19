from io import BytesIO
import lmdb
import os
import glob
import random
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image, ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True




# def parsing2mask(paring):
#     img_numpy = np.array(paring)

#     mask_nose = color_masking(img_numpy, 76, 153, 0)
#     mask_left_eye = color_masking(img_numpy, 204, 0, 204)
#     mask_right_eye = color_masking(img_numpy, 51, 51, 255)
#     mask_skin = color_masking(img_numpy, 204, 0, 0)
#     mask_left_eyebrow = color_masking(img_numpy, 255, 204, 204)
#     mask_right_eyebrow = color_masking(img_numpy, 0, 255, 255)
#     mask_up_lip = color_masking(img_numpy, 255, 255, 0)
#     mask_mouth_inside = color_masking(img_numpy, 102, 204, 0)
#     mask_down_lip = color_masking(img_numpy, 0, 0, 153)
#     mask_left_ear = color_masking(img_numpy, 255, 0, 0)
#     mask_right_ear = color_masking(img_numpy, 102, 51, 0)
#     # mask_glass = color_masking(img_numpy, 204, 204, 0)

#     mask_face = logical_or_masks(
#         [mask_nose, mask_left_eye, mask_right_eye, mask_skin, mask_left_eyebrow, mask_right_eyebrow, mask_up_lip,
#          mask_mouth_inside, mask_down_lip, mask_left_ear, mask_right_ear, ])
#     mask_face = 1.0 * mask_face
#     mask_face = Image.fromarray(np.array(mask_face))
#     return mask_face


class HifiFaceParsingTrainDataset(Dataset):
    def __init__(self, dataset_root_list, same_prob=0.5):
        super(HifiFaceParsingTrainDataset, self).__init__()
        self.datasets = []
        self.N = []
        self.same_prob = same_prob
 
        for dataset_root in dataset_root_list:
            imgpaths_in_root = glob.glob(f'{dataset_root}/*.*g')

            for root, dirs, _ in os.walk(dataset_root):
                for dir in dirs:
                    imgpaths_in_root += glob.glob(f'{root}/{dir}/*.*g')

            self.datasets.append(imgpaths_in_root)
            self.N.append(len(imgpaths_in_root))

        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
        image_path = self.datasets[idx][item]
        
        Xs = Image.open(image_path).convert("RGB")

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = Image.open(image_path).convert("RGB")
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return sum(self.N)




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
        return img

    def __getitem__(self, idx):
        Xs = self.get_img(idx)
        l = self.__len__()
        if random.random() > self.same_prob:
            t_idx = random.randrange(l)
        else:
            t_idx = idx    
        Xs = self.transform(Xs)    
        if t_idx == idx:
            same_person = 1
            Xt = Xs.detach().clone()
        else:
            same_person = 0 
            Xt = self.get_img(t_idx)  
            Xt = self.transform(Xt)
        return Xs, Xt, same_person