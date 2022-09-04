import glob
from statistics import mode
from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
import os
import tqdm
from torchvision import transforms
from PIL import Image
from models.face_models.iresnet import iresnet100
from io import BytesIO
import lmdb
import numpy as np



to_img = transforms.ToPILImage()


def image_to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

class FaceImage(nn.Module):
    def __init__(self):
        super(FaceImage,self).__init__()

        self.segmentation_net = torch.jit.load('./weights/face_parsing.farl.lapa.main_ema_136500_jit191.pt', map_location="cuda")
        self.segmentation_net.eval()
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5))
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())




    @torch.no_grad()
    def get_mask(self,I):
        with torch.no_grad():
            size = I.size()[-1]
            I = self.unnormalize(I)
            logit , _  = self.segmentation_net(F.interpolate(I, size=(448,448), mode='bilinear'))
            parsing = logit.max(1)[1]
            face_mask = torch.where((parsing>0)&(parsing<10), 1, 0)
            face_mask = F.interpolate(face_mask.unsqueeze(1).float(), size=(size,size), mode='nearest')
            face_mask = self.blur(face_mask)
        return face_mask

    @torch.no_grad()
    def forward(self, I):
        mask = self.get_mask(I)
        return  mask
        


def write_img(env,id, img,mask):
    img_key = f"img-{str(id).zfill(7)}".encode("utf-8")
    msk_key = f"msk-{str(id).zfill(7)}".encode("utf-8")
    with env.begin(write=True) as txn:
        txn.put(img_key, img)
        txn.put(msk_key, mask)

def write_id_range(env,id,range):
    key = f"id-{str(id).zfill(7)}".encode("utf-8")
    with env.begin(write=True) as txn:
        txn.put(key,np.array(range))


def process_img(model,env,id,filepath):
    img = Image.open(filepath)
    img = img.convert("RGB")
    data = image_to_tensor(img)
    data = data.unsqueeze(0).cuda()
    mask = model(data)
    mask_img = to_img(mask[0].cpu())
    imgdata = BytesIO()
    maskdata = BytesIO()
    mask_img.save(maskdata,format="png")
    img.save(imgdata, format="jpeg", quality=100)
    write_img(env,id,imgdata.getvalue(),maskdata.getvalue())




dataset_root_list = ["../../FFHQ","../../CelebA-HQ","../../facefuck","../../Customface"]

if __name__ == "__main__":
    model = FaceImage().cuda()
    cur_identity_id = 0
    cur_image_id = 0
    with lmdb.open("./data/test", map_size=50* (1024 ** 3), readahead=False) as env:
        for dataset_root in tqdm.tqdm(dataset_root_list):
            for id_root in tqdm.tqdm(os.listdir(dataset_root)):
                id_path = f"{dataset_root}/{id_root}"
                if os.path.isdir(id_path):
                    cur_id_begin = cur_image_id
                    for img_path in tqdm.tqdm(os.listdir(id_path)):
                        if img_path.endswith("g"):
                            full_path = f"{id_path}/{img_path}"
                            process_img(model,env,cur_image_id,full_path)
                            cur_image_id += 1
                    if cur_image_id > cur_id_begin:
                        write_id_range(env,cur_identity_id,[cur_id_begin,cur_image_id-1])
                        cur_identity_id += 1
                elif os.path.isfile(id_path) and id_path.endswith("g"):
                    cur_id_begin = cur_image_id
                    full_path = f"{id_path}"
                    process_img(model,env,cur_image_id,full_path)
                    cur_image_id += 1
                    write_id_range(env,cur_identity_id,[cur_id_begin,cur_image_id-1])
                    cur_identity_id = cur_identity_id + 1

        
        with env.begin(write=True) as txn:
            txn.put("id-length".encode("utf-8"), str(cur_identity_id).encode("utf-8"))                        
            txn.put("img-length".encode("utf-8"), str(cur_image_id).encode("utf-8"))                 


