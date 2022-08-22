import glob
from msilib.schema import File
import torch
from torch import nn
import torch.nn.functional as F
from models.face_models.iresnet import iresnet100
import os
import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np

class IdentityExtractor(nn.Module):
    def __init__(self):
        super(IdentityExtractor, self).__init__()
        self.F_id = iresnet100(pretrained=False, fp16=False)
        self.F_id.load_state_dict(torch.load('./weights/backbone_r100.pth'))
        self.F_id.eval()
        
    def forward(self, I):
        # v_id = self.F_id(F.interpolate(I[:, :, 16:240, 16:240], [112,112], mode='bilinear', align_corners=True))
        # v_id = self.F_id(F.interpolate(I[:, :, 32:480, 32:480], [112,112], mode='bilinear', align_corners=True))
        v_id = self.F_id(F.interpolate(I, 112, mode='bilinear', align_corners=True))
        v_id = F.normalize(v_id)
        return v_id
        

def image_to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)
    
def takefirst(elem):
    return elem[0]

def clean_path(model,path):
    with torch.no_grad():
        fileslist = glob.glob(f'{path}/*.*g')
        l = []
        i = 0
        # sum_sid = torch.zeros((512,),device="cuda:0")
        for filepath in tqdm.tqdm(fileslist):
            i = i + 1
            img = Image.open(filepath)
            img = img.convert("RGB")
            data = image_to_tensor(img)
            data = data.unsqueeze(0).cuda()
            sid = model(data)[0]
            l.append((filepath,sid))
            # sum_sid = sum_sid + sid       
        # mean_sid = sum_sid/i
        # error_imgs = []
        for cfilepath, csid in l:
            tt_loss = 0
            for _, tsid in l:
                loss = (1 - torch.cosine_similarity(csid, tsid,dim=0))
                tt_loss += loss
            mean_loss = tt_loss/i
            if mean_loss > 0.7:
                os.remove(cfilepath)
                print(f"del:{cfilepath},loss:{mean_loss}")
                # error_imgs.append((loss,cfilepath))
        # if len(error_imgs) >0:
        #     loss, path = max(error_imgs)
        #     os.remove(path)
        #     print(f"del:{path},loss:{loss}")

        


def clean_data_set(dataset_root):
    model = IdentityExtractor().to('cuda:0')
    i = 0
    for dirpath in tqdm.tqdm(os.listdir(dataset_root)):
        i = i + 1
        subpath = f'{dataset_root}/{dirpath}'
        if os.path.isdir(subpath) and i > 7300 and i <= 7400 :
            clean_path(model,subpath)
            
            
    # sid = mode.forward(img)
    
    
    # print(path)
    


if __name__ == "__main__":
    clean_data_set('E:\\VGGface2')