from turtle import forward
from models.ca import CoordAtt
from models.modulated_conv2d import Conv2DMod,StyledConv2d
from models.gen4 import Generator
from models.discriminator import  ProjectedDiscriminator
import torch
import torch.nn as nn
from models.face_models.iresnet import iresnet100
from models.bfm import ParametricFaceModel
from models.reconnet import ReconNet
from torch_utils import training_stats
import numpy as np
import timm






            


if __name__ == '__main__':
    model = torch.jit.script(Generator())
    # half = torch.randn(10, 10, dtype=torch.float16, device='cuda')
    # const = torch.ones(10, 10, device='cuda')
    # inputs = torch.randn(10, 10, device='cuda')
    x = torch.randn(4,3,256,256)
    z_id = torch.randn(4,512)
    out = model(x,z_id)
    print(out.shape)
    
    # F_id = torch.jit.script(ParametricFaceModel(is_train=False))
    # F_id.load_state_dict(torch.load('./weights/backbone_r100.pth'))
    # F_id.eval()
    # m = torch.jit.script(Encoder())
    # # m = GenResBlk(128,64)
    # x = torch.randn(4,3,256,256)
    # # s = torch.randn(4,662)
    # x = m(x)
    # print(x)