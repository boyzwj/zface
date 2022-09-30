from turtle import forward
from models.ca import CoordAtt
from models.modulated_conv2d import Conv2DMod,StyledConv2d
from models.gen2 import  GenResBlk,Decoder,Encoder,ShapeAwareIdentityExtractor
from models.discriminator import  ProjectedDiscriminator
import torch
import torch.nn as nn
from models.face_models.iresnet import iresnet100
from models.bfm import ParametricFaceModel
from models.reconnet import ReconNet
from torch_utils import training_stats
import numpy as np
import timm



def get_losses_weights(losses):
    weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
    return weights



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        model = timm.create_model('maxvit_rmlp_tiny_rw_256',scriptable=True) 
        self.layer0 =  nn.Sequential(
                model.stem, model.stages[0]
            )
        self.layer1 = model.stages[1]

        self.layer2 = model.stages[2]
        self.layer3 = model.stages[3]
    def forward(self,x):
        x = self.layer0(x)
        print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


            


if __name__ == '__main__':
    model = Encoder()
    # half = torch.randn(10, 10, dtype=torch.float16, device='cuda')
    # const = torch.ones(10, 10, device='cuda')
    # inputs = torch.randn(10, 10, device='cuda')
    x = torch.randn(4,3,256,256)
    out = model(x)
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