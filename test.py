from models.ca import CoordAtt
from models.modulated_conv2d import Conv2DMod,StyledConv2d
from models.gen2 import  GenResBlk,Decoder,Encoder,ShapeAwareIdentityExtractor
import torch
from models.face_models.iresnet import iresnet100
from models.bfm import ParametricFaceModel
from models.reconnet import ReconNet







if __name__ == '__main__':
    a = torch.tensor([1,2,3,4]) 
    b = torch.tensor([5,0,2,2])
    print(a * b)

    # F_id = torch.jit.script(ParametricFaceModel(is_train=False))
    # F_id.load_state_dict(torch.load('./weights/backbone_r100.pth'))
    # F_id.eval()
    # m = torch.jit.script(Encoder())
    # # m = GenResBlk(128,64)
    # x = torch.randn(4,3,256,256)
    # # s = torch.randn(4,662)
    # x = m(x)
    # print(x)