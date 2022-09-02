from models.ca import CoordAtt
from models.modulated_conv2d import Conv2DMod,StyledConv2d
from models.gen2 import  GenResBlk
import torch










if __name__ == '__main__':
    m = GenResBlk(512,512)
    x = torch.randn(6,512,8,8)
    s = torch.randn(6,662)
    x = m(x,s)
    print(x.shape)