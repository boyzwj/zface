from models.ca import CoordAtt
from models.modulated_conv2d import Conv2DMod,Upsample,ToRGB
from models.generator import  GenResBlk,Decoder,SemanticFacialFusionModule
import torch










if __name__ == '__main__':
    
    m = ToRGB(10,10)
    print(m)