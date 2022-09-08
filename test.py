from models.ca import CoordAtt
from models.modulated_conv2d import Conv2DMod,StyledConv2d
from models.gen2 import  GenResBlk,Decoder,Encoder,ShapeAwareIdentityExtractor
import torch
from models.face_models.iresnet import iresnet100
from models.bfm import ParametricFaceModel
from models.reconnet import ReconNet


def get_losses_weights(losses):
    weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
    return weights




if __name__ == '__main__':
    losses = torch.tensor([1,2,3,4,10]) 
    weight = get_losses_weights(losses)
    print(weight)
    new_losses =  losses / weight
    print(new_losses)

    # F_id = torch.jit.script(ParametricFaceModel(is_train=False))
    # F_id.load_state_dict(torch.load('./weights/backbone_r100.pth'))
    # F_id.eval()
    # m = torch.jit.script(Encoder())
    # # m = GenResBlk(128,64)
    # x = torch.randn(4,3,256,256)
    # # s = torch.randn(4,662)
    # x = m(x)
    # print(x)