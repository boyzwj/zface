from cgi import print_arguments
import imp
from tkinter.messagebox import NO
from numpy import block
import torch
from torch import nn
import torch.nn.functional as F
from models.bfm import ParametricFaceModel
from models.reconnet import ReconNet
from torchvision import transforms
from models.face_models.iresnet import iresnet100,iresnet50
from models.faceparser import BiSeNet
from models.activation import *
import math
from models.modulated_conv2d import  RGBBlock,Conv2DMod
from models.cbam import CBAM
from models.ca import CoordAtt
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


def exists(val):
    return val is not None

def set_activate_layer(types):
    # initialize activation
    if types == 'relu':
        activation = nn.ReLU()
    elif types == 'relu6':
        activation = nn.ReLU6()
    elif types == 'lrelu':
        activation = nn.LeakyReLU(0.2,inplace=True)
    elif types == 'mish':
        activation = nn.Mish(inplace=True)
        # activation = MemoryEfficientMish()
    elif types ==  "swish":
        activation = nn.SiLU(inplace=True)
    elif types == 'tanh':
        activation = nn.Tanh()
    elif types == 'sig':
        activation = nn.Sigmoid()
    elif types == 'none':
        activation = None
    else:
        assert 0, f"Unsupported activation: {types}"
    return activation




        
        

class GenResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=662, 
                 activation='lrelu', up_sample=False,return_rgb=False):
        super().__init__()
        self.actv = set_activate_layer(activation)
        if up_sample:
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up_sample = nn.Identity()
        self.needSkipConvolution = dim_in != dim_out
        self.conv1 = Conv2DMod(dim_in, dim_out, 3, stride=1, dilation=1)
        self.conv2 = Conv2DMod(dim_out, dim_out, 3, stride=1, dilation=1)
        self.style1 = nn.Linear(style_dim, dim_in)
        self.style2 = nn.Linear(style_dim, dim_out)
        if self.needSkipConvolution:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        self.toRGB = RGBBlock(style_dim, dim_out, up_sample, 3) if return_rgb else None

    def forward(self, x, s ,rgb = None):
        x = self.up_sample(x)
        if self.needSkipConvolution:
            x_ = self.conv1x1(x)
        else:
            x_ = x
        s1 = self.style1(s)
        x = self.conv1(x, s1)
        x = self.actv(x)
        s2 = self.style2(s)
        x = self.conv2(x, s2)
        x = self.actv(x + x_)
        if exists(self.toRGB):
            rgb = self.toRGB(x,s, rgb)
            return x, rgb
        else:
            return x    


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample=False, up_sample=False,attention = False,activation='lrelu'):
        super(ResBlock, self).__init__()

        main_module_list = []
        main_module_list += [
            nn.InstanceNorm2d(in_channel),
            set_activate_layer(activation),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
            ]
        if down_sample:
            main_module_list.append(nn.AvgPool2d(kernel_size=2))
        elif up_sample:
            main_module_list.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        main_module_list += [
            nn.InstanceNorm2d(out_channel),
            set_activate_layer(activation),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        ]
        if attention:
             main_module_list += [
                 CBAM(out_channel,2,3)
             ]
             
        self.main_path = nn.Sequential(*main_module_list)

        side_module_list = [nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=True)]
        if down_sample:
            side_module_list.append(nn.AvgPool2d(kernel_size=2))
        elif up_sample:
            side_module_list.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.side_path = nn.Sequential(*side_module_list)

    def forward(self, x):
        x1 = self.main_path(x)
        x2 = self.side_path(x)
        return (x1 + x2) / math.sqrt(2)
    

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
        
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)




class ShapeAwareIdentityExtractor(nn.Module):
    def __init__(self):
        super(ShapeAwareIdentityExtractor, self).__init__()
        self.F_id = iresnet50(pretrained=False, fp16=True)
        self.F_id.load_state_dict(torch.load('./weights/backbone_r50.pth'))
        self.F_id.eval()
        
        for param in self.F_id.parameters():
            param.requires_grad = False
        self.net_recon = ReconNet()
        self.net_recon.load_state_dict(torch.load('./weights/epoch_20.pth')['net_recon'])
        self.net_recon.eval()
        for param in self.net_recon.parameters():
            param.requires_grad = False

        self.facemodel = ParametricFaceModel(is_train=False)
 

    @torch.no_grad()
    def forward(self, I_s, I_t):
        # id of Is
        with torch.no_grad():
            id_source = self.get_id(I_s)

            # 3d params of Is
            coeff_dict_s = self.get_coeff3d(I_s)

            # 3d params of It
            coeff_dict_t = self.get_coeff3d(I_t)

        # fused 3d parms
        coeff_dict_fuse = coeff_dict_t.copy()
        coeff_dict_fuse["id"] = coeff_dict_s["id"]

        # concat all to obtain the 3D shape-aware identity(v_sid)
        v_sid = torch.cat([id_source, 
                           coeff_dict_fuse["id"],
                           coeff_dict_fuse["exp"],
                           coeff_dict_fuse["angle"],
                           coeff_dict_fuse["trans"]
                    ], dim=1)
        return v_sid, coeff_dict_fuse,id_source
    
    @torch.no_grad()
    def get_id(self, I):
        v_id = self.F_id(F.interpolate(I, size=112, mode='bilinear'))
        v_id = F.normalize(v_id)
        return v_id


    @torch.no_grad()
    def get_coeff3d(self, I):
        coeffs = self.net_recon(F.interpolate(I * 0.5 + 0.5, size=224, mode='bilinear'))
        coeff_dict = self.facemodel.split_coeff(coeffs)
        return coeff_dict

    @torch.no_grad()
    def get_lm3d(self, coeff_dict):
        # get 68 3d landmarks
        face_shape = self.facemodel.compute_shape(coeff_dict['id'], coeff_dict['exp'])
        rotation = self.facemodel.compute_rotation(coeff_dict['angle'])

        face_shape_transformed = self.facemodel.transform(face_shape, rotation, coeff_dict['trans'])
        face_vertex = self.facemodel.to_camera(face_shape_transformed)
        
        face_proj = self.facemodel.to_image(face_vertex)
        lm3d = self.facemodel.get_landmarks(face_proj)

        return lm3d




class Encoder(nn.Module):
    def __init__(self, norm='in', activation='lrelu',size = 256):
        super(Encoder, self).__init__()
        self.FirstConv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(64, 128, down_sample=True,activation=activation)
        self.ResBlock2 = ResBlock(128, 256, down_sample=True,attention=True, activation=activation)
        self.ResBlock3 = ResBlock(256, 512, down_sample=True,attention=True, activation=activation)
        self.ResBlock4 = ResBlock(512, 512, down_sample=True,attention=True, activation=activation)
        self.ResBlock5 = ResBlock(512, 512, down_sample=True,attention=True, activation=activation)
        self.ResBlock6 = ResBlock(512, 512, down_sample=False,attention=True, activation=activation)
        self.ResBlock7 = ResBlock(512, 512, down_sample=False,attention=True, activation=activation)
        self.skip = ResBlock(256, 256, down_sample=False,attention=True, activation=activation)
        self.apply(weight_init)
        
        
        
    def forward(self, x):
        x = self.FirstConv(x) # 64x256x256
        x = self.ResBlock1(x) # 32x128x128
        x = self.ResBlock2(x) # 64x64x64
        y = self.ResBlock3(x) # 128x32x32
        y = self.ResBlock4(y) # 256x16xx16
        y = self.ResBlock5(y) # 512x8x8
        y = self.ResBlock6(y) # 1024x4x4
        y = self.ResBlock7(y) # 1024x4x4
        z = self.skip(x)
        return y, z


class Decoder(nn.Module):
    def __init__(self, styledim=662, activation='lrelu'):
        super(Decoder, self).__init__()
        self.d1 = GenResBlk(512, 512, up_sample=False, style_dim=styledim,activation=activation)
        self.d2 = GenResBlk(512, 512, up_sample=False, style_dim=styledim,activation=activation)
        self.d3 = GenResBlk(512, 512, up_sample=True, style_dim=styledim,activation=activation)
        self.d4 = GenResBlk(512, 512, up_sample=True, style_dim=styledim,activation=activation)
        self.d5 = GenResBlk(512, 256, up_sample=True, style_dim=styledim,activation=activation)
        self.apply(weight_init)

    def forward(self, x, s):
        x = self.d1(x,s)
        x = self.d2(x,s)
        x = self.d3(x,s)
        x = self.d4(x,s)
        x = self.d5(x,s)
        return x



class F_up(nn.Module):
    def __init__(self, styledim,activation = 'lrelu'):
        super(F_up, self).__init__()
        self.block1 = GenResBlk(256, 64, up_sample = True, style_dim=styledim,return_rgb=True,activation=activation)
        self.block2 = GenResBlk(64, 16, up_sample = True, style_dim=styledim,return_rgb=True,activation=activation)
        self.block3 = GenResBlk(16, 1, up_sample = False, style_dim=styledim,return_rgb=True,activation=activation)
    def forward(self, x, s,rgb = None):
        x, rgb = self.block1(x, s,rgb)
        x, rgb = self.block2(x, s,rgb)
        m_r, i_r = self.block3(x, s,rgb)
        m_r = torch.tanh(m_r)
        return i_r, m_r




class SemanticFacialFusionModule(nn.Module):
    def __init__(self, norm='in', activation='lrelu', styledim=662):
        super(SemanticFacialFusionModule, self).__init__()
        self.sigma = ResBlock(256,256,activation=activation)
        self.low_mask_predict = ResBlock(256,1,activation=activation)
        self.z_fuse_block_n  = GenResBlk(256, 256, up_sample=False, style_dim=styledim,return_rgb = True,activation=activation)
        self.f_up_n = F_up(styledim=styledim,activation=activation)

        




    def forward(self, target_image, z_enc, z_dec, id_vector):
        z_enc = self.sigma(z_enc)
        m_low = self.low_mask_predict(z_dec)
        m_low = torch.tanh(m_low)
        z_fuse = m_low * z_dec + (1 - m_low) * z_enc
        z_fuse,i_low = self.z_fuse_block_n(z_fuse, id_vector)
        i_r, m_r = self.f_up_n(z_fuse,id_vector,i_low)
        i_r = m_r * i_r + (1 - m_r) * target_image
        i_low = m_low * i_low + (1 - m_low) * F.interpolate(target_image, scale_factor=0.25)
        return i_r, i_low, m_r, m_low


class HififaceGenerator(nn.Module):
    def __init__(self,activation =  'lrelu', size = 256):
        super(HififaceGenerator, self).__init__()
        self.SAIE = ShapeAwareIdentityExtractor()
        self.SFFM = SemanticFacialFusionModule(activation=activation)
        self.E = torch.jit.script(Encoder(activation=activation))
        self.D = Decoder(activation=activation)

    @torch.no_grad()
    def inference(self,I_s,I_t):
        v_sid, _coeff_dict_fuse, _id_source = self.SAIE(I_s, I_t)
        z_latent, z_enc = self.E(I_t)
        z_dec = self.D(z_latent, v_sid)
        I_swapped_high = self.SFFM(I_t,z_enc, z_dec, v_sid)[0]
        return I_swapped_high
        
    def forward(self, I_s, I_t):
        
        # 3D Shape-Aware Identity Extractor
        v_sid, coeff_dict_fuse,id_source = self.SAIE(I_s, I_t)
        
        # Encoder
        z_latent, z_enc = self.E(I_t)  #z_latent 目标深度隐含变量   z_enc 是目标的浅层隐含变量

        # Decoder
        z_dec = self.D(z_latent, v_sid)

        # Semantic Facial Fusion Module
        I_swapped_high, I_swapped_low, mask_high, mask_low= self.SFFM(I_t,z_enc, z_dec, v_sid)
        
        return I_swapped_high, I_swapped_low, mask_high, mask_low, coeff_dict_fuse ,id_source
    
    
if __name__ == "__main__":
    net = HififaceGenerator()
    print(net)    