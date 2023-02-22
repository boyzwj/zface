from cgi import print_arguments
import imp
from tkinter.messagebox import NO
from numpy import block
import torch
from torch import nn, einsum
import torch.nn.functional as F
from models.bfm import ParametricFaceModel
from models.reconnet import ReconNet
from torchvision import transforms
from models.face_models.iresnet import iresnet100,iresnet50
from models.faceparser import BiSeNet
from models.activation import *
import math
from models.modulated_conv2d import  RGBBlock,Conv2DMod,Blur,StyledConv2d
from models.cbam import CBAM
from models.ca import CoordAtt, ECA
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


def exists(val):
    return val is not None

def set_activate_layer(types):
    # initialize activation
    if types == 'relu':
        activation = nn.ReLU(inplace=True)
    elif types == 'relu6':
        activation = nn.ReLU6(inplace=True)
    elif types == 'lrelu':
        activation = nn.LeakyReLU(0.2,inplace=True)
    elif types == 'mish':
        activation = nn.Mish(inplace=True)
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



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x
    
    
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)
class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.nonlin = nn.Mish(inplace=True)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)
    
class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))
    
    
# attn_and_ff = lambda chan: nn.Sequential(*[
#     Residual(PreNorm(chan, LinearAttention(chan))),
#     Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), nn.Mish(inplace=True), nn.Conv2d(chan * 2, chan, 1))))
# ])





        
        

class GenResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=659, 
                 activation='lrelu', up_sample=False,return_rgb=False):
        super().__init__()
        self.actv = set_activate_layer(activation)
        if up_sample:
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up_sample = nn.Identity()
        self.conv1 = Conv2DMod(dim_in, dim_out, 3, stride=1, dilation=1)
        self.conv2 = Conv2DMod(dim_out, dim_out, 3, stride=1, dilation=1)
        self.style1 = nn.Linear(style_dim, dim_in)
        self.style2 = nn.Linear(style_dim, dim_out)
        if dim_in != dim_out:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        else:
            self.conv1x1 = nn.Identity()
        self.toRGB = RGBBlock(style_dim, dim_out, up_sample, 3) if return_rgb else None

    def forward(self, x, s ,rgb = None):
        x = self.up_sample(x)
        x_ = self.conv1x1(x)
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


    
# class ResBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, down_sample=False, up_sample=False,attention = False,activation='lrelu'):
#         super(ResBlock, self).__init__()     
#         main_module_list = []
#         main_module_list += [
#                 nn.InstanceNorm2d(in_channel),
#                 set_activate_layer(activation),
#                 nn.Conv2d(in_channel,in_channel, 3, 1, 1),
#             ]
#         if down_sample:
#             main_module_list.append(nn.AvgPool2d(kernel_size=2))
#         elif up_sample:
#             main_module_list += [
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#                 ]

#         main_module_list += [
#                 nn.InstanceNorm2d(in_channel),
#                 set_activate_layer(activation),
#                 nn.Conv2d(in_channel,out_channel, 3, 1, 1)
#             ]            
#         if attention:
#              main_module_list += [
#                  ECA(out_channel)
#                 #  CoordAtt(out_channel,out_channel)
#              ]
#         self.main_path = nn.Sequential(*main_module_list)
#         side_module_list = []
#         if in_channel != out_channel:
#             side_module_list += [nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)]
#         else:
#             side_module_list += [nn.Identity()]   
#         if down_sample:
#             side_module_list.append(nn.AvgPool2d(kernel_size=2))
#         elif up_sample:
#             side_module_list += [
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#                 ]
#         self.side_path = nn.Sequential(*side_module_list)

#     def forward(self, x):
#         x1 = self.main_path(x)
#         x2 = self.side_path(x)
#         return (x1 + x2) / math.sqrt(2)
    
    
    
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
        self.F_id100 = torch.jit.script(iresnet100(pretrained=False, fp16=True))
        # self.F_id100.load_state_dict(torch.load('./weights/backbone_r100.pth'))
        self.F_id100.load_state_dict(torch.load('./weights/r100.pth'))
        self.F_id100.eval()
        
        for param in self.F_id100.parameters():
            param.requires_grad = False
        self.net_recon = torch.jit.script(ReconNet())
        self.net_recon.load_state_dict(torch.load('./weights/epoch_20.pth')['net_recon'])
        self.net_recon.eval()
        for param in self.net_recon.parameters():
            param.requires_grad = False

        self.facemodel = ParametricFaceModel(is_train=False)
 

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
                           coeff_dict_fuse["angle"]
                    ], dim=1)
        return v_sid, coeff_dict_fuse, id_source
    

    def get_id(self, I):
        v_id = self.F_id100(F.interpolate(I, size=112, mode='bilinear'))
        v_id = F.normalize(v_id)
        return v_id


    def get_coeff3d(self, I):
        coeffs = self.net_recon(F.interpolate(I * 0.5 + 0.5, size=224, mode='bilinear'))
        coeff_dict = self.facemodel.split_coeff(coeffs)
        return coeff_dict

    def get_lm3d(self, coeff_dict):
        # get 68 3d landmarks
        face_shape = self.facemodel.compute_shape(coeff_dict['id'], coeff_dict['exp'])
        rotation = self.facemodel.compute_rotation(coeff_dict['angle'])

        face_shape_transformed = self.facemodel.transform(face_shape, rotation, coeff_dict['trans'])
        face_vertex = self.facemodel.to_camera(face_shape_transformed)
        
        face_proj = self.facemodel.to_image(face_vertex)
        lm3d = self.facemodel.get_landmarks(face_proj)

        return lm3d


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x




class Encoder(nn.Module):
    def __init__(self, norm='in', activation='lrelu',size = 256):
        super(Encoder, self).__init__()

        self.first =  nn.Sequential(
            nn.Conv2d(3, 96, 4,4),
            LayerNorm(96,eps=1e-6, data_format="channels_first")  
            ) 

        self.b1 = nn.Sequential(
            Block(96),
            Block(96),
        )#64

        self.b2 = nn.Sequential(
            LayerNorm(96,eps=1e-6, data_format="channels_first"),
            nn.Conv2d(96, 192, kernel_size=2, stride=2),
            Block(192),
            Block(192),
        )#32

        self.b3 = nn.Sequential(
            LayerNorm(192,eps=1e-6, data_format="channels_first"),
            nn.Conv2d(192, 384, kernel_size=2, stride=2),
            Block(384),
            Block(384),
            Block(384),
            Block(384),
            Block(384),
            Block(384),
            Block(384), 
        )#16

        self.b4 = nn.Sequential(
            LayerNorm(384,eps=1e-6, data_format="channels_first"),
            nn.Conv2d(384, 768, kernel_size=2, stride=2),
            Block(768),
            Block(768),
        )#8

        self.skip = nn.Sequential(
            Block(96)
        )#8
        # self.apply(self._init_weights)
        
    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         nn.init.constant_(m.bias, 0)   
        
    def forward(self, x):
        x = self.first(x) # 64x256x256
        x = self.b1(x) # 64x128x128
        y = self.b2(x) # 16x64x64
        y = self.b3(y) # 128x32x32
        y = self.b4(y) # 256x16xx16
        z = self.skip(x)
        return y, z


class Decoder(nn.Module):
    def __init__(self, style_dim=659, activation='lrelu'):
        super(Decoder, self).__init__()
        self.d1 = GenResBlk(768, 768, up_sample=False, style_dim=style_dim,activation=activation)
        self.d2 = GenResBlk(768, 768, up_sample=False, style_dim=style_dim,activation=activation)
        self.d3 = GenResBlk(768, 384, up_sample=True, style_dim=style_dim,activation=activation)     
        self.d4 = GenResBlk(384, 192, up_sample=True, style_dim=style_dim,activation=activation)    
        self.d5 = GenResBlk(192, 96, up_sample=True, style_dim=style_dim,activation=activation)     
        self.apply(weight_init)

    def forward(self, x, s):
        x = self.d1(x,s)
        x = self.d2(x,s)
        x = self.d3(x,s)    
        x = self.d4(x,s)    
        x = self.d5(x,s)   
        return x





class FinalUp(nn.Module):
    def __init__(self, style_dim=659,activation = 'lrelu'):
        super(FinalUp, self).__init__()
        self.u1 = GenResBlk(96, 96,  up_sample=True, style_dim=style_dim,activation=activation,return_rgb=True)
        self.u2 = GenResBlk(96, 48,  up_sample=True, style_dim=style_dim,activation=activation,return_rgb=True)
        self.u3 = GenResBlk(48, 48,  up_sample=False, style_dim=style_dim,activation=activation,return_rgb=True)
        self.u4 = GenResBlk(48, 48,  up_sample=False, style_dim=style_dim,activation=activation,return_rgb=True)
        
    def forward(self, x,s,rgb = None):
        x,rgb  = self.u1(x,s,rgb)     
        x,rgb  = self.u2(x,s,rgb)    
        x,rgb  = self.u3(x,s,rgb)     
        x,rgb  = self.u4(x,s,rgb)
        return x,rgb



class SemanticFacialFusionModule(nn.Module):
    def __init__(self, norm='in', activation='lrelu', style_dim=659):
        super(SemanticFacialFusionModule, self).__init__()
        self.z_fuse_block_n  = GenResBlk(96, 96, up_sample=False, style_dim=style_dim,return_rgb = True,activation=activation)
        self.f_up_n = FinalUp(activation=activation,style_dim=style_dim)
        self.apply(weight_init)

    
    def forward(self, target_image,mask_high, z_enc, z_dec, id_vector):
        mask_low = F.interpolate(mask_high, scale_factor=0.25,mode='bilinear')
        z_fuse = mask_low * z_dec + (1 - mask_low) * z_enc
        # z_fuse = torch.cat((z_dec, z_enc),dim=1)
        z_fuse,i_low = self.z_fuse_block_n(z_fuse, id_vector)
        _ , i_r = self.f_up_n(z_fuse,id_vector)
        i_r = mask_high * i_r + (1 - mask_high) * target_image
        i_low = mask_low * i_low + (1 - mask_low) * F.interpolate(target_image, scale_factor=0.25,mode='bilinear')
        return i_r, i_low


class HififaceGenerator(nn.Module):
    def __init__(self,activation =  'lrelu', size = 256):
        super(HififaceGenerator, self).__init__()
        self.SAIE = ShapeAwareIdentityExtractor()
        self.SFFM = SemanticFacialFusionModule(activation=activation)
        self.E = Encoder(activation=activation)
        self.D = Decoder(activation=activation)

    @torch.no_grad()
    def inference(self,I_s,I_t,mask_high):
        v_sid, _coeff_dict_fuse, _ = self.SAIE(I_s, I_t)
        z_latent, z_enc = self.E(I_t)
        z_dec = self.D(z_latent, v_sid)
        I_swapped_high = self.SFFM(I_t,mask_high,z_enc, z_dec, v_sid)[0]
        return I_swapped_high
        
    def forward(self, I_s, I_t, mask_high):
        
        # 3D Shape-Aware Identity Extractor
        v_sid, coeff_dict_fuse,id_source = self.SAIE(I_s, I_t)
        
        # Encoder
        z_latent, z_enc = self.E(I_t)  #z_latent 目标深度隐含变量   z_enc 是目标的浅层隐含变量

        # Decoder
        z_dec = self.D(z_latent, v_sid)

        # Semantic Facial Fusion Module
        I_swapped_high, I_swapped_low = self.SFFM(I_t,mask_high,z_enc, z_dec, v_sid)
        
        return I_swapped_high, I_swapped_low, coeff_dict_fuse,id_source
    
    
if __name__ == "__main__":
    net = HififaceGenerator()
    print(net)    