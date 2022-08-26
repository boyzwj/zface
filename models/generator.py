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


# class AdaIn(nn.Module):
#     def __init__(self, in_channel, vector_size):
#         super(AdaIn, self).__init__()
#         self.eps = 1e-5
#         self.std_style_fc = nn.Linear(vector_size, in_channel)
#         self.mean_style_fc = nn.Linear(vector_size, in_channel)

#     def forward(self, x, style_vector):
#         std_style = self.std_style_fc(style_vector)
#         mean_style = self.mean_style_fc(style_vector)

#         std_style = std_style.unsqueeze(-1).unsqueeze(-1)
#         mean_style = mean_style.unsqueeze(-1).unsqueeze(-1)

#         x = F.instance_norm(x)
#         x = std_style * x + mean_style
#         return x


class AdaIn(nn.Module):
    def __init__(self,in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel, affine=False)
        self.fc = nn.Linear(style_dim, in_channel*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta
    
    
class AdaInResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, up_sample=False, style_dim=662,activation='lrelu'):
        super(AdaInResBlock, self).__init__()
        self.ada_in1 = AdaIn(in_channel, style_dim)
        self.ada_in2 = AdaIn(out_channel, style_dim)
        self.activ = set_activate_layer(activation)
        main_module_list = []
        main_module_list += [
            set_activate_layer(activation),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1,bias=False),
        ]
        if up_sample:
            main_module_list.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        self.main_path1 = nn.Sequential(*main_module_list)

        self.main_path2 = nn.Sequential(
            set_activate_layer(activation),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1,bias=False)
        )

        side_module_list = [nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0,bias=False)]
        if up_sample:
            side_module_list.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        self.side_path = nn.Sequential(*side_module_list)


    def forward(self, x, id_vector):
        x1 = self.ada_in1(x, id_vector)
        x1 = self.activ(x1)
        x1 = self.main_path1(x1)
        x1 = self.ada_in2(x1, id_vector)
        x1 = self.activ(x1)
        x1 = self.main_path2(x1)
        x2 = self.side_path(x)
        return x1 + x2
        
        

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
        self.F_id = iresnet100(pretrained=False, fp16=True)
        self.F_id.load_state_dict(torch.load('./weights/backbone_r100.pth'))
        self.F_id.eval()
        
        for param in self.F_id.parameters():
            param.requires_grad = False
        self.net_recon = ReconNet()
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
                           coeff_dict_fuse["angle"],
                           coeff_dict_fuse["trans"]
                    ], dim=1)
        return v_sid, coeff_dict_fuse, id_source
    

    def get_id(self, I):
        v_id = self.F_id(F.interpolate(I, size=112, mode='bilinear'))
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




class Encoder(nn.Module):
    def __init__(self, norm='in', activation='lrelu',size = 256):
        super(Encoder, self).__init__()
        self.conv_first =   nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(64, 128, down_sample=True,attention=True,activation=activation)  #128
        self.ResBlock2 = ResBlock(128, 256, down_sample=True,attention=False, activation=activation) #64
        self.ResBlock3 = ResBlock(256, 256, down_sample=True,attention=True, activation=activation) #32
        self.ResBlock4 = ResBlock(256, 256, down_sample=True,attention=False, activation=activation) #16
        self.ResBlock5 = ResBlock(256, 512, down_sample=True,attention=True, activation=activation) #8
        self.ResBlock6 = ResBlock(512, 512, down_sample=False,attention=False, activation=activation)
        self.ResBlock7 = ResBlock(512, 512, down_sample=False,attention=True, activation=activation)
        self.skip = ResBlock(256, 256, down_sample=False,attention=True, activation=activation)
        self.apply(weight_init)

  
    def forward(self, x):
        x = self.conv_first(x)# 64x256x256
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
    def __init__(self, style_dim=662, activation='lrelu'):
        super(Decoder, self).__init__()
        self.d1 = AdaInResBlock(512, 512, up_sample=False, style_dim=style_dim,activation=activation)
        self.d2 = AdaInResBlock(512, 512, up_sample=False, style_dim=style_dim,activation=activation)
        self.d3 = AdaInResBlock(512, 384, up_sample=True, style_dim=style_dim,activation=activation)
        self.d4 = AdaInResBlock(384, 384, up_sample=True, style_dim=style_dim,activation=activation)
        self.d5 = AdaInResBlock(384, 256, up_sample=True, style_dim=style_dim,activation=activation)
        self.apply(weight_init)

    def forward(self, x, s):
        x = self.d1(x,s)
        x = self.d2(x,s)
        x = self.d3(x,s)
        x = self.d4(x,s)
        x = self.d5(x,s)
        return x

class AE(nn.Module):
    def __init__(self, style_dim=662, activation='lrelu'):
        super(AE, self).__init__()
        self.encoder = Encoder(activation=activation)
        self.decoder = Decoder(activation=activation,style_dim=style_dim)

    def forward(self,I_t,v_sid):
        z_latent, z_enc = self.encoder(I_t)
        z_dec = self.decoder(z_latent, v_sid)
        return z_enc,z_dec




class F_up(nn.Module):
    def __init__(self, style_dim,activation = 'lrelu'):
        super(F_up, self).__init__()
        self.block1 = GenResBlk(256, 64, up_sample = True, style_dim=style_dim,return_rgb=True,activation=activation)
        self.block2 = GenResBlk(64, 32, up_sample = True, style_dim=style_dim,return_rgb=True,activation=activation)
        self.block3 = GenResBlk(32, 1, up_sample = False, style_dim=style_dim,return_rgb=True,activation=activation)
    def forward(self, x, s,rgb = None):
        x, rgb = self.block1(x, s,rgb)
        x, rgb = self.block2(x, s,rgb)
        m_r, i_r = self.block3(x, s,rgb)
        m_r = torch.sigmoid(m_r)
        return i_r, m_r




class SemanticFacialFusionModule(nn.Module):
    def __init__(self, norm='in', activation='lrelu', style_dim=662):
        super(SemanticFacialFusionModule, self).__init__()
        self.z_fuse_block_n  = GenResBlk(256, 256, up_sample=False, style_dim=style_dim,return_rgb = True,activation=activation)
        self.f_up_n = F_up(style_dim=style_dim,activation=activation)
        self.segmentation_net = torch.jit.load('./weights/face_parsing.farl.lapa.main_ema_136500_jit191.pt', map_location="cuda")
        self.segmentation_net.eval()
        for param in self.segmentation_net.parameters():
            param.requires_grad = False
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5))

    def get_mask(self,I):
        with torch.no_grad():
            size = I.size()[-1]
            I = unnormalize(I)
            logit , _  = self.segmentation_net(F.interpolate(I, size=(448,448), mode='bilinear'))
            parsing = logit.max(1)[1]
            face_mask = torch.where((parsing>0)&(parsing<10), 1, 0)
            face_mask = F.interpolate(face_mask.unsqueeze(1).float(), size=(size,size), mode='nearest')
            face_mask = self.blur(face_mask)
        return face_mask
    
    
    def forward(self, target_image, z_enc, z_dec, id_vector):
        mask_high = self.get_mask(target_image).detach()
        mask_low = F.interpolate(mask_high, scale_factor=0.25,mode='bilinear')
        z_fuse = mask_low * z_dec + (1 - mask_low) * z_enc
        z_fuse,i_low = self.z_fuse_block_n(z_fuse, id_vector)
        i_r, out_mask = self.f_up_n(z_fuse,id_vector,i_low)
        i_r = mask_high * i_r + (1 - mask_high) * target_image
        i_low = mask_low * i_low + (1 - mask_low) * F.interpolate(target_image, scale_factor=0.25,mode='bilinear')
        return i_r, i_low ,out_mask,mask_high


class HififaceGenerator(nn.Module):
    def __init__(self,activation =  'lrelu', size = 256):
        super(HififaceGenerator, self).__init__()
        self.SAIE = ShapeAwareIdentityExtractor()
        self.SFFM = SemanticFacialFusionModule(activation=activation)
        self.AE = torch.jit.script(AE(activation=activation))

    @torch.no_grad()
    def inference(self,I_s,I_t):
        v_sid, _coeff_dict_fuse, _ = self.SAIE(I_s, I_t)
        z_enc,z_dec = self.AE(I_t,v_sid)
        I_swapped_high = self.SFFM(I_t,z_enc, z_dec, v_sid)[0]
        return I_swapped_high
        
    def forward(self, I_s, I_t):
        
        # 3D Shape-Aware Identity Extractor
        v_sid, coeff_dict_fuse,id_source = self.SAIE(I_s, I_t)
        # AutoEncoder
        z_enc,z_dec = self.AE(I_t,v_sid)
        # Semantic Facial Fusion Module
        I_swapped_high, I_swapped_low, mask_high,mask_target = self.SFFM(I_t,z_enc, z_dec, v_sid)
        return I_swapped_high, I_swapped_low,mask_high,mask_target, coeff_dict_fuse,id_source
    
    
if __name__ == "__main__":
    net = HififaceGenerator()
    print(net)    