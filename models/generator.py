import imp
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
from models.modulated_conv2d import Conv2DMod, RGBBlock

def exists(val):
    return val is not None

def set_activate_layer(types):
    # initialize activation
    if types == 'relu':
        activation = nn.ReLU()
    elif types == 'lrelu':
        activation = nn.LeakyReLU(0.2,inplace=True)
    elif types == 'mish':
        activation = MemoryEfficientMish()
    elif types ==  "swish":
        activation = Swish()
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
    def __init__(self, dim_in, dim_out, style_dim=64, 
                 activation='lrelu', upsample=False,return_rgb=False):
        super().__init__()
        self.actv = set_activate_layer(activation)
        self.upsample = upsample
        self.needSkipConvolution = dim_in != dim_out
        self.conv1 = Conv2DMod(dim_in, dim_out, 3, stride=1, dilation=1)
        self.conv2 = Conv2DMod(dim_out, dim_out, 3, stride=1, dilation=1)
        self.style1 = nn.Linear(style_dim, dim_in)
        self.style2 = nn.Linear(style_dim, dim_out)
        if self.needSkipConvolution:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        self.toRGB = RGBBlock(style_dim, dim_out, upsample, 3) if return_rgb else None

    def forward(self, x, s ,rgb = None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
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
    def __init__(self, in_c, out_c, downsample=False, norm='in', activation='lrelu'):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.norm1 = nn.InstanceNorm2d(out_c)
        self.norm2 = nn.InstanceNorm2d(out_c)
        self.activ = set_activate_layer(activation)
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, feat):
        feat1 = self.norm1(feat)
        feat1 = self.activ(feat1)
        feat1 = self.conv1(feat1)
        if self.downsample:
            feat1 = F.interpolate(feat1, scale_factor=0.5, mode='bilinear', align_corners=False)
        feat1 = self.norm2(feat1)
        feat1 = self.activ(feat1)
        feat1 = self.conv2(feat1)
        feat2 = self.conv1x1(feat)
        if self.downsample:
            feat2 = F.interpolate(feat2, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = feat1 + feat2
        return x / math.sqrt(2)  


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
 

    def forward(self, I_s, I_t):
        # id of Is
        with torch.no_grad():
            v_id = self.get_id(I_s)

            # 3d params of Is
            coeff_dict_s = self.get_coeff3d(I_s)

            # 3d params of It
            coeff_dict_t = self.get_coeff3d(I_t)

        # fused 3d parms
        coeff_dict_fuse = coeff_dict_t.copy()
        coeff_dict_fuse["id"] = coeff_dict_s["id"]

        # concat all to obtain the 3D shape-aware identity(v_sid)
        v_sid = torch.cat([v_id, 
                           coeff_dict_fuse["id"],
                           coeff_dict_fuse["exp"],
                           coeff_dict_fuse["angle"]
                    ], dim=1)
        return v_sid, coeff_dict_fuse

    def get_id(self, I):
        v_id = self.F_id(F.interpolate(I[:, :, 16:240, 16:240], [112,112], mode='bilinear', align_corners=True))
        v_id = F.normalize(v_id)
        return v_id

    def get_coeff3d(self, I):
        coeffs = self.net_recon(I[:, :, 16:240, 16:240]*0.5+0.5)
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
    def __init__(self, norm='in', activation='lrelu'):
        super(Encoder, self).__init__()
        self.InitConv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(64, 128,  downsample=True, norm=norm, activation=activation)
        self.ResBlock2 = ResBlock(128, 256, downsample=True, norm=norm, activation=activation)
        self.ResBlock3 = ResBlock(256, 512, downsample=True, norm=norm, activation=activation)
        self.ResBlock4 = ResBlock(512, 512, downsample=True, norm=norm, activation=activation)
        self.ResBlock5 = ResBlock(512, 512, downsample=True, norm=norm, activation=activation)
        self.ResBlock6 = ResBlock(512, 512, downsample=False, norm=norm, activation=activation)
        self.ResBlock7 = ResBlock(512, 512, downsample=False, norm=norm, activation=activation)
        self.skip = ResBlock(256, 256, downsample=False, norm=norm, activation=activation)
        self.apply(weight_init)
        
        
        
    def forward(self, x):
        x = self.InitConv(x) # 64x256x256
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
    def __init__(self, styledim=659, activation='lrelu'):
        super(Decoder, self).__init__()
        self.net = nn.ModuleList([
            GenResBlk(512, 512, upsample=False, style_dim=styledim,activation=activation),
            GenResBlk(512, 512, upsample=False, style_dim=styledim,activation=activation),
            # GenResBlk(512, 512, upsample=True, style_dim=styledim,activation=activation),
            GenResBlk(512, 512, upsample=True, style_dim=styledim,activation=activation),
            GenResBlk(512, 512, upsample=True, style_dim=styledim,activation=activation),
            GenResBlk(512, 256, upsample=True, style_dim=styledim,activation=activation),
        ])
        self.apply(weight_init)

    def forward(self, attr, s):
        y = attr
        for i in range(5):
            y = self.net[i](y,s)
        return y




class F_up(nn.Module):
    def __init__(self, styledim,activation = 'lrelu'):
        super(F_up, self).__init__()
        self.block1 = GenResBlk(256, 64, upsample = True, style_dim=styledim,return_rgb=True,activation=activation)
        self.block2 = GenResBlk(64, 64, upsample = True, style_dim=styledim,return_rgb=True,activation=activation)
        self.block3 = GenResBlk(64, 16, upsample = False, style_dim=styledim,return_rgb=True,activation=activation)
    def forward(self, x, s,rgb = None):
        x, rgb = self.block1(x, s,rgb)
        x, rgb = self.block2(x, s,rgb)
        x, rgb = self.block3(x, s,rgb)
        return rgb, x




class SemanticFacialFusionModule(nn.Module):
    def __init__(self, norm='in', activation='lrelu', styledim=659):
        super(SemanticFacialFusionModule, self).__init__()


        self.AdaINResBlock = GenResBlk(256, 259, upsample=False, style_dim=styledim,return_rgb = False,activation=activation)
        
        
        self.F_up = F_up(styledim=styledim,activation=activation)

        self.face_pool = nn.AdaptiveAvgPool2d((64, 64)).eval()

        # face Segmentation model: HRNet [Sun et al., 2019]
        self.segmentation_net = BiSeNet(n_classes=19).to('cuda')
        self.segmentation_net.load_state_dict(torch.load('./weights/faceparser.pth', map_location="cuda"))
        self.segmentation_net.eval()
        for param in self.segmentation_net.parameters():
            param.requires_grad = False
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5))

    def get_mask(self, I):
        with torch.no_grad():
            size = I.size()[-1]
            parsing = self.segmentation_net(F.interpolate(I, size=(512,512), mode='bilinear', align_corners=True)).max(1)[1]
            mask = torch.where((parsing>0)&(parsing<14), 1, 0)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(size,size), mode='nearest')
            mask = self.blur(mask)
        return mask




    def forward(self, z_enc, z_dec, v_sid, I_target):

        M_high = self.get_mask(I_target).detach()
        M_low = self.face_pool(M_high)


        # z_enc 256 64 64
        # z_dec 256 64 64
        
        # z_fuse 256 64 64
        z_fuse = z_dec * M_low.repeat(1, 256, 1, 1) + z_enc * (1-M_low.repeat(1, 256, 1, 1))
        
        
        I_out_low  = self.AdaINResBlock(z_fuse, v_sid) 

        # I_low 3 64 64
        # I_swapped_low = I_out_low[:,:3,...] * M_low.repeat(1, 3, 1, 1) + self.face_pool(I_target) * (1-M_low.repeat(1, 3, 1, 1))

        # I_out_high 3 256 256
        I_out_high, _  = self.F_up(I_out_low[:,3:,...],v_sid)

        # I_r 3 256 256
        # I_swapped_high = I_out_high * M_high.repeat(1, 3, 1, 1) + I_target * (1-M_high.repeat(1, 3, 1, 1))
        I_swapped_high = I_out_high
        I_swapped_low = I_out_low[:,:3,...]
        return I_swapped_high, I_swapped_low


class HififaceGenerator(nn.Module):
    def __init__(self,activation =  'lrelu'):
        super(HififaceGenerator, self).__init__()
        
        self.SAIE = ShapeAwareIdentityExtractor()
        self.SFFM = SemanticFacialFusionModule(activation=activation)
        self.E = Encoder(activation=activation)
        self.D = Decoder(activation=activation)


    def forward(self, I_s, I_t):
        
        # 3D Shape-Aware Identity Extractor
        v_sid, coeff_dict_fuse = self.SAIE(I_s, I_t)
        
        # Encoder
        z_latent, z_enc = self.E(I_t)  #z_latent 目标深度隐含变量   z_enc 是目标的浅层隐含变量

        # Decoder
        z_dec = self.D(z_latent, v_sid)

        # Semantic Facial Fusion Module
        I_swapped_high, I_swapped_low = self.SFFM(z_enc, z_dec, v_sid, I_t)
        
        return I_swapped_high, I_swapped_low, coeff_dict_fuse
    
    
if __name__ == "__main__":
    net = HififaceGenerator()
    print(net)    