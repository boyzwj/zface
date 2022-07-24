import torch
import torch.nn as nn
import timm
from models.constants import NORMALIZED_INCEPTION, NORMALIZED_IMAGENET, NORMALIZED_CLIP, VITS
from models.vit import _make_vit_b16_backbone,forward_vit
from models.pretrained_builder import _make_pretrained

def norm_with_stats(x, stats):
    x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.5 / stats['mean'][0]) + (0.5 - stats['std'][0]) / stats['mean'][0]
    x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.5 / stats['mean'][1]) + (0.5 - stats['std'][1]) / stats['mean'][1]
    x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.5 / stats['mean'][2]) + (0.5 - stats['std'][2]) / stats['mean'][2]
    x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    return x

def get_backbone_normstats(backbone):
    if backbone in NORMALIZED_INCEPTION:
        return {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
        }

    elif backbone in NORMALIZED_IMAGENET:
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        }

    elif backbone in NORMALIZED_CLIP:
        return {
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
        }

    else:
        raise NotImplementedError

def _make_vit(model, name):
    if 'tiny' in name:
        features = [24, 48, 96, 192]
        hooks = [2, 5, 8, 11]
        vit_features = 192

    elif 'small' in name:
        features = [48, 96, 192, 384]
        hooks = [2, 5, 8, 11]
        vit_features = 384

    elif 'base' in name:
        features = [96, 192, 384, 768]
        hooks = [2, 5, 8, 11]
        vit_features = 768

    elif 'large' in name:
        features = [256, 512, 1024, 1024]
        hooks = [5, 11, 17, 23]
        vit_features = 1024

    else:
        raise NotImplementedError('Invalid ViT backbone not available')

    return _make_vit_b16_backbone(
        model,
        features=features,
        size=[224, 224],
        hooks=hooks,
        vit_features=vit_features,
        start_index=2 if 'deit' in name else 1,
    )

class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, lowest=False):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
    
    
def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    # shapes
    out_channels = [cout, cout*2, cout*4, cout*8] if expand else [cout]*4

    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)

    scratch.CHANNELS = out_channels

    return scratch


def _make_scratch_csm(scratch, in_channels, cout, expand):
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], nn.ReLU(False), expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], nn.ReLU(False))

    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout*2, cout*4] if expand else [cout]*4

    return scratch



# def calc_channels(pretrained, inp_res=224):
#     channels = []
#     tmp = torch.zeros(1, 3, inp_res, inp_res)

#     # forward pass
#     tmp = pretrained.layer0(tmp)
#     channels.append(tmp.shape[1])
#     tmp = pretrained.layer1(tmp)
#     channels.append(tmp.shape[1])
#     tmp = pretrained.layer2(tmp)
#     channels.append(tmp.shape[1])
#     tmp = pretrained.layer3(tmp)
#     channels.append(tmp.shape[1])

#     return channels


def _make_projector(im_res, backbone, cout, proj_type, expand=False):
    assert proj_type in [0, 1, 2], "Invalid projection type"

    ### Build pretrained feature network
    pretrained = _make_pretrained(backbone)

    # Following Projected GAN
    im_res = 256
    pretrained.RESOLUTIONS = [im_res//4, im_res//8, im_res//16, im_res//32]

    if proj_type == 0: return pretrained, None

    ### Build CCM
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)

    pretrained.CHANNELS = scratch.CHANNELS

    if proj_type == 1: return pretrained, scratch

    ### build CSM
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)

    # CSM upsamples x2 so the feature map resolution doubles
    pretrained.RESOLUTIONS = [res*2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch


class F_RandomProj(nn.Module):
    def __init__(
        self,
        backbone="tf_efficientnet_lite3",
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        **kwargs,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.backbone = backbone
        self.cout = cout
        self.expand = expand
        
        
        self.normstats = get_backbone_normstats(backbone)
        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(im_res=im_res, backbone=self.backbone , cout=self.cout, proj_type=self.proj_type, expand=self.expand)
        
        
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x):
        # predict feature maps
        if self.normstats is not None:
            x = norm_with_stats(x, self.normstats)
            
        # predict feature maps
        if self.backbone in VITS:
            out0, out1, out2, out3 = forward_vit(self.pretrained, x)
        else:
            out0 = self.pretrained.layer0(x)
            out1 = self.pretrained.layer1(out0)
            out2 = self.pretrained.layer2(out1)
            out3 = self.pretrained.layer3(out2)
            
        # start enumerating at the lowest layer (this is where we put the first discriminator)
        out = {
            '0': out0,
            '1': out1,
            '2': out2,
            '3': out3,
        }

        if self.proj_type == 0: return out

        out0_channel_mixed = self.scratch.layer0_ccm(out['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(out['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(out['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(out['3'])

        out = {
            '0': out0_channel_mixed,
            '1': out1_channel_mixed,
            '2': out2_channel_mixed,
            '3': out3_channel_mixed,
        }

        if self.proj_type == 1: return out

        # from bottom to top
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            '0': out0_scale_mixed,
            '1': out1_scale_mixed,
            '2': out2_scale_mixed,
            '3': out3_scale_mixed,
        }

        return out
