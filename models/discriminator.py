import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.diff_augment import DiffAugment
from models.projector import F_RandomProj
from torch.nn.utils import spectral_norm
from torch_utils.ops import upfirdn2d
from torchvision.transforms import Normalize
from models.constants import VITS
from inplace_abn import InPlaceABN
def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def NormLayer(c, mode='batch'):
    if mode == 'group':
        return nn.GroupNorm(c//2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)
    
    
    
class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, width=1):
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes*width, 4, 2, 1, bias=False),
            InPlaceABN(out_planes*width)
            # NormLayer(out_planes*width),
            # nn.Mish(inplace=True)
        )

    def forward(self, feat):
        return self.main(feat)


class DownBlockPatch(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes),
            conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            InPlaceABN(out_planes)
            # NormLayer(out_planes),
            # nn.Mish(inplace=True)
        )

    def forward(self, feat):
        return self.main(feat)    

class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, patch=False):
        super().__init__()

        # midas channels
        nfc_midas = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                     256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in nfc_midas.keys():
            sizes = np.array(list(nfc_midas.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = nfc_midas
        else:
            nfc = {k: ndf for k, v in nfc_midas.items()}

        # for feature map discriminators with nfc not in nfc_midas
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.Mish(inplace=True)
                       ]

        # Down Blocks
        DB = DownBlockPatch if patch else DownBlock
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=4,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4, 5]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), SingleDisc(nc=cin, start_sz=start_sz, end_sz=8, patch=patch)],
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k]).view(features[k].size(0), -1))

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


class ProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        backbones,
        im_res = 128,
        diff_aug=True,
        backbone_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.diff_aug = diff_aug
        self.im_res = im_res
        feature_networks, discriminators = [], []

        for i, bb_name in enumerate(backbones):

            feat = F_RandomProj(bb_name, **backbone_kwargs)
            disc = MultiScaleD(
                channels=feat.CHANNELS,
                resolutions=feat.RESOLUTIONS,
                **backbone_kwargs,
            )

            feature_networks.append([bb_name, feat])
            discriminators.append([bb_name, disc])

        self.feature_networks = nn.ModuleDict(feature_networks)
        self.feature_networks.eval()
        self.feature_networks = self.feature_networks.train(False)
        for param in self.feature_networks.parameters():
            param.requires_grad = False


        self.discriminators = nn.ModuleDict(discriminators)
        
        

    def train(self, mode=True):
        self.feature_networks = self.feature_networks.train(False)
        self.discriminators = self.discriminators.train(mode)
        return self


    def eval(self):
        return self.train(False)

    def forward(self, x, blur_sigma = 0):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=x.device).div(blur_sigma).square().neg().exp2()
            x = upfirdn2d.filter2d(x, f / f.sum())
            
        logits = []
        for bb_name, feat in self.feature_networks.items():
            x_aug = DiffAugment(x,['translation', 'color','cutout']) if self.diff_aug else x
            x_aug = x_aug.add(1).div(2)
            x_n = Normalize(feat.normstats['mean'], feat.normstats['std'])(x_aug)
            
            if self.im_res < 256 or bb_name in VITS :
                 x_n = F.interpolate(x_n, 224, mode='bilinear', align_corners=False)
            features = feat(x_n)
            logits += self.discriminators[bb_name](features)

        return logits