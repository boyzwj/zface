import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from models.ca import ECA
from inplace_abn import InPlaceABN

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample=False, up_sample=False,attention = False,activation='lrelu'):
        super(ResBlock, self).__init__()     
        main_module_list = []

        main_module_list += [
                nn.BatchNorm2d(in_channel),
                nn.Mish(inplace=True),
                nn.Conv2d(in_channel,in_channel, 3, 1, 1,bias=False)
            ]

        if down_sample:
            main_module_list.append(nn.AvgPool2d(kernel_size=2))
        elif up_sample:
            main_module_list += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                ]

        main_module_list += [
                nn.BatchNorm2d(in_channel),
                nn.Mish(inplace=True),
                nn.Conv2d(in_channel,out_channel, 3, 1, 1,bias=False)
            ]     
        if attention:
             main_module_list += [
                 ECA(out_channel)
             ]
        self.main_path = nn.Sequential(*main_module_list)
        side_module_list = []
        if in_channel != out_channel:
            side_module_list += [nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)]
        else:
            side_module_list += [nn.Identity()]   
        if down_sample:
            side_module_list.append(nn.AvgPool2d(kernel_size=2))
        elif up_sample:
            side_module_list += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
        self.side_path = nn.Sequential(*side_module_list)

    def forward(self, x):
        x1 = self.main_path(x)
        x2 = self.side_path(x)
        return (x1 + x2) / math.sqrt(2)

class StarGANv2Discriminator(nn.Module):
    def __init__(self, img_size=256, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size

        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlock(dim_in, dim_out, down_sample=True,attention=True)]
            dim_in = dim_out

        blocks += [nn.Mish(inplace=True)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.Mish(inplace=True)]
        blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out