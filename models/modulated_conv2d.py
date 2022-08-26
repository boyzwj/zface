from typing import Sequence
import torch
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from kornia.filters import filter2d

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)


# class Upsample(nn.Module):
#     def __init__(self, kernel, factor=2):
#         super().__init__()

#         self.factor = factor
#         kernel = make_kernel(kernel) * (factor ** 2)
#         self.register_buffer("kernel", kernel)

#         p = kernel.shape[0] - factor

#         pad0 = (p + 1) // 2 + factor - 1
#         pad1 = p // 2

#         self.pad = (pad0, pad1)

#     def forward(self, input):
#         out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

#         return out    



def add_spectral_norm(m, use_spectral_norm):
  if not use_spectral_norm:
    return m
  return spectral_norm(m)


def exists(val):
    return val is not None




class OutputBlockSumAndUpsample(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, num_outputs = 3, use_spectral_norm=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)
        self.conv = add_spectral_norm(Conv2DMod(input_channel, num_outputs, 1, demod=False), use_spectral_norm)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Blur()
        )  if upsample else None
        
    def finishForward(self, x, prev_output):
        if exists(prev_output):
            x = x + prev_output
        if exists(self.upsample):
            x = self.upsample(x,self.resample_filter)
        return x        

    def forward(self, x, istyle, prev_output):
        # b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)
        return self.finishForward(x, prev_output)

class RGBBlock(OutputBlockSumAndUpsample):
    def __init__(self, latent_dim, input_channel, upsample, num_outputs = 3):
        super().__init__(latent_dim, input_channel, upsample, num_outputs)
       
        
    def finishForward(self, x, prev_output):
        if exists(prev_output):
            if exists(self.upsample):
                prev_output = self.upsample(prev_output)
                x = x + prev_output
            else:
                x = x + prev_output
        return x 


# from https://github.com/lucidrains/stylegan2-pytorch/blob/05f7585e8da9c09752696872e04de0414a972486/stylegan2_pytorch/stylegan2_pytorch.py
class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, z_id):
        b, c, h, w = x.shape

        w1 = z_id[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, dilation={self.dilation}, demodulation={self.demod})"
        )



# class Conv2DMod1(nn.Module):
#     def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
#         super().__init__()
#         self.filters = out_chan
#         self.demod = demod
#         self.kernel = kernel
#         self.stride = stride
#         self.dilation = dilation
#         self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
#         self.eps = eps
#         nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

#     def _get_same_padding(self, size, kernel, dilation, stride):
#         return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

#     def forward(self, x, z_id):
#         # b, c, h, w = x.shape

#         w1 = z_id[:, None, :, None, None]
#         w2 = self.weight[None, :, :, :, :]
#         weights = w2 * (w1 + 1)

#         if self.demod:
#             d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
#             weights = weights * d

#         x = x.reshape(1, -1, x.size(2), x.size(3))
        
#         # _, _, *ws = weights.shape
#         ws = weights.size()[2:]
        

#         weights = weights.reshape(x.size(0) * self.filters, *ws)

#         padding = self._get_same_padding(x.size(2), self.kernel, self.dilation, self.stride)
#         x = F.conv2d(x, weights, padding=padding, groups=x.size(0))

#         x = x.reshape(-1, self.filters, x.size(2), x.size(3))
#         return x

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
#             f" {self.weight.shape[2]}, stride={self.stride}, dilation={self.dilation}, demodulation={self.demod})"
#         )
        