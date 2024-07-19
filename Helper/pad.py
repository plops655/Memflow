import utils.consts as consts
from math import *

def pad_for_conv2d(krnl, strd=1):
    sx, sy = strd, strd
    kx, ky = krnl
    px = max(0, ceil((consts.H // sx * sx - sx + kx - consts.H) / 2))
    py = max(0, ceil((consts.W // sy * sy - sy + ky - consts.W) / 2))
    p = (px, py)
    return p

if __name__ == '__main__':

    import torch
    import torch.nn as nn

    consts.set_H(192)
    consts.set_W(168)

    stride = 2
    kernel = (5,7)

    batch_size = 1
    in_channels=1
    out_channels=1

    in_tensor = torch.empty(batch_size, in_channels, consts.H, consts.W)

    padding = pad_for_conv2d(kernel, stride)

    out_tensor = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
              padding=padding).forward(in_tensor)

    print("in_tensor shape:", in_tensor.shape)
    print("padding:", padding)
    print("out_tensor shape:", out_tensor.shape)
