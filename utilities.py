import torch
import torch.nn as nn
from math import *

def pad_for_conv2d(H, W, kernel, stride=1):
    sx, sy = stride, stride
    kx, ky = kernel
    px = ceil(((sx - 1) * H + kx - sx) / 2)
    py = ceil(((sy - 1) * W + ky - sy) / 2)
    p = (px, py)
    return p

class Res_Layer(nn.Module):

    def __init__(self, H, W, D, out_dim1, out_dim2):
        ksz = (7, 7)
        padding = pad_for_conv2d(H, W, ksz)

        self.relu = nn.ReLU()

        self.LN1 = nn.LayerNorm(normalized_shape=(1, D, H, W), dtype=torch.Tensor.float)

        self.W1 = nn.Conv2d(in_channels=D, out_channels=out_dim1, kernel_size=ksz, padding=padding)

        self.LN2 = nn.LayerNorm(normalized_shape=(1, D, H, W), dtype=torch.Tensor.float)

        self.W2 = nn.Conv2d(in_channels=out_dim1, out_channels=out_dim1, kernel_size=ksz, padding=padding)

    # full preactivation forward pass

    def forward(self, x):
        # ResUnit: BN -> ReLU -> Conv2d -> BN -> ReLU -> Conv2d

        model = (self.LN1, self.relu, self.W1, self.LN2, self.relu, self.W2)
        out = model(x)

        return out