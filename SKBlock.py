# Super Kernel Block Designs
from math import *
from torch import *
import torch.nn as nn
import numpy

from utilities import pad_for_conv2d


class SKBlock(nn.Module):

    name = "SKBlock"
    def __init__(self, large_kernel, small_kernel, stride, out_channels):
        super(SKBlock, self).__init__()
        self.L = large_kernel
        self.S = small_kernel
        self.stride = stride
        self.out_channels = out_channels


class DirectBlock(SKBlock):

    name = "DirectBlock"
    def __init__(self, large_kernel, small_kernel, stride, out_channels):
        super(DirectBlock, self).__init__(large_kernel, small_kernel, stride, out_channels)

    def forward(self, x):
        # Input: rgb image with 3 channels (red, green, blue)
        # Output: feature maps

        num_channels = x.shape[-3]
        H, W = x.shape[-2], x.shape[-1]

        padding = pad_for_conv2d(H, W, self.L, stride=self.stride)

        x1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=self.L,
                       stride=self.stride, groups=num_channels, padding=padding)(x)

        x2 = nn.Sigmoid()(x1 + x)

        padding = pad_for_conv2d(H, W, (1, 1), stride=1)

        x3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 1),
                       stride=1, padding=padding)(x2)

        x4 = nn.Sigmoid()(x2 + x3)

        out_x = nn.Conv2d(in_channels=num_channels, out_channels=self.out_channels, kernel_size=(1, 1),
                  stride=1, padding=padding)(x4)
        return out_x


class ParallelBlock(SKBlock):

    name = "ParallelBlock"
    def __init__(self, large_kernel, small_kernel, stride, out_channels):
        super(ParallelBlock, self).__init__(large_kernel, small_kernel, stride, out_channels)

    def forward(self, x):
        # Input: rgb image with 3 channels (red, green, blue)
        # Output: feature maps

        num_channels = x.shape[-3]
        H, W = x.shape[-2], x.shape[-1]

        padding = pad_for_conv2d(H, W, self.L, stride=self.stride)

        x1_L = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=self.L,
                         stride=self.stride, groups=num_channels, padding=padding)(x)

        padding = pad_for_conv2d(H, W, self.S, stride=self.stride)

        x1_S = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=self.S,
                         stride=self.stride, groups=num_channels, padding=padding)(x)

        x2 = nn.Sigmoid()(x1_L + x1_S + x)

        padding = pad_for_conv2d(H, W, (1, 1), stride=1)

        x3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 1),
                       stride=1, padding=padding)(x2)

        x4 = nn.Sigmoid()(x2 + x3)

        out_x = nn.Conv2d(in_channels=num_channels, out_channels=self.out_channels, kernel_size=(1, 1),
                          stride=1, padding=padding)(x4)
        return out_x


class FunnelBlock(SKBlock):

    name = "FunnelBlock"
    def __init__(self, large_kernel, small_kernel, stride, out_channels):
        super(FunnelBlock, self).__init__(large_kernel, small_kernel, stride, out_channels)

    def forward(self, x):
        num_channels = x.shape[-3]
        H, W = x.shape[-2], x.shape[-1]

        padding = pad_for_conv2d(H, W, self.L, stride=self.stride)

        x1_L = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=self.L,
                         stride=self.stride, groups=num_channels, padding=padding)(x)

        x2_L = nn.Sigmoid()(x + x1_L)

        padding = pad_for_conv2d(H, W, self.S, stride=self.stride)

        x1_S = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=self.S,
                         stride=self.stride, groups=num_channels, padding=padding)(x2_L)

        x2_S = nn.Sigmoid()(x2_L + x1_S)

        padding = pad_for_conv2d(H, W, (1, 1), stride=1)

        x3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 1),
                       stride=1, padding=padding)(x2_S)

        x4 = nn.Sigmoid()(x3 + x2_S)

        out_x = nn.Conv2d(in_channels=num_channels, out_channels=self.out_channels, kernel_size=(1, 1),
                          stride=1, padding=padding)(x4)

        return out_x


# we use ConicalBlock
class ConicalBlock(SKBlock):

    name = "ConicalBlock"
    def __init__(self, large_kernel, small_kernel, stride, out_channels):
        super(ConicalBlock, self).__init__(large_kernel, small_kernel, stride, out_channels)

    def forward(self, x):
        num_channels = x.shape[-3]
        H, W = x.shape[-2], x.shape[-1]

        padding = pad_for_conv2d(H, W, self.S, stride=self.stride)

        x1_S = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=self.S,
                         stride=self.stride, groups=num_channels, padding=padding)(x)

        x2_S = nn.Sigmoid()(x + x1_S)

        padding = pad_for_conv2d(H, W, self.L, stride=self.stride)

        x1_L = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=self.L,
                         stride=self.stride, groups=num_channels, padding=padding)(x2_S)

        x2_L = nn.Sigmoid()(x2_S + x1_L)

        padding = pad_for_conv2d(H, W, (1, 1), stride=1)

        x3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 1),
                       stride=1, padding=padding)(x2_L)

        x4 = nn.Sigmoid()(x3 + x2_L)

        out_x = nn.Conv2d(in_channels=num_channels, out_channels=self.out_channels, kernel_size=(1, 1),
                          stride=1, padding=padding)(x4)

        return out_x
