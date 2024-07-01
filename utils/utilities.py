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
