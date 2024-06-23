import torch
import torch.nn as nn

from utilities import pad_for_conv2d, Res_Layer

from SKBlock import ConicalBlock
from utilities import *


class Siamese_Encoder(nn.Module):

    def __init__(self, frame1, frame2):
        super(Siamese_Encoder, self).__init__()
        assert frame1.shape == frame2.shape, f"ERROR: Frames {frame1} and {frame2} have different shapes. All frames must have the same size."

        H, W = frame1.shape[0:2]
        self.frame1 = frame1
        self.frame2 = frame2

        self.ru1 = (ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3)), ConicalBlock())



