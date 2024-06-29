import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUnit import ResUnit

from utilities import *


class Siamese_Encoder(nn.Module):

    def __init__(self, frame1, frame2):
        super(Siamese_Encoder, self).__init__()
        assert frame1.shape == frame2.shape, f"ERROR: Frames {frame1} and {frame2} have different shapes. All frames must have the same size."

        self.H, self.W, self.D = frame1.shape
        self.frame1 = frame1
        self.frame2 = frame2

        self.conv1 = nn.Conv2d(in_channels = self.D, out_channels = 64, kernel_size=7, stride = 2, padding = 0)

        self.layer1 = nn.Sequential(ResUnit(in_dimensions=64, out_dimensions=64, stride=1),
                                    ResUnit(in_dimensions=64, out_dimensions=64, stride=2))

        self.layer2 = nn.Sequential(ResUnit(in_dimensions=64, out_dimensions=128, stride=1),
                                    ResUnit(in_dimensions=128, out_dimensions=128, stride=2))

        self.layer3 = nn.Sequential(ResUnit(in_dimensions=128, out_dimensions=192, stride=1),
                                    ResUnit(in_dimensions=192, out_dimensions=192, stride=1))

        self.conv2 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3)

    def forward(self, x):
        y = nn.Sequential(self.conv1, self.layer1, self.layer2, self.layer3, self.conv2)(x)
        return y


