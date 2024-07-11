import torch.nn.functional as F
from ResUnit import ResUnit

from utils.utilities import *


class Encoder(nn.Module):

    def __init__(self, norm_fn = 'group', dropout = 0.0):
        super(Encoder, self).__init__()

        self.norm_fn = norm_fn
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=7, stride = 2, padding = 0)

        self.layer1 = nn.Sequential(ResUnit(in_dimensions=64, out_dimensions=64, stride=1),
                                    ResUnit(in_dimensions=64, out_dimensions=64, stride=2))

        self.layer2 = nn.Sequential(ResUnit(in_dimensions=64, out_dimensions=128, stride=1),
                                    ResUnit(in_dimensions=128, out_dimensions=128, stride=2))

        self.layer3 = nn.Sequential(ResUnit(in_dimensions=128, out_dimensions=192, stride=1),
                                    ResUnit(in_dimensions=192, out_dimensions=192, stride=1))

        self.conv2 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3)

        num_groups = 8

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=64)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=256)

        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(num_features=64)
            self.norm2 = nn.BatchNorm2d(num_features=256)

        if self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(num_features=64)
            self.norm1 = nn.InstanceNorm2d(num_features=256)

        if self.norm_fn is None:
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()



    def forward(self, x):

        # Input

        x = nn.Sequential(self.conv1, self.norm1, F.relu)(x)
        x = nn.Sequential(self.layer1, self.layer2, self.layer3)(x)
        x = nn.Sequential(self.conv2, self.norm2, F.relu)(x)

        # Dropout at the end unless we overfit. Prevents slow feature encoding.

        if self.dropout and self.dropout > 0.0:
            x = nn.Dropout2d(p=self.dropout)(x)

        return x


