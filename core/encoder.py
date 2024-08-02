import torch.nn as nn
from core.ResUnit import ResUnit

from Helper.pad import pad_for_conv2d


class Encoder(nn.Module):

    def __init__(self, norm_fn = 'group', dropout = 0.0):
        super(Encoder, self).__init__()

        self.norm_fn = norm_fn
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=7, stride = 2,
                               padding = pad_for_conv2d((7,7), 2))

        self.layer1 = nn.Sequential(ResUnit(in_dimensions=64, out_dimensions=64, stride=1),
                                    ResUnit(in_dimensions=64, out_dimensions=64, stride=2))

        self.layer2 = nn.Sequential(ResUnit(in_dimensions=64, out_dimensions=128, stride=1),
                                    ResUnit(in_dimensions=128, out_dimensions=128, stride=2))

        self.layer3 = nn.Sequential(ResUnit(in_dimensions=128, out_dimensions=192, stride=1),
                                    ResUnit(in_dimensions=192, out_dimensions=192, stride=1))

        self.conv2 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, padding='same')

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

        # Input: x ~ (batch_sz, 3, H, W) tensors for frames
        # Output: x ~ (batch_sz, 256, H / 8, W / 8)

        x = nn.Sequential(self.conv1, self.norm1, nn.ReLU())(x)

        # x = self.layer1(x)
        x = ResUnit(in_dimensions=64, out_dimensions=64, stride=1).forward(x)
        x = ResUnit(in_dimensions=64, out_dimensions=64, stride=2).forward(x)

        # x = self.layer2(x)
        x = ResUnit(in_dimensions=64, out_dimensions=128, stride=1).forward(x)
        x = ResUnit(in_dimensions=128, out_dimensions=128, stride=2).forward(x)

        # x = self.layer3(x)
        x = ResUnit(in_dimensions=128, out_dimensions=192, stride=1).forward(x)
        x = ResUnit(in_dimensions=192, out_dimensions=192, stride=1).forward(x)

        x = nn.Sequential(self.conv2, self.norm2, nn.ReLU())(x)

        # Dropout at the end unless we overfit. Prevents slow feature encoding.

        if self.dropout and self.dropout > 0.0:
            x = nn.Dropout2d(p=self.dropout)(x)

        return x


