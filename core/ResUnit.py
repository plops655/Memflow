import torch
import torch.nn as nn

from Helper.pad import pad_for_conv2d

class ResUnit(nn.Module):

    def __init__(self, in_dimensions, out_dimensions, stride, norm_fn='group'):

        super(ResUnit, self).__init__()

        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels=in_dimensions, out_channels=out_dimensions, kernel_size=(3,3), stride=stride,
                               padding=pad_for_conv2d((3,3), stride))
        self.conv2 = nn.Conv2d(in_channels=out_dimensions, out_channels=out_dimensions, kernel_size=(3,3),
                               padding='same')

        num_groups = out_dimensions // 8


        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dimensions)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dimensions)
            if stride != 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dimensions)

        if norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(num_features=out_dimensions)
            self.norm2 = nn.BatchNorm2d(num_features=out_dimensions)
            if stride != 1:
                self.norm3 = nn.BatchNorm2d(num_features=out_dimensions)

        if norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(num_features=out_dimensions)
            self.norm2 = nn.InstanceNorm2d(num_features=out_dimensions)
            if stride != 1:
                self.norm3 = nn.InstanceNorm2d(num_features=out_dimensions)

        if norm_fn == 'None':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if stride != 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_dimensions, out_channels=out_dimensions,
                                                  kernel_size=(3, 3), stride=stride,
                                                  padding=pad_for_conv2d((3, 3), stride)), self.norm3)

    def forward(self, x):

        # y = nn.Sequential(self.conv1, self.norm1, nn.ReLU(), self.conv2, self.norm2, nn.ReLU())(x)

        y = self.conv1(x)
        y = self.norm1(y)
        y = nn.ReLU()(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = nn.ReLU()(y)

        if self.downsample:
            x = self.downsample(x)

        assert x.shape == y.shape

        return nn.ReLU()(x + y)

