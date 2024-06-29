import torch
import torch.nn as nn

class ResUnit(nn.Module):

    def __init__(self, in_dimensions, out_dimensions, stride, norm_fn='group'):

        super(ResUnit, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_dimensions, out_channels=out_dimensions, kernel_size=(3,3), padding=1,
                               stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_dimensions, out_channels=out_dimensions, kernel_size=(3,3), padding=1)

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

        self.downsample = None

        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=out_dimensions, out_channels=out_dimensions,
                                                      stride=stride), self.norm3)

    def forward(self, x):

        y = self.relu(self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x))))))

        if self.downsample:
            x = self.downsample(x)

        return self.relu(x + y)
