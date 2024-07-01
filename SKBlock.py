# Super Kernel Block Designs
import torch.nn as nn
import torch.nn.functional as F


class SKBlock(nn.Module):

    name = "SKBlock"
    def __init__(self, large_kernel, small_kernel, stride, in_channels, out_channels, norm_fn='group'):
        super(SKBlock, self).__init__()
        self.L = large_kernel
        self.S = small_kernel
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=in_channels//8, num_channels=in_channels)
            self.norm2 = nn.GroupNorm(num_groups=in_channels//8, num_channels=in_channels)
            self.norm3 = nn.GroupNorm(num_groups=out_channels//8, num_channels=out_channels)

        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(num_features=in_channels)
            self.norm2 = nn.BatchNorm2d(num_features=in_channels)
            self.norm3 = nn.BatchNorm2d(num_features=out_channels)

        if self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(num_features=in_channels)
            self.norm2 = nn.InstanceNorm2d(num_features=in_channels)
            self.norm3 = nn.InstanceNorm2d(num_features=out_channels)

        if self.norm_fn is None:
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()


class DirectBlock(SKBlock):

    name = "DirectBlock"
    def __init__(self, large_kernel, small_kernel, stride, in_channels, out_channels, norm_fn='group'):
        super(DirectBlock, self).__init__(large_kernel, small_kernel, stride, in_channels, out_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.L,
                       stride=self.stride, groups=in_channels, padding=0)

        self.conv00 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 1.5, kernel_size=(1, 1),
                       stride=1, padding=0)

        self.conv01 = nn.Conv2d(in_channels=in_channels * 1.5, out_channels=in_channels, kernel_size=(1, 1),
                       stride=1, padding=0)

        self.conv10 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 1.5, kernel_size=(1, 1),
                       stride=1, padding=0)

        self.conv11 = nn.Conv2d(in_channels=in_channels * 1.5, out_channels=out_channels, kernel_size=(1, 1),
                       stride=1, padding=0)

    def forward(self, x):
        # Input: rgb image with 3 channels (red, green, blue)
        # Output: feature maps

        x = F.relu(self.norm1(x + self.conv1(x)))
        x = F.relu(self.norm2(self.conv01(self.conv00(x)) + x))
        x = self.norm3(self.conv11(self.conv10(x)))

        return x


class ParallelBlock(SKBlock):

    name = "ParallelBlock"
    def __init__(self, large_kernel, small_kernel, stride, in_channels, out_channels, norm_fn='group'):
        super(ParallelBlock, self).__init__(large_kernel, small_kernel, stride, in_channels, out_channels, norm_fn)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.L,
                       stride=self.stride, groups=in_channels, padding=0)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.S,
                         stride=self.stride, groups=in_channels, padding=0)

        self.conv00 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 1.5, kernel_size=(1, 1),
                                stride=1, padding=0)

        self.conv01 = nn.Conv2d(in_channels=in_channels * 1.5, out_channels=in_channels, kernel_size=(1, 1),
                                stride=1, padding=0)

        self.conv10 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 1.5, kernel_size=(1, 1),
                                stride=1, padding=0)

        self.conv11 = nn.Conv2d(in_channels=in_channels * 1.5, out_channels=out_channels, kernel_size=(1, 1),
                                stride=1, padding=0)

    def forward(self, x):
        # Input: rgb image with 3 channels (red, green, blue)
        # Output: feature maps

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = F.relu(self.norm1(x1 + x2 + x))
        x = F.relu(self.norm2(x + self.conv01(self.conv00(x))))
        x = self.norm3(self.conv11(self.conv10(x)))

        return x


class FunnelBlock(SKBlock):

    name = "FunnelBlock"
    def __init__(self, large_kernel, small_kernel, stride, in_channels, out_channels, norm_fn='group'):
        super(FunnelBlock, self).__init__(large_kernel, small_kernel, stride, in_channels, out_channels, norm_fn)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.L,
                               stride=self.stride, groups=in_channels, padding=0)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.S,
                               stride=self.stride, groups=in_channels, padding=0)

        self.conv00 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 1.5, kernel_size=(1, 1),
                                stride=1, padding=0)

        self.conv01 = nn.Conv2d(in_channels=in_channels * 1.5, out_channels=in_channels, kernel_size=(1, 1),
                                stride=1, padding=0)

        self.conv10 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 1.5, kernel_size=(1, 1),
                                stride=1, padding=0)

        self.conv11 = nn.Conv2d(in_channels=in_channels * 1.5, out_channels=out_channels, kernel_size=(1, 1),
                                stride=1, padding=0)

    def forward(self, x):

        x = F.relu(x + self.conv1(x))
        x = F.relu(self.norm1(x + self.conv2(x)))
        x = F.relu(self.norm2(x + self.conv01(self.conv00(x))))
        x = self.norm3(self.conv11(self.conv10(x)))

        return x

# we use ConicalBlock
class ConicalBlock(SKBlock):

    name = "ConicalBlock"

    def __init__(self, large_kernel, small_kernel, stride, in_channels, out_channels, norm_fn):
        super(ConicalBlock, self).__init__(large_kernel, small_kernel, stride, in_channels, out_channels, norm_fn)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.S,
                               stride=self.stride, groups=in_channels, padding=0)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.L,
                               stride=self.stride, groups=in_channels, padding=0)

        self.conv00 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 1.5, kernel_size=(1, 1),
                                stride=1, padding=0)

        self.conv01 = nn.Conv2d(in_channels=in_channels * 1.5, out_channels=in_channels, kernel_size=(1, 1),
                                stride=1, padding=0)

        self.conv10 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 1.5, kernel_size=(1, 1),
                                stride=1, padding=0)

        self.conv11 = nn.Conv2d(in_channels=in_channels * 1.5, out_channels=out_channels, kernel_size=(1, 1),
                                stride=1, padding=0)


    def forward(self, x):

        x = F.relu(x + self.conv1(x))
        x = F.relu(self.norm1(x + self.conv2(x)))
        x = F.relu(self.norm2(x + self.conv01(self.conv00(x))))
        x = self.norm3(self.conv11(self.conv10(x)))

        return x
