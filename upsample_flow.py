import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.consts import H, W, hidden_dimension


class Upsample_Flow(nn.Module):
    def __init__(self, norm_fn='group'):
        super(Upsample_Flow, self).__init__()

        num_groups = 8

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dimension)

        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(num_features=hidden_dimension)

        if self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(num_features=hidden_dimension)

        if self.norm_fn is None:
            self.norm1 = nn.Sequential()

        self.mask = nn.init.xavier_uniform_(torch.randn(1, 576, H // 8, W // 8))
        self.conv1 = nn.Conv2d(in_channels=576, out_channels = 576, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels = 2, kernel_size=(3,3))

    def forward(self, x):

        # Input: x ~ 1 x hidden_dim=64 x H // 8 x W // 8

        x = F.unfold(x, kernel_size=3, padding=2, stride=1) # 1 x 576 x H // 8 x W // 8
        x = x.view(1, 9, H, W)

        self.mask = F.relu(self.norm1(self.conv1(self.mask)))
        mask_unsqueezed = self.mask.view(1, 9, H, W)
        mask_updated = self.conv2(mask_unsqueezed)

        mask_softmax = F.softmax(mask_updated, dim=1)
        x = (x * mask_softmax).sum(dim=1, keepdim=True)

        return x

