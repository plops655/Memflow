import torch
import torch.nn as nn

from utils.consts import H, W, batch_sz, hidden_dim

from Helper.pad import pad_for_conv2d


class Upsample_Flow(nn.Module):
    def __init__(self, norm_fn='group'):
        super(Upsample_Flow, self).__init__()

        num_groups = 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim * 9)

        if norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(num_features=hidden_dim)

        if norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(num_features=hidden_dim)

        if norm_fn is None:
            self.norm1 = nn.Sequential()

        # self.mask = nn.init.xavier_uniform_(torch.randn(1, 9, 8, 8, consts.H // 8, consts.W // 8))
        # self.conv1 = nn.Conv2d(in_channels=576, out_channels=576, kernel_size=(3,3), padding=pad_for_conv2d((3,3)))
        # self.conv2 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=(3,3), padding=pad_for_conv2d((3,3)))

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=576, kernel_size=(3,3), padding=pad_for_conv2d((3,3)))
        self.conv2 = nn.Conv2d(in_channels=576, out_channels=576, kernel_size=(3,3), padding=pad_for_conv2d((3,3)))

    def forward(self, x, h):

        # make sure of this!!

        # Input: x ~ unshaped predicted flow: batch_sz x 2 x H // 8 x W // 8
        #        h ~ hidden state: batch_sz x hidden_dim=64 x H // 8 x W // 8

        mask_model = nn.Sequential(self.conv1, self.norm1, nn.ReLU(), self.conv2)

        mask = mask_model(h)
        mask = mask.view(batch_sz, 1, 9, 8, 8, H // 8, W // 8)

        x = nn.Unfold(kernel_size=3, padding=1, stride=1)(x) # batch_sz x 18 x H // 8 x W // 8
        x = x.view(batch_sz, 2, 9, 1, 1, H // 8, W // 8)

        mask_softmax = nn.Softmax(dim=2)(mask)
        x = (x * mask_softmax).sum(dim=2, keepdim=False)

        # x ~ (batch_sz, 2, 8, 8, H // 8, W // 8)
        x = x.permute((0, 1, 2, 4, 3, 5)).reshape((batch_sz, 2, H, W))

        return x



