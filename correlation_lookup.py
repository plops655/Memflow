import torch
import torch.nn as nn
import numpy

from corr_vol import compute_corr_vol

class Lookup:

    def __init__(self, feat1, feat2, curr_flow, r):

        # Inputs: feat1        ~ H x W x D feature map of 1st image
        #         feat2        ~ H x W x D feature map of 2nd image
        #         curr_flow    ~ H x W x 2 flow map
        #         r            ~ lookup radius

        self.feat1 = feat1
        self.feat2 = feat2
        self.flow = curr_flow

        assert self.feat1.shape[0:2] == self.feat2.shape[0:2], f"shape of feature 1: {self.feat1.shape} doesn't match that of feature 2: {self.feat2.shape}"
        assert self.flow.shape[0:2] == self.feat1.shape[0:2], f"H,W of feature 1: {self.feat1.shape[0:2]} doesn't match that of flow: {self.flow.shape[0:2]}"

        self.H, self.W = self.flow.shape[0:2]
        self.r = r
        self.lookup_len = 2 * r ** 2 + 2 * r + 1

    def lookup_i_j(self, i, j, k):
        # inputs: i, j ~ the position of lookup
        #         k ~ log base 2 pooling depth of image we lookup
        #         r    ~ L1 radius of lookup

        # output: len(lookup) x 2 tensor

        h, w = self.H / 2 ** k, self.W / 2 ** k
        tensor_lst = []
        for y in range(-self.r, self.r + 1):
            for x in range(abs(y) - self.r, self.r - abs(y) + 1):
                tensor_lst.append(torch.tensor([(i / 2 ** k + self.flow[i][j][0] / 2 ** k + y) / (h / 2) - 1,
                                                (j / 2 ** k + self.flow[i][j][1] / 2 ** k + x) / (w / 2) - 1]))

        lookout_tensor = torch.stack(tensor_lst, dim=0)
        return lookout_tensor

    def lookup_from_corr(self, corr_vol, k):

        # Input: k    ~ depth of pooling of correlation
        # .       flow ~ H x W tensor of current flow
        #        r    ~ L1 distance to lookup

        # Output: H x W x len(Lookup) x 4

        # Below isn't power efficient


        lookup_tensor = torch.empty((self.H, self.W, 1, self.lookup_len))

        for j in range(self.W):

            sub_grid = torch.empty((self.H, 1, self.lookup_len, 2))

            sub_corr_vol = corr_vol[:, j, :, :].unsqueeze(1)

            # make grid by stacking lookups

            for i in range(self.H):
                sub_grid[i:, :, :, ] = self.lookup_i_j(i=i, j=j, k=k)

            sub_grid = sub_grid.float()
            sub_corr_vol = sub_corr_vol.float()
            interpolated_lookup = torch.nn.functional.grid_sample(input=sub_corr_vol, grid=sub_grid, mode='bilinear',
                                                                  align_corners=True)
            lookup_tensor[:, j, :, :] = interpolated_lookup.squeeze(1)

        return lookup_tensor

    def forward(self):

        # output: motion_lookup ~ H x W x 4 x lookup_len tensor containing desired correlation data for motion encoder

        corr_vol_1, corr_vol_2, corr_vol_4, corr_vol_8 = compute_corr_vol(
            g_0=self.feat1, g_1=self.feat2, depth_of_features=3)

        lookup_tensor_1 = self.lookup_from_corr(corr_vol=corr_vol_1, k=0)
        lookup_tensor_2 = self.lookup_from_corr(corr_vol=corr_vol_2, k=1)
        lookup_tensor_4 = self.lookup_from_corr(corr_vol=corr_vol_4, k=2)
        lookup_tensor_8 = self.lookup_from_corr(corr_vol=corr_vol_8, k=3)

        motion_lookup = torch.empty((self.H, self.W, 4, self.lookup_len))

        motion_lookup[:, :, 0, :] = lookup_tensor_1.squeeze(2)
        motion_lookup[:, :, 1, :] = lookup_tensor_2.squeeze(2)
        motion_lookup[:, :, 2, :] = lookup_tensor_4.squeeze(2)
        motion_lookup[:, :, 3, :] = lookup_tensor_8.squeeze(2)

        return motion_lookup
