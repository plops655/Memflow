import torch
import torch.nn as nn
import numpy

from corr_vol import compute_corr_vol

from utils.consts import H, W, r, len_of_lookup

class Lookup(nn.Module):

    def __init__(self):
        super(Lookup, self).__init__()

    def lookup_i_j(self, i, j, k, curr_flow):
        # inputs: i, j ~ the position of lookup
        #         k ~ log base 2 pooling depth of image we lookup
        #         curr_flow    ~ 2 x H x W flow map

        # output: len(lookup) x 2 tensor

        h, w = H / 2 ** k, W / 2 ** k
        tensor_lst = []
        for y in range(-r, r + 1):
            for x in range(abs(y) - r, r - abs(y) + 1):
                tensor_lst.append(torch.tensor([(i / 2 ** k + curr_flow[0][i][j] / 2 ** k + y) / (h / 2) - 1,
                                                (j / 2 ** k + curr_flow[1][i][j] / 2 ** k + x) / (w / 2) - 1]))

        lookout_tensor = torch.stack(tensor_lst, dim=0)
        return lookout_tensor

    def lookup_from_corr(self, corr_vol, curr_flow, k):

        # Input: corr_vol ~ correlation cost volume for depth k
        #         curr_flow    ~ 2 x H x W flow map
        #        k        ~ depth of pooling


        # Output: lookup_tensor ~ (len(lookup), 1, H, W) tensor of lookups

        lookup_tensor = torch.empty((H, W, 1, self.lookup_len))

        for j in range(W):

            sub_grid = torch.empty((H, 1, self.lookup_len, 2))

            sub_corr_vol = corr_vol[:, j, :, :].unsqueeze(1)

            # make grid by stacking lookups

            for i in range(H):
                sub_grid[i, :, :, :] = self.lookup_i_j(i=i, j=j, k=k, curr_flow=curr_flow).unsqueeze(0)

            interpolated_lookup = torch.nn.functional.grid_sample(input=sub_corr_vol, grid=sub_grid, mode='bilinear',
                                                                  align_corners=True)
            lookup_tensor[:, j, :, :] = interpolated_lookup.squeeze(1)

        lookup_tensor = lookup_tensor.permute((3, 2, 0, 1))

        return lookup_tensor

    def forward(self, feat1, feat2, curr_flow):

        # Inputs: feat1        ~ D x H x W feature map of 1st image
        #         feat2        ~ D x H x W feature map of 2nd image
        #         curr_flow    ~ 2 x H x W flow map

        # output: motion_lookup ~ len(lookup) x 4 x H x W tensor containing desired correlation data for motion encoder

        corr_vol_1, corr_vol_2, corr_vol_4, corr_vol_8 = compute_corr_vol(
            g_0=feat1, g_1=feat2, depth_of_features=3)

        lookup_tensor_1 = self.lookup_from_corr(corr_vol=corr_vol_1, curr_flow=curr_flow, k=0)
        lookup_tensor_2 = self.lookup_from_corr(corr_vol=corr_vol_2, curr_flow=curr_flow, k=1)
        lookup_tensor_4 = self.lookup_from_corr(corr_vol=corr_vol_4, curr_flow=curr_flow, k=2)
        lookup_tensor_8 = self.lookup_from_corr(corr_vol=corr_vol_8, curr_flow=curr_flow, k=3)

        motion_lookup = torch.empty((len_of_lookup, 4, H, W))

        motion_lookup[:, 0, :, :] = lookup_tensor_1.squeeze(1)
        motion_lookup[:, 1, :, :] = lookup_tensor_2.squeeze(1)
        motion_lookup[:, 2, :, :] = lookup_tensor_4.squeeze(1)
        motion_lookup[:, 3, :, :] = lookup_tensor_8.squeeze(1)

        return motion_lookup
