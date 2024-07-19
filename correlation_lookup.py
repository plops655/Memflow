import torch
import torch.nn as nn
import numpy

from corr_vol import compute_corr_vol

import utils.consts as consts

class Lookup(nn.Module):

    def __init__(self):
        super(Lookup, self).__init__()

    def lookup_i_j(self, i, j, k, curr_flow):
        # inputs: i, j ~ the position of lookup
        #         k ~ log base 2 pooling depth of image we lookup
        #         curr_flow    ~ 2 x H x W flow map

        # output: len(lookup) x 2 tensor

        h, w = consts.H / 2 ** k, consts.W / 2 ** k
        tensor_lst = []
        for y in range(-consts.r, consts.r + 1):
            for x in range(abs(y) - consts.r, consts.r - abs(y) + 1):
                tensor_lst.append(torch.tensor([(j / 2 ** k + curr_flow[1][i][j] / 2 ** k + x) / (w / 2) - 1,
                                                (i / 2 ** k + curr_flow[0][i][j] / 2 ** k + y) / (h / 2) - 1]))

        lookout_tensor = torch.stack(tensor_lst, dim=0)
        return lookout_tensor

    def lookup_from_corr(self, corr_vol, curr_flow, k):

        # Input: corr_vol ~ batch of correlation cost volume for depth k
        #         curr_flow    ~ 2 x H x W flow map
        #        k        ~ depth of pooling


        # Output: lookup_tensors ~ (batch_sz, len(lookup), 1, H / 8, W / 8) tensor of lookups

        batch_sz = consts.batch_sz
        lookup_tensors = torch.empty((batch_sz, consts.len_of_lookup, 1, consts.H // 8, consts.W // 8))
        for n in range(batch_sz):
            lookup_tensor = torch.empty((consts.H // 8, consts.W // 8, 1, consts.len_of_lookup))
            for j in range(consts.W // 8):

                sub_grid = torch.empty((consts.H // 8, 1, consts.len_of_lookup, 2))

                sub_corr_vol = corr_vol[n, :, j, :, :].unsqueeze(1)

                # make grid by stacking lookups

                for i in range(consts.H // 8):
                    sub_grid[i, :, :, :] = self.lookup_i_j(i=i, j=j, k=k, curr_flow=curr_flow).unsqueeze(0)

                interpolated_lookup = torch.nn.functional.grid_sample(input=sub_corr_vol, grid=sub_grid, mode='bilinear',
                                                                      align_corners=True)
                lookup_tensor[:, j, :, :] = interpolated_lookup.squeeze(1)

            lookup_tensor = lookup_tensor.permute((3, 2, 0, 1))

            lookup_tensors[n, :, :, :, :] = lookup_tensor

        return lookup_tensors

    def forward(self, feat1, feat2, curr_flow):

        # Inputs: feat1        ~ batch_sz x D x H / 8 x W / 8 feature map of 1st image
        #         feat2        ~ batch_sz x D x H / 8 x W / 8 feature map of 2nd image
        #         curr_flow    ~ 2 x H x W flow map

        # output: motion_lookup ~ len(lookup) x 4 x H x W tensor containing desired correlation data for motion encoder

        corr_vol_1, corr_vol_2, corr_vol_4, corr_vol_8 = compute_corr_vol(
            g_0=feat1, g_1=feat2, depth_of_features=3)

        lookup_tensor_1 = self.lookup_from_corr(corr_vol=corr_vol_1, curr_flow=curr_flow, k=0)
        lookup_tensor_2 = self.lookup_from_corr(corr_vol=corr_vol_2, curr_flow=curr_flow, k=1)
        lookup_tensor_4 = self.lookup_from_corr(corr_vol=corr_vol_4, curr_flow=curr_flow, k=2)
        lookup_tensor_8 = self.lookup_from_corr(corr_vol=corr_vol_8, curr_flow=curr_flow, k=3)

        batch_sz = consts.batch_sz

        motion_lookup = torch.empty((batch_sz, consts.len_of_lookup, 4, consts.H // 8, consts.W // 8))

        motion_lookup[:, :, 0, :, :] = lookup_tensor_1.squeeze(2)
        motion_lookup[:, :, 1, :, :] = lookup_tensor_2.squeeze(2)
        motion_lookup[:, :, 2, :, :] = lookup_tensor_4.squeeze(2)
        motion_lookup[:, :, 3, :, :] = lookup_tensor_8.squeeze(2)

        return motion_lookup
