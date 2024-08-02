import torch
import torch.nn as nn
import torch.nn.functional as F

from core.corr_vol import compute_corr_vol

from utils.consts import H, W, batch_sz, r, len_of_lookup

from Helper.debug_read_write import write_to_debug

class Lookup(nn.Module):

    def __init__(self):
        super(Lookup, self).__init__()

    def lookup_i_j(self, i, j, k, curr_flow):
        # curr_frame: i, j ~ the position of lookup
        #         k ~ log base 2 pooling depth of image we lookup
        #         curr_flow    ~ batch_sz x 2 x H / 8 x W / 8 flow map

        # output: batch_sz x len(lookup) x 2 tensor

        h, w = H / (8 * 2 ** k), W / (8 * 2 ** k)
        tensor_lst = []
        ones = torch.ones(batch_sz)
        for y in range(-r, r + 1):
            for x in range(abs(y) - r, r - abs(y) + 1):
                curr_tensor = torch.stack(((j / 2 ** k * ones + curr_flow[:, 1, i, j] / 2 ** k + x * ones) / (w / 2) - ones,
                                        (i / 2 ** k * ones + curr_flow[:, 0, i, j] / 2 ** k + y * ones) / (h / 2) - ones))
                tensor_lst.append(curr_tensor)

        lookout_tensor = torch.stack(tensor_lst, dim=0).permute((2, 0, 1))
        return lookout_tensor

    def lookup_from_corr(self, corr_vol, curr_flow, k):

        # Input: corr_vol ~ batch of correlation cost volume for depth k
        #         curr_flow    ~ batch_sz x 2 x H / 8 x W / 8 flow map
        #        k        ~ depth of pooling

        # Output: lookup_tensors ~ (batch_sz, H / 8, W / 8, len(lookup))

        lookup_tensor = torch.empty(batch_sz, H // 8, W // 8, len_of_lookup)
        for i in range(H // 8):
            for j in range(W // 8):
                sub_corr_vol = corr_vol[:, i, j, :, :].unsqueeze(1)
                sub_grid = self.lookup_i_j(i, j, k, curr_flow)
                sub_grid = sub_grid.unsqueeze(1)
                try:
                    assert torch.max(torch.abs(sub_grid)) <= 5
                except AssertionError:
                    write_to_debug(sub_grid, 'sub_grid')
                    print(f"Assertion Error in correlation_lookup.py/lookup_from_corr where i,j ={i},{j}")
                interpolated_lookup = F.grid_sample(sub_corr_vol, sub_grid, mode='bilinear', align_corners=True)
                lookup_tensor[:, i, j, :] = interpolated_lookup.squeeze((1, 2))
        return lookup_tensor

    def forward(self, feat1, feat2, curr_flow):

        # Inputs: feat1        ~ batch_sz x D x H / 8 x W / 8 feature map of 1st image
        #         feat2        ~ batch_sz x D x H / 8 x W / 8 feature map of 2nd image
        #         curr_flow    ~ batch_sz x 2 x H / 8 x W / 8 flow map

        # curr_flow reshape

        # output: motion_lookup ~ batch_sz x H / 8 x W / 8 x 4 x len(lookup) tensor containing desired correlation data for motion encoder

        corr_vol_1, corr_vol_2, corr_vol_4, corr_vol_8 = compute_corr_vol(
            g_0=feat1, g_1=feat2, depth_of_features=3)

        lookup_tensor_1 = self.lookup_from_corr(corr_vol=corr_vol_1, curr_flow=curr_flow, k=0)
        lookup_tensor_2 = self.lookup_from_corr(corr_vol=corr_vol_2, curr_flow=curr_flow, k=1)
        lookup_tensor_4 = self.lookup_from_corr(corr_vol=corr_vol_4, curr_flow=curr_flow, k=2)
        lookup_tensor_8 = self.lookup_from_corr(corr_vol=corr_vol_8, curr_flow=curr_flow, k=3)

        motion_lookup = torch.empty((batch_sz, H // 8, W // 8, 4, len_of_lookup))

        motion_lookup[:, :, :, 0, :] = lookup_tensor_1
        motion_lookup[:, :, :, 1, :] = lookup_tensor_2
        motion_lookup[:, :, :, 2, :] = lookup_tensor_4
        motion_lookup[:, :, :, 3, :] = lookup_tensor_8

        return motion_lookup
