import torch
import torch.nn as nn
from utils.consts import H, W

# Computing Visual Similarity Between Frames
def compute_corr_vol(g_0, g_1, depth_of_features=0):
    # Input: g_0, g_1 are R^(batch_sz x D x H / 8 x W / 8) feature frames; depth of features is level to which features are pooled
    # Output: list of correlation Volume of g_0, g_1 in [R^(batch_sz x H / 8 x W / 8 x H / 8 x W / 8), R^(batch_sz x H / 8 x W / 8 x H/16 x W/16),
    #                                                    R^(batch_sz x H / 8 x W / 8 x H/8*2^k x W/8*2^k)] where k is depth of features

    assert depth_of_features >= 0, "corr_vol.py: depth_of_features to be pooled is too small"
    assert H>=8 and W >= 8, f"corr_vol.py: H={H}, W={W} of features not big enough to downsample {depth_of_features} times"

    corr_vol_lst = []
    downsampler = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

    for i in range(0, depth_of_features + 1):

        corr_vol = torch.einsum('nhij,nhkl->nijkl', g_0, g_1)

        corr_vol_lst.append(corr_vol)
        if i < depth_of_features:
            g_1 = downsampler(g_1)

    return corr_vol_lst