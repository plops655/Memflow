import torch
import torch.nn as nn
import utils.consts as consts

# Computing Visual Similarity Between Frames
def compute_corr_vol(g_0, g_1, depth_of_features=0):
    # Input: g_0, g_1 are R^(N x D x H x W) feature frames; depth of features is level to which features are pooled
    # Output: list of correlation Volume of g_0, g_1 in [R^(N x H x W x H x W), R^(N x H x W x H/2 x W/2),
    #                                                    R^(N x H x W x H/2^k x W/2^k)] where k is depth of features

    assert depth_of_features >= 0, "corr_vol.py: depth_of_features to be pooled is too small"
    assert consts.H>=8 and consts.W >= 8, f"corr_vol.py: H={consts.H}, W={consts.W} of features not big enough to downsample {depth_of_features} times"

    corr_vol_lst = []
    downsampler = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

    for i in range(0, depth_of_features + 1):

        corr_vol = torch.einsum('nhij,nhkl->nijkl', g_0, g_1)

        corr_vol_lst.append(corr_vol)
        if i < depth_of_features:
            g_1 = downsampler(g_1)

    return corr_vol_lst