import torch
import torch.nn as nn

# Computing Visual Similarity Between Frames
def compute_corr_vol(g_0, g_1, depth_of_features=0):
    # Input: g_0, g_1 are R^(H x W x D) feature frames; depth of features is level to which features are pooled
    # Output: list of correlation Volume of g_0, g_1 in [R^(H x W x H x W), R^(H x W x H/2 x W/2),
    #                                                    R^(H x W x H/2^k x W/2^k)] where k is depth of features

    assert depth_of_features >= 0, "depth_of_features to be pooled is too small"

    corr_vol_lst = []

    H, W = g_0.shape[0:2]

    reshaped_tensor = g_0.view(H, 1, W, 1)
    downsample_ave_g1 = reshaped_tensor.sum(dim=(1, 3)) / 2 ** (2 * depth_of_features)
    corr_vol = torch.einsum('ijh,klh->ijkl', g_0, downsample_ave_g1)

    corr_vol_lst.append(corr_vol.clone())

    for i in range(1, depth_of_features + 1):
        corr_vol = nn.Avg2d(kernel_size=2)
        corr_vol_lst.append(corr_vol.clone())

    return corr_vol_lst