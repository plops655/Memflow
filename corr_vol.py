import torch
import torch.nn as nn

# Computing Visual Similarity Between Frames
def compute_corr_vol(g_0, g_1, depth_of_features=0):
    # Input: g_0, g_1 are R^(H x W x D) feature frames; depth of features is level to which features are pooled
    # Output: list of correlation Volume of g_0, g_1 in [R^(H x W x H x W), R^(H x W x H/2 x W/2),
    #                                                    R^(H x W x H/2^k x W/2^k)] where k is depth of features

    assert depth_of_features >= 0, "corr_vol.py: depth_of_features to be pooled is too small"
    assert g_0.shape == g_1.shape, f"corr_vol.py: g_0.shape, {g_0.shape}, and g_1.shape, {g_1.shape}, are not the same"
    H, W = g_0.shape[0:2]
    assert H>=8 and W >= 8, f"corr_vol.py: H={H}, W={W} of features not big enough to downsample {depth_of_features} times"

    corr_vol_lst = []
    downsampler = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

    g_1_downsampled = g_1.permute(dims=(2, 0, 1)).unsqueeze(0)
    for i in range(0, depth_of_features + 1):

        corr_vol = torch.einsum('ijh,mhkl->ijkl', g_0, g_1_downsampled)

        corr_vol_lst.append(corr_vol.clone())
        if i < depth_of_features:
            g_1_downsampled = downsampler(g_1_downsampled)

    return corr_vol_lst