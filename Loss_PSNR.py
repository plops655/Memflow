import torch
import torch.nn as nn
import torch.nn.functional as F

from Helper.psnr import psnr, compute_psnr


def add_flow(img, flw):
    # inputs: img ~ D x H x W
    #         flw ~ 1 x H x W x 2

    # Outputs: shifted_img ~ D x H x W

    D, H, W = img.shape

    # make base flow grids
    grid_x = torch.empty(H, W)
    grid_y = torch.empty(H, W)

    for h in range(H):
        for w in range(W):
            grid_x[h, w] = w
            grid_y[h, w] = h

    # add flow

    grid_x = grid_x + flw[:, :, 0]
    grid_y = grid_y + flw[:, :, 1]

    ones = torch.ones(H, W)
    grid_x = 2 / (W - 1) * grid_x - ones
    grid_y = 2 / (H - 1) * grid_y - ones

    # make grid

    grid = torch.empty(1, H, W, 2)
    grid[0, :, :, 0] = grid_x
    grid[0, :, :, 1] = grid_y

    in_image = img.unsqueeze(0)

    out_image = F.grid_sample(in_image, grid, align_corners=True)

    return out_image.squeeze(0)

class PSNRLoss(nn.Module):
    def __init__(self, N, cascading_loss = 0.85):
        super(PSNRLoss, self).__init__()
        self.N = N
        self.cascading_loss = cascading_loss

    def forward(self, flow_tup, frame1, frame2):

        # Inputs: flow_tup ~ f1, f2, f3, ..., fn residual flows via GRU
        #         frame1   ~ first frame      shape: 3 x H x W
        #         frame2   ~ frame we try to estimate w.t. flow    shape: 3 x H x W

        # Output: Cascading L1 Loss tensor

        # Cascade -PSNR for each residual flow

        aggr_psnr = 0
        for i in range(self.N):
            frame2_estim = add_flow(frame1, flow_tup[i]).to(torch.uint8)
            curr_psnr = compute_psnr(frame2, frame2_estim)
            aggr_psnr += curr_psnr * self.cascading_loss ** (self.N - i)

        return torch.tensor(-aggr_psnr)