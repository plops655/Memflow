import torch
import torch.nn as nn
import torch.nn.functional as F

from Helper.psnr import psnr, compute_psnr

from utils.consts import H, W, batch_sz, GRU_iterations

from Helper.debug_read_write import write_to_debug

class CascadingL1Loss(nn.Module):

    def __init__(self, cascading_loss = 0.85):
        super(CascadingL1Loss, self).__init__()
        self.cascading_loss = cascading_loss

    def forward(self, curr_flow, flow_tup, flow_gt):

        # Inputs: flow_tup ~ f1, f2, f3, ..., fn residual flows via GRU ~ [batch_sz x 2 x H / 8 x W / 8, ...]
        #         flow_gt  ~ flow ground truth ~ batch_sz x 2 x H / 8 x W / 8

        # Output: Cascading L1 Loss tensor

        aggr_sum = 0
        for i in range(GRU_iterations):
            aggr_sum += self.cascading_loss ** (GRU_iterations - i) * torch.sum(torch.abs(flow_tup[i] + curr_flow - flow_gt)).item()

        loss = torch.tensor(aggr_sum)
        loss.requires_grad = True
        return loss



def add_flow(img, flw):
    # curr_frame: img ~ batch_sz x D x H x W
    #         flw ~ batch_sz x 2 x H x W

    # Outputs: shifted_img ~ batch_sz x D x H x W

    # D in {1, 3}

    # make base flow grids
    grid_x = torch.zeros(batch_sz, H, W)
    grid_y = torch.zeros(batch_sz, H, W)

    for h in range(H):
        for w in range(W):
            grid_x[:, h, w] = w * torch.ones(batch_sz)
            grid_y[:, h, w] = h * torch.ones(batch_sz)

    # add flow

    grid_x = grid_x + flw[:, 1, :, :]
    grid_y = grid_y + flw[:, 0, :, :]

    ones = torch.ones(batch_sz, H, W)
    grid_x = 2 / (W - 1) * grid_x - ones
    grid_y = 2 / (H - 1) * grid_y - ones

    # make grid

    grid = torch.empty(batch_sz, H, W, 2)
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y

    try:
        assert torch.max(torch.abs(grid)) <= 5
    except AssertionError:
        write_to_debug(grid, "add_flow_grid")
        print("Error in Loss/add_flow")

    out_image = F.grid_sample(img, grid, align_corners=True)

    return out_image

class PSNRLoss(nn.Module):
    def __init__(self, cascading_loss = 0.85):
        super(PSNRLoss, self).__init__()
        self.cascading_loss = cascading_loss

    def forward(self, flow_tup, frame1, frame2):

        # Inputs: flow_tup ~ f1, f2, f3, ..., fn upsampled flows      shape: batch_sz x 2 x H / 8 x W / 8
        #         frames_0   ~ first frame      shape: batch_sz x 3 x H x W
        #         frames_1   ~ frame we try to estimate w.t. flow      shape: batch_sz x 3 x H x W

        # Output: Cascading L1 Loss tensor

        # Cascade -PSNR for each residual flow

        aggr_psnr = 0
        for i in range(GRU_iterations):
            # cast to uint8 during testing
            frame2_estim = add_flow(frame1, flow_tup[i])
            curr_psnr = compute_psnr(frame2, frame2_estim)
            aggr_psnr += curr_psnr * self.cascading_loss ** (GRU_iterations - i)

        loss = torch.tensor(-aggr_psnr)
        loss.requires_grad = True
        return loss