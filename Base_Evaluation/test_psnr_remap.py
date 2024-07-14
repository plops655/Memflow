import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

def add_flow(img, flw):
    # inputs: img ~ D x H x W
    #         flw ~ H x W x 2

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

def set_rgbw_frame():

    frame1 = torch.empty((3, 512, 1392))

    for r in range(256):
        for c in range(696):
            frame1[:, r, c] = red
        for c in range(696, 1392):
            frame1[:, r, c] = green

    for r in range(256, 512):
        for c in range(696):
            frame1[:, r, c] = blue
        for c in range(696, 1392):
            frame1[:, r, c] = white

    return frame1


def set_rgbw_flow():

    flow = torch.empty((512, 1392, 2))
    for r in range(512):
        for c in range(696):
            flow[r, c, :] = torch.Tensor([696, 0])
        for c in range(696, 1392):
            flow[r, c, :] = torch.Tensor([-696, 0])

    return flow

if __name__ == "__main__":

    red = torch.Tensor([255, 0, 0])
    green = torch.Tensor([0, 255, 0])
    blue = torch.Tensor([0, 0, 255])
    white = torch.Tensor([255, 255, 255])

    first_frame = set_rgbw_frame()

    plt.imshow(first_frame.permute((1, 2, 0)).to(torch.uint8))

    flow = set_rgbw_flow()

    warped_frame = add_flow(first_frame, flow)

    plt.imshow(warped_frame.permute((1, 2, 0)).to(torch.uint8))

    plt.waitforbuttonpress()



