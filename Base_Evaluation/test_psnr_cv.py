# Take frame of batch_sz = 4 images from TOYOTA
# Replicate psnr remap for each frame
# Output as video in cv
import os

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import cv2 as cv

from pathlib import Path

from Dataset import PSNRBasedDataset
from utils.consts import H, W

def add_flow(img, flw):
    # curr_frame: img ~ D x H x W
    #         flw ~ H x W x 2

    # Outputs: shifted_img ~ D x H x W

    D = img.shape[0]

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

def set_flow() -> torch.Tensor:
    flow = torch.empty((H, W, 2))
    for r in range(H):
        for c in range(W // 2):
            flow[r, c, :] = torch.Tensor([W // 2, 0])
        for c in range(W // 2, W):
            flow[r, c, :] = torch.Tensor([-W // 2, 0])

    return flow

if __name__ == '__main__':

    toyota_img_dir = str(Path(__file__).parent.parent.parent /"TOYOTA/")
    batch_sz = 4
    # Load 0-4 .png
    t = Compose([Resize(320), CenterCrop(320), ToTensor()])
    dataset = PSNRBasedDataset(toyota_img_dir, 4, transform=t)
    for i in range(len(dataset)):
        img, next_img = dataset[i]
        flow = set_flow()
        img_estim = add_flow(img, flow)
        cv_img = cv.cvtColor(img.permute((1, 2, 0)).numpy(), cv.COLOR_RGB2BGR)
        cv_img_estim = cv.cvtColor(img_estim.permute((1, 2, 0)).numpy(), cv.COLOR_RGB2BGR)
        cv_next_img = cv.cvtColor(img.permute((1, 2, 0)).numpy(), cv.COLOR_RGB2BGR)
        cv.imshow("frame", cv_img)
        cv.waitKey(500)
        cv.imshow("estimated frame", cv_img_estim)
        cv.waitKey(500)
        cv.imshow("next frame", cv_next_img)



    cv.waitKey(0)
    cv.destroyAllWindows()



