import torch
import torch.nn.functional as F

from Datasets.dataset_paths import KITTI_data_path
from Datasets.KittiDataset import KittiDataset

import cv2 as cv
import numpy as np

from pathlib import Path

def shift_image(img: torch.Tensor):

    D, H, W = img.shape

    grid = torch.empty((H, W, 2))

    for i in range(H):
        for j in range(W):
            grid[i, j, :] = torch.Tensor([j, i])



    grid[:, :, 0] = 2 / (W - 1) * grid[:, :, 0] - 1
    grid[:, :, 1] = 2 / (H - 1) * grid[:, :, 1] - 1

    img = img.unsqueeze(0).to(torch.float32)
    grid = grid.unsqueeze(0).to(torch.float32)

    output_img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)

    output_img = output_img.squeeze(0)
    return output_img

if __name__ == "__main__":
    dataset = KittiDataset(KITTI_data_path)
    first_frame = cv.imread(dataset[0])
    cv.imshow("Frame", first_frame)

    first_frame_rgb = cv.cvtColor(first_frame, cv.COLOR_BGR2RGB)
    first_frame_tensor = torch.from_numpy(first_frame_rgb).permute((2, 0, 1))

    shifted_frame_tensor = shift_image(first_frame_tensor)

    shifted_frame_numpy = shifted_frame_tensor.permute((1, 2, 0)).detach().cpu().numpy()

    shifted_frame_bgr = np.round(cv.cvtColor(shifted_frame_numpy, cv.COLOR_RGB2BGR)).astype(np.uint8)

    cv.imshow("Shifted Frame", shifted_frame_bgr)

    cv.waitKey(0)
    cv.destroyAllWindows()