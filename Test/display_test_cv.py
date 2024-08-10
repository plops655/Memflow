import sys
sys.path.append("./")

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torch.utils.data import DataLoader, Subset
import cv2 as cv


from Dataset import PSNRBasedDataset, FlowBasedDataset
from Models import PSNRModel
from Loss import PSNRLoss, CascadingL1Loss

from pathlib import Path

from utils.consts import H, W, train_sz, lr, batch_sz

def add_flow(img, flw):
    # curr_fr: img ~ D x H x W
    #         flw ~ 2 x H x W

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

    grid_x = grid_x + flw[1, :, :]
    grid_y = grid_y + flw[0, :, :]

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

def display(curr_fr, next_fr, flow):

    cv_img = cv.cvtColor(curr_fr.permute((1,2,0)).cpu().numpy(), cv.COLOR_RGB2BGR)
    cv.imshow("Current Frame", cv_img)
    frame_estim = add_flow(curr_fr, flow)
    cv_img_estim = cv.cvtColor(frame_estim.permute((1,2,0)).cpu().numpy(), cv.COLOR_RGB2BGR)
    cv.imshow("Estimated Frame", cv_img_estim)
    cv_next_img = cv.cvtColor(next_fr.permute((1,2,0)).cpu().numpy(), cv.COLOR_RGB2BGR)
    cv.imshow("Next Frame", cv_next_img)

def display_batch(curr_frs, next_frs, flow):

    for i in range(batch_sz):
        display(curr_frs[i, :, :, :], next_frs[i, :, :, :], flow[i, :, :, :])
        cv.waitKey(100)

if __name__ == '__main__':

    # Train KITTI Dataset

    t = Compose([Resize(H), CenterCrop(W), ToTensor()])

    toyota_img_dir = str(Path(__file__).parent.parent.parent / "TOYOTA/")
    kitti_img_dir = str(Path(__file__).parent.parent.parent / "KITTI/2011_09_26/"
                                                              "2011_09_26_drive_0001_extract/image_00/data")
    # img_dir = kitti_img_dir
    img_dir = toyota_img_dir

    # no flow = PSNRBased
    test_dataset = PSNRBasedDataset(img_dir=img_dir, transform=t,jump=5)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)
    model = PSNRModel(inTrain=False)
    state_dict = torch.load(str(Path(__file__).parent.parent/"trained_model.pth"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    criterion = PSNRLoss()

    test_loss = 0
    with torch.no_grad():
        for curr_frame, next_frame in test_dataloader:
            delta_f = model(curr_frame, next_frame)
            display_batch(curr_frs=curr_frame, next_frs=next_frame, flow=delta_f)




