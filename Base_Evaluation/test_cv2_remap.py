from Datasets.dataset_paths import KITTI_data_path
from Datasets.KittiDataset import KittiDataset

import cv2 as cv
import numpy as np


def add_flow(img, flw):
    # curr_frame: img ~ H x W x D
    #         flw ~ H x W x 2

    # Outputs: shifted_img ~ H x W x D

    H, W, D = img.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = grid_x.astype(np.float32) + flw[:,:,1].astype(np.float32)
    map_y = grid_y.astype(np.float32) + flw[:,:,0].astype(np.float32)

    out_img = cv.remap(img.astype(np.float32), map_x, map_y, interpolation=cv.INTER_LINEAR)
    return out_img

def set_rgbw_frame():

    first_frame = np.empty((512, 1392, 3))

    for r in range(256):
        for c in range(696):
            first_frame[r, c, :] = red
        for c in range(696, 1392):
            first_frame[r, c, :] = green

    for r in range(256, 512):
        for c in range(696):
            first_frame[r, c, :] = blue
        for c in range(696, 1392):
            first_frame[r, c, :] = white

    return first_frame

def set_rgbw_flow():

    flow = np.empty((512, 1392, 2))
    for r in range(512):
        for c in range(696):
            flow[r, c, :] = np.array([0, 696])
        for c in range(696, 1392):
            flow[r, c, :] = np.array([0, -696])

    return flow

if __name__ == "__main__":
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    blue = np.array([0, 0, 255])
    white = np.array([255, 255, 255])

    first_frame = set_rgbw_frame()

    cv.imshow("frame", first_frame)

    flow = set_rgbw_flow()

    warped_frame = add_flow(first_frame, flow)

    cv.imshow("warped frame", warped_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()