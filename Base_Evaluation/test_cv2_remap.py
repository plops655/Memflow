from Datasets.dataset_paths import KITTI_data_path
from Datasets.KittiDataset import KittiDataset

import cv2 as cv
import numpy as np


def add_flow(img, flw):
    # inputs: img ~ H x W x D
    #         flw ~ H x W x 2

    # Outputs: shifted_img ~ H x W x D

    H, W, D = img.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = grid_x.astype(np.float32) + flw[:,:,0].astype(np.float32)
    map_y = grid_y.astype(np.float32) + flw[:,:,1].astype(np.float32)

    out_img = cv.remap(img.astype(np.float32), map_x, map_y, interpolation=cv.INTER_LINEAR)
    return out_img

if __name__ == "__main__":
    dataset = KittiDataset(KITTI_data_path)

    first_frame = cv.imread(dataset[0])
    cv.imshow("frame", first_frame)

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    second_frame = cv.imread(dataset[1])

    gray = cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)

    next_frame = add_flow(first_frame, flow)
    next_frame = next_frame.astype(np.uint8)
    cv.imshow("Next frame", next_frame)

    cv.waitKey(0)
    cv.destroyAllWindows()