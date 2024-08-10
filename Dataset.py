import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import torch.nn.functional as F

from PIL import Image

import cv2 as cv
import numpy as np

import os
from pathlib import Path

import time

class OptFlowDataset(Dataset):

    def __init__(self, img_dir, cutoff=None, transform=None, jump=1):

        self.img_dir = img_dir
        self.image_files = np.sort(np.array([f for f in os.listdir(img_dir) if f.endswith(".png")]))[0:-1:jump]
        self.next_image_files = np.sort(np.array([f for f in os.listdir(img_dir) if f.endswith(".png")]))[1::jump]
        self.cutoff = cutoff
        self.transform = transform

    def __len__(self):

        if not self.cutoff:
            return self.image_files.shape[0]
        return min(self.cutoff, self.image_files.shape[0])

    def __getitem__(self, idx):

        img_name = os.path.join(self.img_dir, self.image_files[idx])
        next_img_name = os.path.join(self.img_dir, self.next_image_files[idx])

        imag = Image.open(img_name)
        next_image = Image.open(next_img_name)

        if self.transform:
            imag = self.transform(imag)
            next_image = self.transform(next_image)

        return imag, next_image

    def get_file_names(self, idx):
        return self.image_files[idx], self.next_image_files[idx]

def scale_dims(dims: tuple[int], output_size):
    assert len(dims) == 2
    assert isinstance(output_size, (int, tuple))

    H, W = dims

    if isinstance(output_size, int):
        if H > W:
            new_H, new_W = output_size * H / W, output_size
        else:
            new_H, new_W = output_size, output_size * W / H
    else:
        new_H, new_W = output_size

    return new_H, new_W

class Rescale_Flow:
    """Rescale and crop the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size


    def __call__(self, flow):

        # Input: flow ~ (H, W, 2) np array for optical flow ground truth
        # Output: rescaled_flow ~ (2, *output_size) scaled torch tensor for optical flow

        flow = ToTensor()(flow)
        flow_y = flow[0, :, :]
        flow_x = flow[1, :, :]

        H, W = flow.shape[1:3]

        if isinstance(self.output_size, int):
            new_H, new_W = scale_dims((H, W), self.output_size)

        reshaped_flow_y = F.interpolate(flow_y, size=self.output_size, mode='bilinear', align_corners=False)
        reshaped_flow_x = F.interpolate(flow_x, size=self.output_size, mode='bilinear', align_corners=False)

        scaled_flow_y = reshaped_flow_y * new_H / H
        scaled_flow_x = reshaped_flow_x * new_W / W

        rescaled_flow = torch.cat((scaled_flow_y, scaled_flow_x), dim=0)
        rescaled_flow = CenterCrop(self.output_size)(rescaled_flow)

        return rescaled_flow


class FlowBasedDataset(OptFlowDataset):

    def __init__(self, img_dir, flow_dir=None, cutoff=None, transform=None, jump=1):

        super(FlowBasedDataset, self).__init__(img_dir, cutoff, transform, jump)
        self.flow_dir = flow_dir
        self.flow_files = np.sort(np.array([f for f in os.listdir(img_dir) if f.endswith(".png")]))  # change to actual ending

    # fix this crap. oof.
    def __getitem__(self, idx):

        image, next_image = super(FlowBasedDataset, self).__getitem__(idx)
        flow_name = os.path.join(self.flow_dir, self.image_files[idx])
        # replace with actual way to open flow file
        flow = Image.open(flow_name)
        rescaled_flow = Rescale_Flow(320)(flow)

        return image, next_image, rescaled_flow


class PSNRBasedDataset(OptFlowDataset):

    def __init__(self, img_dir, cutoff=None, transform=None, jump=1):

        super(PSNRBasedDataset, self).__init__(img_dir, cutoff, transform, jump)


if __name__ == '__main__':

    t = Compose([Resize(320), CenterCrop(320), ToTensor()])

    toyota_img_dir = str(Path(__file__).parent.parent/"TOYOTA/")
    kitti_img_dir = str(Path(__file__).parent.parent/"KITTI/2011_09_26/"
                                                            "2011_09_26_drive_0001_extract/image_00/data")

    # toyota dataset has no flow
    toyota_dataset = PSNRBasedDataset(img_dir=toyota_img_dir, transform=t)

    dataloader = DataLoader(dataset=toyota_dataset, batch_size=32, shuffle=False)

    for batch_idx, (img, next_img) in enumerate(dataloader):
        batch_sz = next_img.shape[0]
        for i in range(batch_sz):
            torch_RGB_img = next_img[i, :, :, :]
            np_RGB_img = (next_img[i, :, :, :].permute((1, 2, 0)).numpy() * 255).astype(np.uint8)
            image = cv.cvtColor(np_RGB_img, cv.COLOR_RGB2BGR)
            cv.imshow("Image", image)
            cv.waitKey(100)

    cv.waitKey(0)
    cv.destroyAllWindows()