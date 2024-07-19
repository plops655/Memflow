import torch
from torch.utils.data import Dataset

import os
from PIL import Image
from torchvision import transforms

class BaseDataset(Dataset):

    def __init__(self, cutoff=None, transform=None):

        # possible transform

        self.items = []
        i = 0

        if not cutoff:
            while os.path.isfile(self.filename(i)):
                self.items.append(self.filename(i))
                i += 1
        else:
            while os.path.isfile(self.filename(i)) and i < cutoff:
                self.items.append(self.filename(i))
                i += 1

        D, H, W = self.gis()

        transform_list = [transforms.CenterCrop((H // 64 * 64, W // 64 * 64)), transforms.ToTensor()]

        if transform:
            transform_list.append(transform)

        self.transform = transforms.Compose(transform_list)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.items[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def gis(self):
        img = Image.open(self.items[0])
        img = transforms.ToTensor()(img)
        return img.shape

    def img_size(self):
        frame1 = self[0]
        return frame1.shape