import torch
from torch.utils.data import Dataset

import os
from PIL import Image
from torchvision import transforms

class BaseDataset(Dataset):

    def __init__(self, transform=None):

        self.items = []

        i = 0
        while os.path.isfile(self.filename(i)):
            self.items.append(self.filename(i))
            i += 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item: int) -> torch.Tensor:
        img = Image.open(self.items[item])
        img = transforms.ToTensor()(img)
        return img

    def get_img_size(self):
        img = self[0]
        return img.shape