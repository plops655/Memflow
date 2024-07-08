from Datasets.dataset_paths import KITTI_data_path
import os

from torch.utils.data import Dataset


class KittiDataset(Dataset):

    def __init__(self, img_dir):

        self.img_dir = img_dir
        self.items = []

        i = 0
        while os.path.isfile(self.filename(i)):
            self.items.append(self.filename(i))
            i += 1

    def filename(self, i: int) -> str:
        return self.img_dir + "/" + f"{i:010d}" + ".png"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

if __name__ == "__main__":
    sample_dataset = KittiDataset(KITTI_data_path)
    print(len(sample_dataset))