from Datasets.BaseDataset import BaseDataset
from pathlib import Path

class KittiDataset(BaseDataset):
    def __init__(self, transform=None):

        super(KittiDataset, self).__init__(transform)

    def filename(self, i: int) -> str:
        return (str(Path(__file__).parent.parent.parent/"KITTI/2011_09_26/2011_09_26_drive_0001_extract/image_00/data") +
                "/" + f"{i:010d}" + ".png")

if __name__ == "__main__":
    sample_dataset = KittiDataset()