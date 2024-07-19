from Datasets.BaseDataset import BaseDataset
from pathlib import Path

class ToyotaDataset(BaseDataset):
    def __init__(self, cutoff=None, transform=None):
        super(ToyotaDataset, self).__init__(cutoff, transform)
        self.has_flow = False

    def filename(self, i: int) -> str:
        return str(Path(__file__).parent.parent.parent/"TOYOTA/ezgif-frame-") + f"{(i + 1):03d}" + ".png"

class KittiDataset(BaseDataset):
    def __init__(self, cutoff=None, transform=None):
        super(KittiDataset, self).__init__(cutoff, transform)
        self.has_flow = True

    def filename(self, i: int) -> str:
        return (str(Path(__file__).parent.parent.parent/"KITTI/2011_09_26/2011_09_26_drive_0001_extract/image_00/data") +
                "/" + f"{i:010d}" + ".png")