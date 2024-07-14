from Datasets.BaseDataset import BaseDataset
from pathlib import Path

class ToyotaDataset(BaseDataset):
    def __init__(self, transform=None):

        super(ToyotaDataset, self).__init__(transform)

    def filename(self, i: int) -> str:
        return str(Path(__file__).parent.parent.parent/"TOYOTA/ezgif-frame-") + f"{i:03d}" + ".png"

if __name__ == "__main__":
    sample_dataset = ToyotaDataset()