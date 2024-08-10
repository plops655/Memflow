import sys
sys.path.append("./")

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torch.utils.data import DataLoader, Subset

from Dataset import PSNRBasedDataset, FlowBasedDataset
from Models import PSNRModel
from Loss import PSNRLoss, CascadingL1Loss

from pathlib import Path

from utils.consts import H, W, train_sz, lr, batch_sz

if __name__ == '__main__':

    # Train KITTI Dataset

    t = Compose([Resize(H), CenterCrop(W), ToTensor()])

    kitti_img_dir = str(Path(__file__).parent.parent.parent / "KITTI/2011_09_26/"
                                                              "2011_09_26_drive_0001_extract/image_00/data")
    img_dir = kitti_img_dir

    # no flow = PSNRBased
    test_dataset = PSNRBasedDataset(img_dir=img_dir, transform=t,jump=5)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)
    model = PSNRModel(inTrain=False)
    model = model.load_state_dict(torch.load(str(Path(__file__).parent.parent/"trained_model.pth")))
    criterion = PSNRLoss()

    test_loss = 0
    with torch.no_grad():
        for curr_frame, next_frame in test_dataloader:
            delta_f = model(curr_frame, next_frame)
            loss = criterion(delta_f, curr_frame, next_frame)
            test_loss += loss.item()

    print(f"Overall loss: {test_loss / len(test_dataset)}")

