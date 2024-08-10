import sys
import time

sys.path.append("./")

import torch

import torch.optim as optim
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torch.utils.data import DataLoader, Subset

from Dataset import PSNRBasedDataset, FlowBasedDataset
from DeviceLoader import DeviceLoader
from Models import PSNRModel
from Loss import PSNRLoss, CascadingL1Loss

from pathlib import Path

from utils.consts import device, H, W, lr, batch_sz
from train_test_split import train_test_split

if __name__ == '__main__':

    t = Compose([Resize(H), CenterCrop(W), ToTensor()])

    toyota_img_dir = str(Path(__file__).parent.parent.parent / "TOYOTA/")
    kitti_img_dir = str(Path(__file__).parent.parent.parent / "KITTI/2011_09_26/"
                                                       "2011_09_26_drive_0001_extract/image_00/data")

    # toyota dataset has no flow
    img_dir = toyota_img_dir

    # no flow = PSNRBased
    dataset = PSNRBasedDataset(img_dir=img_dir, transform=t)
    model = PSNRModel(inTrain=True).to(device)
    criterion = PSNRLoss().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    epochs = 2

    train_dataloader, eval_dataloader = train_test_split(device, dataset, batch_sz, 0.3)
    train_sz = len(train_dataloader)
    eval_sz = len(eval_dataloader)

    train_iterator = iter(train_dataloader)
    eval_iterator = iter(eval_dataloader)

    # split total dataset across epochs

    for epoch_idx, epoch in enumerate(range(epochs)):
        model.train()
        train_loss = 0.0

        time_start = time.time()
        for i in range(train_sz // epochs):

            curr_frame, next_frame = next(train_iterator)

            curr_frame.requires_grad = True
            next_frame.requires_grad = True

            list_of_delta_f = model(curr_frame, next_frame)
            loss = criterion(list_of_delta_f, curr_frame, next_frame)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Finished training with {i}th frames")
            time_end = time.time()
            print(f"Time Elapse since last pass: {time_end - time_start}")
            time_start = time_end

            train_loss += loss.item()

        model.eval()
        eval_loss = 0.0

        time_start = time.time()
        with torch.no_grad():
            for i in range(eval_sz // epochs):

                curr_frame, next_frame = next(eval_iterator)
                list_of_delta_f = model(curr_frame, next_frame)
                loss = criterion(list_of_delta_f, curr_frame, next_frame)

                print(f"Finished training with {i}th frames")
                time_end = time.time()
                print(f"Time Elapse since last pass: {time_end - time_start}")
                time_start = time_end

                eval_loss += loss.item()


        print(f"Epoch {epoch_idx + 1} / {epochs}, "
              f"Training Loss: {train_loss / len(train_dataloader)}, Eval Loss: {eval_loss / len(eval_dataloader)}")

    torch.save(model.state_dict(), str(Path(__file__).parent.parent/"trained_model.pth"))






