import sys
sys.path.append("./")

import torch
import torch.optim as optim
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torch.utils.data import DataLoader, Subset

from Dataset import PSNRBasedDataset, FlowBasedDataset
from Models import PSNRModel
from Loss import PSNRLoss, CascadingL1Loss

from pathlib import Path

from utils.consts import train_sz, lr, batch_sz

if __name__ == '__main__':

    t = Compose([Resize(320), CenterCrop(320), ToTensor()])

    toyota_img_dir = str(Path(__file__).parent.parent.parent / "TOYOTA/")
    kitti_img_dir = str(Path(__file__).parent.parent.parent / "KITTI/2011_09_26/"
                                                       "2011_09_26_drive_0001_extract/image_00/data")

    # toyota dataset has no flow
    img_dir = toyota_img_dir

    # no flow = PSNRBased
    dataset = PSNRBasedDataset(img_dir=img_dir, transform=t)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_sz, shuffle=False)
    model = PSNRModel()
    criterion = PSNRLoss()

    epochs = 20
    optim_func = optim.Adadelta
    optimizer = optim_func(model.parameters(), lr=lr)

    assert (train_sz + 5) * batch_sz <= len(dataset), len(dataset)

    index_arr = torch.arange(len(dataset) // batch_sz) * batch_sz
    index_arr = index_arr[torch.randperm(len(index_arr))]
    train_index = index_arr[:train_sz]
    eval_index = index_arr[train_sz:]

    train_dataset = Subset(dataset, train_index)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)

    eval_dataset = Subset(dataset, eval_index)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_sz, shuffle=True)

    for epoch_idx, epoch in enumerate(range(epochs)):
        model.train()
        train_loss = 0.0

        for curr_frame, next_frame in train_dataloader:
            curr_frame.requires_grad = True
            next_frame.requires_grad = True

            list_of_delta_f = model(curr_frame, next_frame)
            loss = criterion(list_of_delta_f, curr_frame, next_frame)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for curr_frame, next_frame in eval_dataloader:
                list_of_delta_f = model(curr_frame, next_frame)
                loss = criterion(list_of_delta_f, curr_frame, next_frame)

                eval_loss += loss.item()


        print(f"Epoch {epoch_idx + 1} / {epochs}, "
              f"Training Loss: {train_loss / len(train_dataloader)}, Eval Loss: {eval_loss / len(eval_dataloader)}")









