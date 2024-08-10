import torch
from torch.utils.data import DataLoader, Subset
from DeviceLoader import DeviceLoader

def train_test_split(device, dataset, batch_sz, test_size):

    index_arr = torch.arange(len(dataset) // batch_sz) * batch_sz
    index_arr = index_arr[torch.randperm(len(index_arr))]
    index_arr = (index_arr.unsqueeze(1) + torch.arange(4)).flatten()
    train_sz = int(len(index_arr) // batch_sz * (1 - test_size)) * batch_sz
    train_index = index_arr[:train_sz]
    test_index = index_arr[train_sz: len(index_arr) // batch_sz * batch_sz]

    train_dataset = Subset(dataset, train_index)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)
    train_dataloader = DeviceLoader(train_dataloader, device)

    test_dataset = Subset(dataset, test_index)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_sz, shuffle=True)
    test_dataloader = DeviceLoader(test_dataloader, device)

    return train_dataloader, test_dataloader

