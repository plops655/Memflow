import sys
sys.path.append("./")

from Datasets.ChildDatasets import *

# from Models.FlowModel import FlowModel
from Models.PSNRModel import PSNRModel

from utils.consts import set_H, set_W

import torch.optim as optim

if __name__ == "__main__":

    # first dataset

    # dataset_str = input("Enter name of dataset from: TOYOTA, KITTI: ")
    # cutoff = input("Enter cutoff if relevant. Press <Enter> to skip.")
    #
    # if cutoff:
    #     cutoff = int(cutoff)
    #
    # if dataset_str == "TOYOTA":
    #     dataset = ToyotaDataset(cutoff=cutoff)
    # elif dataset_str == "KITTI":
    #     dataset = KittiDataset(cutoff=cutoff)

    # Delete once done debugging
    #####################################
    dataset = ToyotaDataset()
    #####################################

    has_flow = dataset.has_flow

    # set dimensions H, W

    D, H, W = dataset.img_size()
    set_H(H)
    set_W(W)

    # next model

    # print("Enter model inputs")
    #
    # epochs = input("epochs (default: 20): ")
    # if not epochs:
    #     epochs = 20
    #
    # optimizer_str = input("optimizer (default: Adadelta): ")
    # if not optimizer_str:
    #     optimizer_str = "Adadelta"
    #
    # optimizer = None
    #
    # if optimizer_str == "Adadelta":
    #     optimizer = optim.Adadelta
    # elif optimizer_str == "Adagrad":
    #     optimizer = optim.Adagrad
    # elif optimizer_str == "Adam":
    #     optimizer = optim.Adam
    # elif optimizer_str == "AdamW":
    #     optimizer = optim.AdamW
    # elif optimizer_str == "RMSprop":
    #     optimizer = optim.RMSprop
    # elif optimizer_str == "SGD":
    #     optimizer = optim.SGD


    # Delete once done debugging
    #####################################
    epochs = 20
    optimizer = optim.Adadelta
    #####################################
    print("Default dropout is 0.1 and lr is 0.001")


    model = None

    if has_flow:
        pass
        # model = FlowModel(dataset, epochs, optimizer)
    else:
        model = PSNRModel(dataset, epochs, optimizer)

    model.forward_loss_psnr()



