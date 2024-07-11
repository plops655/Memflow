from Datasets.dataset_paths import KITTI_data_path
from Datasets.KittiDataset import KittiDataset

from encoder import Encoder
from motion_encoder import Motion_Encoder
from MemAttention import MemAttention
from GRU import RAFT_GRU
from Loss import CascadingL1Loss

import cv2 as cv
from PIL import Image
import numpy as np

from utils.consts import set_H, set_W

import torch
import torch.optim as optim

import sys

class Kitti_Model:

    def __init__(self, H, W, dataset, epochs=20, memory_buffer_length = 2, optimizer: optim = optim.Adadelta, dropout = 0.1, GRU_iterations = 15,
                 loss_cascade_ratio: int = 0.85):

        self.dataset = dataset
        self.epochs = epochs
        self.mem_buffer_length = memory_buffer_length

        self.siamese_encoder = Encoder(dropout = dropout)
        self.context_encoder = Encoder(dropout = dropout)
        self.motion_encoder = Motion_Encoder()
        self.mem_buffer = MemAttention()
        self.GRU = RAFT_GRU()
        self.loss = CascadingL1Loss(N=GRU_iterations, cascading_loss=loss_cascade_ratio)

        if optimizer == "Adadelta":
            optimizer = optim.Adadelta

        # Define the optimizer
        self.optimizer = optimizer(params=list(self.siamese_encoder.parameters()) +
                                          list(self.context_encoder.parameters()) +
                                          list(self.motion_encoder.parameters()) +
                                          list(self.mem_buffer.parameters()) +
                                          list(self.GRU.parameters()) +
                                          list(self.loss.parameters()))


        self.lcr = loss_cascade_ratio

        set_H(H)
        set_W(W)
        self.curr_flow = torch.empty((2, H, W))

    def slide_2_frames(self):
        for i in range(len(self.dataset) - 1):
            yield self.dataset[i:i+1]
    def forward(self):
        for epoch in range(epochs):
            frame_pairs = self.slide_2_frames()
            pair = next(frame_pairs)
            while pair:
                frame1, frame2 = pair
                feat1 = self.siamese_encoder.forward(frame1)
                feat2 = self.siamese_encoder.forward(frame2)
                delta_f = 0
                for GRU_iteration in range(GRU_iterations):
                    predicted_flow = self.curr_flow.detach() + delta_f
                    context_feature = self.context_encoder.forward(frame1)
                    motion_feature = self.motion_encoder.forward(feat1, feat2, predicted_flow)
                    aggr_motion = self.mem_buffer.forward(context_feature, motion_feature)

                    # Concatenate motion_feature, aggr_motion, context_feature
                    torch.cat((motion_feature, aggr_motion, context_feature), dim=1)



def scrape_KITTI_data():

    dataset = KittiDataset(KITTI_data_path)
    return dataset


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Error: Need to input optimizer, dropout, number of GRU iterations")

    epochs, optimizer, dropout, GRU_iterations = sys.argv

    dataset = scrape_KITTI_data()

    # get H, W info

    model = Kitti_Model(dataset=dataset, epochs=epochs, optimizer=optimizer, dropout=dropout, GRU_iterations=GRU_iterations)
