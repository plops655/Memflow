from Datasets.KittiDataset import KittiDataset
from Datasets.ToyotaDataset import ToyotaDataset

from encoder import Encoder
from motion_encoder import Motion_Encoder
from MemAttention import MemAttention
from GRU import RAFT_GRU
from upsample_flow import Upsample_Flow
from Loss import CascadingL1Loss
from Loss_PSNR import PSNRLoss

from utils.consts import set_H, set_W, set_r_lookup, set_L, set_N

import torch
import torch.optim as optim

import sys

class BaseModel:

    def __init__(self, r, N, L, dataset_str, cutoff=sys.maxsize, epochs=20, norm_fn = 'group', memory_buffer_length = 2,
                 optimizer: str = "Adadelta", dropout = 0.1, GRU_iterations = 15,
                 loss_cascade_ratio: int = 0.85):

        set_r_lookup(r)
        set_L(L)
        set_N(N)

        self.siamese_encoder = Encoder(dropout=dropout)
        self.context_encoder = Encoder(dropout=dropout)
        self.motion_encoder = Motion_Encoder()
        self.mem_buffer = MemAttention()
        self.GRU = RAFT_GRU()
        self.upsampler = Upsample_Flow(norm_fn=norm_fn)
        self.loss_normal = CascadingL1Loss(N=GRU_iterations, cascading_loss=loss_cascade_ratio)
        self.loss_psnr = PSNRLoss(N=GRU_iterations, cascading_loss=loss_cascade_ratio)

        self.curr_flow = torch.empty((2, H, W))

        if optimizer == "Adadelta":
            optimizer = optim.Adadelta
        elif optimizer == "Adagrad":
            optimizer = optim.Adagrad
        elif optimizer == "Adam":
            optimizer = optim.Adam
        elif optimizer == "AdamW":
            optimizer = optim.AdamW
        elif optimizer == "RMSprop":
            optimizer = optim.RMSprop
        elif optimizer == "SGD":
            optimizer = optim.SGD

        # Define the optimizer
        self.optimizer = optimizer(params=list(self.siamese_encoder.parameters()) +
                                          list(self.context_encoder.parameters()) +
                                          list(self.motion_encoder.parameters()) +
                                          list(self.mem_buffer.parameters()) +
                                          list(self.GRU.parameters()) +
                                          list(self.upsampler.parameters()) +
                                          list(self.loss_normal.parameters()))

        if dataset_str == "TOYOTA":
            self.dataset = ToyotaDataset()
        elif dataset_str == "KITTI":
            self.dataset = KittiDataset()

        _, D, H, W = self.dataset.get_img_size()

        set_H(H)
        set_W(W)

        self.H = H
        self.W = W

        self.cutoff = cutoff

        self.epochs = epochs

        self.mem_buffer_length = memory_buffer_length

    def slide_2_frames(self):
        for i in range(len(self.dataset) - 1):
            yield self.dataset[i:i+1]

