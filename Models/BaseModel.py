from Datasets.ChildDatasets import *

from encoder import Encoder
from correlation_lookup import Lookup
from motion_encoder import Motion_Encoder
from MemAttention import MemAttention
from GRU import RAFT_GRU
from upsample_flow import Upsample_Flow
from Loss import CascadingL1Loss
from Loss_PSNR import PSNRLoss

# from utils.consts import H, W, norm_fn, GRU_iterations, loss_cascade_ratio, dropout, lr
import utils.consts as consts

import torch
import torch.optim as optim

import sys

class BaseModel:

    def __init__(self, dataset, epochs, optimizer):

        self.dataset = dataset

        self.curr_flow = torch.empty((2, consts.H, consts.W))

        self.siamese_encoder = Encoder(dropout=consts.dropout)
        self.context_encoder = Encoder(dropout=consts.dropout)
        self.motion_lookup = Lookup()
        self.motion_encoder = Motion_Encoder()
        self.mem_buffer = MemAttention()
        self.GRU = RAFT_GRU()
        self.upsampler = Upsample_Flow(norm_fn=consts.norm_fn)
        self.loss_normal = CascadingL1Loss(N=consts.GRU_iterations, cascading_loss=consts.loss_cascade_ratio)
        self.loss_psnr = PSNRLoss(N=consts.GRU_iterations, cascading_loss=consts.loss_cascade_ratio)

        # Define the optimizer
        self.optimizer = optimizer(params=list(self.siamese_encoder.parameters()) +
                                          list(self.context_encoder.parameters()) +
                                          list(self.motion_encoder.parameters()) +
                                          list(self.mem_buffer.parameters()) +
                                          list(self.GRU.parameters()) +
                                          list(self.upsampler.parameters()) +
                                          list(self.loss_normal.parameters()), lr=consts.lr)

        self.epochs = epochs

    def slide_2_frames(self):
        for i in range(len(self.dataset) - 1):
            yield [self.dataset[i], self.dataset[i + 1]]

