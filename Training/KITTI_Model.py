from Datasets.dataset_paths import KITTI_data_path
from Datasets.KittiDataset import KittiDataset

from encoder import Encoder
from motion_encoder import Motion_Encoder
from MemAttention import MemAttention
from GRU import RAFT_GRU

import cv2 as cv
from PIL import Image
import numpy as np

import torch.optim as optim

class Kitti_Model:

    def __init__(self, optimizer: optim = optim.Adadelta, loss_cascade_ratio: int = 0.85):

        self.optimizer = optimizer
        self.lcr = loss_cascade_ratio

        self.feature_encoder = Encoder(dropout = 0.1)
        self.context_encoder = Encoder(dropout = 0.1)
        self.motion_encoder = Motion_Encoder()
        self.mem_buffer = MemAttention()
        self.GRU = RAFT_GRU()
        self.loss = ?

