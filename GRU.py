import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.consts import H, W, len_of_lookup

class RAFT_GRU(nn.Module):

    def __init__(self):

        self.Dk = 256
        self.Dv = 4 * len_of_lookup + 4

        C_in = 2 * self.Dk + self.Dv + 64
        C_out = 64

        self.Z = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=(3,3))
        self.R = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=(3,3))
        self.H = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=(3,3))


    def forward(self, x, h):

        # Input: x ~ 1 x (2 * Dk + Dv) x H / 8 x W / 8 concatenation of motion, context, global motion encodings
        #        h ~ 1 x self.hidden_dim x H / 8 x W / 8

        zt = F.sigmoid(self.Z(torch.cat((h, x), dim=1)))
        rt = F.sigmoid(self.R(torch.cat((h, x), dim=1)))
        ht_update = F.tanh(self.H(torch.cat((rt * h, x), dim=1)))

        ones = torch.ones(1, 64, H // 8, W // 8)
        ht = (ones - zt) * h + zt * ht_update

        return ht