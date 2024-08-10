import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.consts import device, H, W, hidden_dim, batch_sz, len_of_lookup
from Helper.pad import pad_for_conv2d

class RAFT_GRU(nn.Module):

    def __init__(self):

        super(RAFT_GRU, self).__init__()

        self.Dk = 256
        self.Dv = 4 * len_of_lookup + 4

        C_in = 2 * self.Dv + self.Dk + hidden_dim
        C_out = hidden_dim

        self.ones = torch.ones(batch_sz, hidden_dim, H // 8, W // 8).to(device).detach()

        self.Z = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=(3,3), padding='same')
        self.R = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=(3,3), padding='same')
        self.H = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=(3,3), padding='same')


    def forward(self, x, h):

        # Inputs: x ~ batch_sz x (2 * Dv + Dk) x H / 8 x W / 8 concatenation of motion, context, global motion encodings
        #         h ~ batch_sz x 64 x H / 8 x W / 8

        # Output: ht ~ batch_sz x 64 x H / 8 x W / 8

        zt = F.sigmoid(self.Z(torch.cat((h, x), dim=1)))
        rt = F.sigmoid(self.R(torch.cat((h, x), dim=1)))
        ht_update = F.tanh(self.H(torch.cat((rt * h, x), dim=1)))

        ht = (self.ones - zt) * h + zt * ht_update

        return ht