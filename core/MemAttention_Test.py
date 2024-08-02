import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from utils.consts import H, W, len_of_lookup, L

class MemAttention(nn.Module):

    def __init__(self, fc, fm):

        super(MemAttention, self).__init__()

        self.alpha = 0

        self.Dk = 256
        self.Dv = 4 * len_of_lookup + 4

        self.key_buffer = []
        self.value_buffer = []

        self.fc = fc
        self.fm = fm
        self.fam = torch.empty(1, self.Dv, H, W)

        self.current_batch = 0

        self.Q = nn.init.xavier_uniform_(torch.empty(self.Dk, self.Dk, H, W))
        self.K = nn.init.xavier_uniform_(torch.empty(self.Dk, self.Dk, H, W))
        self.V = nn.init.xavier_uniform_(torch.empty(self.Dv, self.Dv, H, W))


    def forward(self, x):

        # Input: x ~ (1, self.Dk, H, W) from context encoder
        #        x ~ (1, self.Dv, H, W) from motion encoder

        # each key km is l <= L x self.Dk x H x W

        matmul = lambda a, b: torch.einsum('ijkl,jmkl->imkl', a, b)

        q = matmul(self.fc, self.Q)
        k = torch.cat(matmul(self.fc, self.K), torch.stack(self.key_buffer))
        v = torch.cat(matmul(self.fm, self.V), torch.stack(self.value_buffer))

        self.fam = self.fm + self.alpha * matmul(F.softmax(1/sqrt(self.Dk) * torch.einsum('ijkl,mjkl->imkl', q, k)), v)

        # output 1 x self.Dv x H x W tensor









