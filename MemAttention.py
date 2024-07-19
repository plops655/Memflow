import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt, log

import utils.consts as consts

class MemAttention(nn.Module):

    def __init__(self):
        super(MemAttention, self).__init__()

        self.alpha = 0

        H, W, len_of_lookup, batch_sz = (consts.H, consts.W, consts.len_of_lookup, consts.batch_sz)

        self.Dk = 256
        self.Dv = 4 * batch_sz * len_of_lookup + 4

        self.key_buffer = torch.empty(0, self.Dk, H // 8, W // 8)
        self.value_buffer = torch.empty(0, self.Dv, H // 8, W // 8)

        self.current_batch = 0

        self.Q = nn.init.xavier_uniform_(torch.empty(self.Dk, self.Dk))
        self.K = nn.init.xavier_uniform_(torch.empty(self.Dk, self.Dk))
        self.V = nn.init.xavier_uniform_(torch.empty(self.Dv, self.Dv))

    def forward(self, fc, fm):

        # Inputs: fc ~ 1 x Dk x H / 8 x W / 8
        #         fm ~ 1 x Dv x H / 8 x W / 8

        # Output: fam ~ 1 x Dv x H / 8 x W / 8

        H, W, L, N = (consts.H, consts.W, consts.L, consts.N)

        matmul = lambda a, b: torch.einsum('ijkl,jm->imkl', a, b)

        q = matmul(fc, self.Q)              # 1 x Dk x H / 8 x W / 8
        k = torch.cat(matmul(fc, self.K), self.key_buffer)      # L x Dk x H / 8 x W / 8
        v = torch.cat(matmul(fm, self.V), self.value_buffer)

        self.current_batch += 1

        if self.current_batch >= L:
            k = k[-L:, :, :, :]
            v = v[-L:, :, :, :]

        self.key_buffer = k
        self.value_buffer = v

        fam = fm + self.alpha * torch.einsum('imkl,mnkl->inkl',
                                             F.softmax(log(L * H // 8 * W // 8 + H // 8 * W // 8, consts.N) / sqrt(self.Dk) * torch.einsum('ijkl,mjkl->imkl', q, k)),
                                       v)

        return fam



