import torch
import torch.nn as nn
import torch.nn.functional as F

import gc

from math import sqrt, log

from utils.consts import device, H, W, len_of_lookup, batch_sz, L, train_sz
class MemAttention(nn.Module):

    def __init__(self):
        super(MemAttention, self).__init__()

        self.alpha = 0

        self.Dk = 256
        self.Dv = 4 * len_of_lookup + 4

        self.key_buffer = torch.zeros(batch_sz, L, self.Dk, H // 8, W // 8).to(device)
        self.value_buffer = torch.zeros(batch_sz, L, self.Dv, H // 8, W // 8).to(device)

        self.current_batch = 0

        self.Q = nn.init.xavier_uniform_(torch.empty(self.Dk, self.Dk)).to(device)
        self.K = nn.init.xavier_uniform_(torch.empty(self.Dk, self.Dk)).to(device)
        self.V = nn.init.xavier_uniform_(torch.empty(self.Dv, self.Dv)).to(device)

    def forward(self, fc, fm):

        # Inputs: fc ~ batch_sz x Dk x H / 8 x W / 8
        #         fm ~ batch_sz x Dv x H / 8 x W / 8

        # Output: fam ~ batch_sz x Dv x H / 8 x W / 8

        matmul = lambda a, b: torch.einsum('ijkl,jm->imkl', a, b)

        q = matmul(fc, self.Q)                                    # batch_sz x Dk x H / 8 x W / 8

        # TEST FOR MEMORY LEAK:

        # self.key_buffer = torch.cat((matmul(fc, self.K), self.key_buffer), dim=0)      # (L * batch_sz) x Dk x H / 8 x W / 8
        # self.value_buffer = torch.cat((matmul(fm, self.V), self.value_buffer), dim=0)

        # for i in range(1, L):
        #     self.key_buffer[i - 1] = self.key_buffer[i]
        #     self.value_buffer[i - 1] = self.value_buffer[i]
        # self.key_buffer[L - 1] = matmul(fc, self.K).squeeze(0)
        # self.value_buffer[L - 1] = matmul(fm, self.V).squeeze(0)

        for i in range(1, L):
            self.key_buffer[:, i - 1, :, :, :] = self.key_buffer[:, i, :, :, :].detach()
            self.value_buffer[:, i - 1, :, :, :] = self.value_buffer[:, i, :, :, :].detach()

        self.key_buffer[:, L - 1, :, :, :] = matmul(fc, self.K).detach()
        self.value_buffer[:, L - 1, :, :, :] = matmul(fm, self.V).detach()

        self.current_batch += 1

        fam = fm + self.alpha * torch.einsum('imkl,imnkl->inkl',
                                             F.softmax(log(L * H // 8 * W // 8 + H // 8 * W // 8, train_sz) / sqrt(self.Dk) * torch.einsum('ijkl,imjkl->imkl', q, self.key_buffer), dim=-1),
                                             self.value_buffer)

        return fam



