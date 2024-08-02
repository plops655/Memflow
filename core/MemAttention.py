import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt, log

from utils.consts import H, W, len_of_lookup, L, train_sz
class MemAttention(nn.Module):

    def __init__(self):
        super(MemAttention, self).__init__()

        self.alpha = 0

        self.Dk = 256
        self.Dv = 4 * len_of_lookup + 4

        self.key_buffer = torch.empty(0, self.Dk, H // 8, W // 8)
        self.value_buffer = torch.empty(0, self.Dv, H // 8, W // 8)

        self.current_batch = 0

        self.Q = nn.init.xavier_uniform_(torch.empty(self.Dk, self.Dk))
        self.K = nn.init.xavier_uniform_(torch.empty(self.Dk, self.Dk))
        self.V = nn.init.xavier_uniform_(torch.empty(self.Dv, self.Dv))

    def forward(self, fc, fm):

        # Inputs: fc ~ batch_sz x Dk x H / 8 x W / 8
        #         fm ~ batch_sz x Dv x H / 8 x W / 8

        # Output: fam ~ batch_sz x Dv x H / 8 x W / 8

        matmul = lambda a, b: torch.einsum('ijkl,jm->imkl', a, b)

        q = matmul(fc, self.Q)                                    # batch_sz x Dk x H / 8 x W / 8
        k = torch.cat((matmul(fc, self.K), self.key_buffer), dim=0)      # (L * batch_sz) x Dk x H / 8 x W / 8
        v = torch.cat((matmul(fm, self.V), self.value_buffer), dim=0)

        self.current_batch += 1

        if self.current_batch >= L:
            k = k[-L:, :, :, :]
            v = v[-L:, :, :, :]

        self.key_buffer = k
        self.value_buffer = v

        fam = fm + self.alpha * torch.einsum('imkl,mnkl->inkl',
                                             F.softmax(log(L * H // 8 * W // 8 + H // 8 * W // 8, train_sz) / sqrt(self.Dk) * torch.einsum('ijkl,mjkl->imkl', q, k), dim=-1),
                                             v)

        return fam



