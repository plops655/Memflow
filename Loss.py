import torch
import torch.nn as nn

class CascadingL1Loss(nn.Module):

    def __init__(self, N, cascading_loss = 0.85):
        super(CascadingL1Loss, self).__init__()
        self.N = N
        self.cascading_loss = cascading_loss

    def forward(self, curr_flow, flow_tup, flow_gt):

        # Inputs: flow_tup ~ f1, f2, f3, ..., fn residual flows via GRU
        #         flow_gt  ~ flow ground truth

        # Output: Cascading L1 Loss tensor

        aggr_sum = 0
        for i in range(self.N):
            aggr_sum += self.cascading_loss ** (self.N - i) * torch.sum(torch.abs(flow_tup[i] + curr_flow - flow_gt)).item()

        return torch.tensor(aggr_sum)