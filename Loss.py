import torch
import torch.nn as nn

class CascadingL1Loss(nn.Module):

    def __init__(self, curr_flow, N, cascading_loss = 0.85):
        super(CascadingL1Loss, self).__init__()
        self.curr_flow = curr_flow
        self.N = N
        self.cascading_loss = cascading_loss

    def forward(self, flow_tup, flow_gt):

        # Inputs: flow_tup ~ f1, f2, f3, ..., fn residual flows via GRU
        #         flow_gt  ~ flow ground truth

        sum = 0
        for i in range(self.N):
            sum += self.cascading_loss ** i * torch.sum(torch.abs(flow_tup[i] + self.curr_flow - flow_gt)).item()

        return sum