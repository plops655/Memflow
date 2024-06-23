import torch
import torch.nn as nn

class Motion_Encoder(nn.Module):

    # why do we use SKBlocks? As an alternative to CNN? Why do we get rid of the GRU from SKFlow?
    #

    def __init__(self, motion_lookup, curr_flow, cost_encoder, flow_encoder, costflow_encoder):
        # Inputs: motion_lookup ~ H x W x 4 x lookup_len tensor containing desired correlation data
        #         cost_encoder  ~ learnable (SK_Block, SK_Block) for cost (why??)
        #         flow_encoder  ~ learnable (SK_Block, SK_Block) for cost

        super(Motion_Encoder, self).__init__()
        self.motion_lookup = motion_lookup
        self.curr_flow = curr_flow
        self.cost_encoder = cost_encoder
        self.flow_encoder = flow_encoder
        self.costflow_encoder = costflow_encoder

    def forward(self):
        # start SK MOE:

        # reshape motion_lookup to 4 x lookup_len_tensor x H x W. What does it mean to have lookup_len_tensor channels?

        encoded_cost = self.cost_encoder(self.motion_lookup)
        encoded_flow = self.flow_encoder(self.curr_flow)

        H, W, _, len_of_lookup = encoded_cost.shape
        H1, W1 = encoded_flow.shape

        assert _ == 4, f"Error: encoded_cost.shape is {encoded_cost.shape}. Third dimension must be 4."
        assert H == H1 and W == W1, "Error: first 2 dimensions of flow and cost don't match"

        # 4 * len_of_lookup x H x W
        encoded_cost_flat = encoded_cost.view((H, W, 4 * len_of_lookup))

        # 2 x H x W
        encoded_flow_flat = encoded_flow.view((2, H, W))

        concat_costflow = torch.cat(
            (self.costflow_encoder(torch.cat((encoded_cost_flat, encoded_flow_flat), dim=0)), encoded_flow_flat), dim=0)

        return concat_costflow
