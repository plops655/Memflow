import torch
import torch.nn as nn

from SKBlock import ConicalBlock
from correlation_lookup import Lookup

from utils.consts import H, W, r, len_of_lookup

class Motion_Encoder(nn.Module):


    def __init__(self, feat1, feat2, curr_flow):
        # Inputs: motion_lookup ~ H x W x 4 x lookup_len tensor containing desired correlation data
        #         cost_encoder  ~ learnable (SK_Block, SK_Block) for cost (why??)
        #         flow_encoder  ~ learnable (SK_Block, SK_Block) for cost

        super(Motion_Encoder, self).__init__()
        C_out = feat1.shape[-3]          # C_out = W
        self.motion_lookup = Lookup(feat1, feat2, curr_flow, r=r)
        self.curr_flow = curr_flow
        self.cost_encoder = nn.Sequential(ConicalBlock(large_kernel=(7,7), small_kernel=(3,3), stride=1, in_channels=C_out,
                                                       out_channels=C_out, norm_fn='group'),
                                          ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3), stride=1,
                                                       in_channels=C_out, out_channels=C_out, norm_fn='group'))

        self.flow_encoder = nn.Sequential(ConicalBlock(large_kernel=(7,7), small_kernel=(3,3), stride=1, in_channels=C_out,
                                                       out_channels=C_out, norm_fn='group'),
                                          ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3), stride=1,
                                                       in_channels=C_out, out_channels=C_out, norm_fn='group'))

        self.costflow_encoder = ConicalBlock(large_kernel=(7,7), small_kernel=(3,3), stride=1, in_channels = C_out,
                                             out_channels=C_out, norm_fn='group')

    def forward(self, x):
        # start SK MOE:
        # reshape motion_lookup to 4 x lookup_len_tensor x H x W.

        # should do permutations in correlation_lookup for consistency

        encoded_cost = self.cost_encoder(self.motion_lookup)
        encoded_flow = self.flow_encoder(self.curr_flow.unsqueeze(0))

        # H1, W1 = encoded_flow.shape
        #
        # assert _ == 4, f"Error: encoded_cost.shape is {encoded_cost.shape}. Third dimension must be 4."
        # assert H == H1 and W == W1, "Error: first 2 dimensions of flow and cost don't match"

        # 4 * len_of_lookup x H x W
        encoded_cost_flat = encoded_cost.view(4 * len_of_lookup, H, W)

        # 2 x H x W
        encoded_flow_flat = encoded_flow.view(2, H, W)

        concat_costflow = torch.cat(
            (self.costflow_encoder(torch.cat((encoded_cost_flat, encoded_flow_flat), dim=0)).unsqueeze(0),
             encoded_flow_flat), dim=0)

        return concat_costflow
