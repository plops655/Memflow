import torch
import torch.nn as nn

from SKBlock import ConicalBlock
from correlation_lookup import Lookup

from utils.consts import H, W, r, len_of_lookup

class Motion_Encoder(nn.Module):


    def __init__(self):
        # Inputs: motion_lookup ~ H / 8 x W / 8 x 4 x lookup_len tensor containing desired correlation data
        #         cost_encoder  ~ learnable (SK_Block, SK_Block) for cost (why??)
        #         flow_encoder  ~ learnable (SK_Block, SK_Block) for cost

        super(Motion_Encoder, self).__init__()
        C_out = W // 8
        self.motion_lookup = Lookup()
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

    def forward(self, feat1, feat2, curr_flow):
        # start SK MOE:
        # reshape motion_lookup to 4 x lookup_len_tensor x H / 8 x W / 8

        # should do permutations in correlation_lookup for consistency

        encoded_cost = self.cost_encoder(self.motion_lookup.forward(feat1, feat2, curr_flow))
        encoded_flow = self.flow_encoder.forward(curr_flow.unsqueeze(0))

        # 4 * len_of_lookup x H / 8 x W / 8
        encoded_cost_flat = encoded_cost.view(4 * len_of_lookup, H // 8, W // 8)

        # 2 x H / 8 x W / 8
        encoded_flow_flat = encoded_flow.view(2, H // 8, W // 8)

        concat_costflow = torch.cat(
            (self.costflow_encoder(torch.cat((encoded_cost_flat, encoded_flow_flat), dim=0).unsqueeze(0)),
             encoded_flow_flat.unsqueeze(0)), dim=1)

        return concat_costflow
