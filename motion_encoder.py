import torch
import torch.nn as nn

from SKBlock import ConicalBlock

import utils.consts as consts
from Helper.debug_read_write import write_to_debug


class Motion_Encoder(nn.Module):


    def __init__(self):
        # Inputs: motion_lookup ~ lookup_len x 4 x H / 8 x W / 8 tensor containing desired correlation data
        #         cost_encoder  ~ learnable (SK_Block, SK_Block) for cost (why??)
        #         flow_encoder  ~ learnable (SK_Block, SK_Block) for cost

        super(Motion_Encoder, self).__init__()
        C_in = C_out = 4
        self.cost_encoder = nn.Sequential(ConicalBlock(large_kernel=(7,7), small_kernel=(3,3), stride=1, in_channels=C_in,
                                                       out_channels=C_out, norm_fn='group'),
                                          ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3), stride=1,
                                                       in_channels=C_out, out_channels=C_out, norm_fn='group'))

        C_in = C_out = 2
        self.flow_encoder = nn.Sequential(ConicalBlock(large_kernel=(7,7), small_kernel=(3,3), stride=1, in_channels=C_in,
                                                       out_channels=C_out, norm_fn=None),
                                          ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3), stride=1,
                                                       in_channels=C_out, out_channels=C_out, norm_fn=None))

        self.costflow_encoder = ConicalBlock(large_kernel=(7,7), small_kernel=(3,3), stride=1, in_channels = C_out,
                                             out_channels=C_out, norm_fn='group')

    def forward(self, motion_lookup, curr_flow):
        # start SK MOE:
        # reshape motion_lookup to batch_sz x 4 x lookup_len_tensor x H / 8 x W / 8

        # should do permutations in correlation_lookup for consistency

        # encoded_cost = self.cost_encoder(motion_lookup)

        # TODO: DEBUGGING

        write_to_debug(motion_lookup, 'motion_lookup')
        write_to_debug(curr_flow, 'curr_flow')

        batch_sz = consts.batch_sz

        encoded_cost = [self.cost_encoder(motion_lookup[i, :, :, :, :]) for i in range(batch_sz)]

        encoded_flow = self.flow_encoder.forward(curr_flow.unsqueeze(0))

        # 4 * batch_sz * len_of_lookup x H / 8 x W / 8
        encoded_cost_flat = encoded_cost.view(4 * batch_sz * consts.len_of_lookup, consts.H // 8, consts.W // 8)

        # 2 x H / 8 x W / 8
        encoded_flow_flat = encoded_flow.view(2, consts.H // 8, consts.W // 8)

        concat_costflow = torch.cat(
            (self.costflow_encoder(torch.cat((encoded_cost_flat, encoded_flow_flat), dim=0).unsqueeze(0)),
             encoded_flow_flat.unsqueeze(0)), dim=1)

        return concat_costflow
