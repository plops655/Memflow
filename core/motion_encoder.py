import torch
import torch.nn as nn

from core.SKBlock import ConicalBlock


from utils.consts import H, W, batch_sz, r, len_of_lookup


class Motion_Encoder(nn.Module):


    def __init__(self):
        # Inputs: motion_lookup ~ lookup_len x 4 x H / 8 x W / 8 tensor containing desired correlation data
        #         cost_encoder  ~ learnable (SK_Block, SK_Block) for cost (why??)
        #         flow_encoder  ~ learnable (SK_Block, SK_Block) for cost

        super(Motion_Encoder, self).__init__()

        C_in = C_out = 4 * len_of_lookup

        self.cost_encoder = nn.Sequential(ConicalBlock(large_kernel=(7,7), small_kernel=(3,3), stride=1, in_channels=C_in,
                                                       out_channels=C_out, norm_fn='group'),
                                          ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3), stride=1,
                                                       in_channels=C_out, out_channels=C_out, norm_fn='group'))

        C_in = C_out = 2

        self.flow_encoder = nn.Sequential(ConicalBlock(large_kernel=(7,7), small_kernel=(3,3), stride=1, in_channels=C_in,
                                                       out_channels=C_out, norm_fn='batch'),
                                          ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3), stride=1,
                                                       in_channels=C_out, out_channels=C_out, norm_fn='batch'))

        C_in = C_out = 4 * len_of_lookup + 2

        self.costflow_encoder = ConicalBlock(large_kernel=(7,7), small_kernel=(3,3), stride=1, in_channels = C_in,
                                             out_channels=C_out, norm_fn='batch')

    def forward(self, motion_lookup, curr_flow):
        # Inputs: motion_lookup ~ batch_sz x H / 8 x W / 8 x 4 x len(lookup) tensor containing desired correlation data for motion encoder
        #         curr_flow     ~ batch_sz x 2 x H / 8 x W / 8 flow map

        # Output: concat_costflow ~ batch_sz x (4 * len(lookup) + 2) x H / 8 x W / 8

        # idea to make this cleaner: motion_lookup -> (batch_sz, 4 * len_of_lookup, H // 8, W // 8)
        # cost_encoder(motion_lookup)
        # flow_enocder(curr_flow)

        motion_lookup = motion_lookup.view((batch_sz, H // 8, W // 8, 4 * len_of_lookup)).permute((0, 3, 1, 2))
        encoded_cost = self.cost_encoder(motion_lookup)
        encoded_flow = self.flow_encoder(curr_flow)

        concat_costflow = torch.cat((self.costflow_encoder(torch.cat((encoded_cost, encoded_flow), dim=1)),
                                     encoded_flow), dim=1)

        return concat_costflow
