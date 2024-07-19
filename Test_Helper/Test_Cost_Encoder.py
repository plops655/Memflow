from Helper.debug_read_write import read_from_debug
from SKBlock import ConicalBlock


if __name__ == '__main__':
    motion_lookup = read_from_debug('motion_lookup')
    curr_flow = read_from_debug('curr_flow')

    # C_out = motion_lookup.shape[2]
    #
    # con1_lookup = ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3), stride=1, in_channels=C_out,
    #                            out_channels=C_out, norm_fn='group')
    # con2_lookup = ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3), stride=1,
    #                            in_channels=C_out, out_channels=C_out, norm_fn='group')
    #
    # encoded_cost = []
    # for i in range(motion_lookup.shape[0]):
    #     val = con1_lookup.forward(motion_lookup[i, :, :, :, :])
    #     val = con2_lookup.forward(val)
    #     encoded_cost.append(val)
    #
    # print(encoded_cost)

    # cost0_lookup = con2_lookup.forward(con1_lookup.forward(motion_lookup[0, :, :, :, :]))
    # print(cost0_lookup)


    C_out = 2

    con1_flow = ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3), stride=1, in_channels=C_out,
                               out_channels=C_out, norm_fn=None)
    con2_flow = ConicalBlock(large_kernel=(7, 7), small_kernel=(3, 3), stride=1,
                               in_channels=C_out, out_channels=C_out, norm_fn=None)

    cost0_flow = con2_flow.forward(con1_flow.forward(curr_flow))

    print(cost0_flow)


