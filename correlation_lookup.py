class Lookup:

    def __init__(self, feat1, feat2, curr_flow, r):

        # Inputs: feat1        ~ H x W x D feature map of 1st image
        #         feat2        ~ H x W x D feature map of 2nd image
        #         curr_flow    ~ H x W flow map
        #         r            ~ lookup radius

        self.feat1 = feat1
        self.feat2 = feat2
        self.flow = curr_flow
        self.r = r

    def lookup_i_j(self, i, j, k, r):
        # inputs: i, j ~ the position of lookup
        #         k ~ log base 2 pooling depth of image we lookup
        #         r    ~ L1 radius of lookup

        # output: len(lookup) x 2 tensor

        h, w = H / 2 ** k, W / 2 ** k
        tensor_lst = []
        for y in range(-r, r + 1):
            for x in range(abs(y) - r, r - abs(y) + 1):
                tensor_lst.append(torch.tensor([(i / 2 ** k + flow[i][j][0] / 2 ** k + y) / (h / 2) - 1,
                                                (j / 2 ** k + flow[i][j][1] / 2 ** k + x) / (w / 2) - 1]))

        lookout_tensor = torch.stack(tensor_lst, dim=0)
        return lookout_tensor

    def lookup_from_corr(self, k, flow, r):

        # Input: k    ~ depth of pooling of correlation
        # .       flow ~ H x W tensor of current flow
        #        r    ~ L1 distance to lookup

        # Output: H x W x len(Lookup) x 4

        # Below isn't power efficient

        H, W = flow.shape

        lookup_len = 2 * r ** 2 + 2 * r + 1

        lookup_tensor = torch.empty((H, W, 1, lookup_len))

        for j in range(W):

            sub_grid = torch.empty((H, 1, lookup_len, 2))

            sub_corr_vol = corr_vol[:, j, :, :].unsqueeze(1)

            # make grid by stacking lookups

            for i in range(H):
                sub_grid[i:, :, :, ] = self.lookup_i_j(i=i, j=j, k=k, r=r)

            sub_grid = sub_grid.float()
            sub_corr_vol = sub_corr_vol.float()
            interpolated_lookup = torch.nn.functional.grid_sample(input=sub_corr_vol, grid=sub_grid, mode='bilinear',
                                                                  align_corners=True)
            lookup_tensor[:, j, :, :] = interpolated_lookup.squeeze(1)

        return lookup_tensor

    def forward():

        # output: motion_lookup ~ H x W x 4 x lookup_len tensor containing desired correlation data for motion encoder

        corr_vol_1, corr_vol_2, corr_vol_4, corr_vol_8 = compute_corr_vol(
            g_0=self.feat1, g_1=self.feat2, depth_of_features=3)

        lookup_tensor_1 = self.lookup_from_corr(k=0, flow=self.flow, r=self.r)
        lookup_tensor_2 = self.lookup_from_corr(k=1, flow=self.flow, r=self.r)
        lookup_tensor_4 = self.lookup_from_corr(k=2, flow=self.flow, r=self.r)
        lookup_tensor_8 = self.lookup_from_corr(k=3, flow=self.flow, r=self.r)

        motion_lookup = torch.empty((H, W, 4, lookup_len))

        motion_lookup[:, :, 0, :] = lookup_tensor_1.unsqueeze(2)
        motion_lookup[:, :, 1, :] = lookup_tensor_2.unsqueeze(2)
        motion_lookup[:, :, 2, :] = lookup_tensor_4.unsqueeze(2)
        motion_lookup[:, :, 3, :] = lookup_tensor_8.unsqueeze(2)

        return motion_lookup
