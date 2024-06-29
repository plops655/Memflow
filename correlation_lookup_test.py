import torch
import torch.nn as nn

from correlation_lookup import Lookup

if __name__ == "__main__":

    size = (10, 10, 3)

    feat1 = torch.empty(size=size)
    feat2 = torch.empty(size=size)
    for i in range(3):
        feat1[:,:,i] = torch.arange(i * 100, (i + 1) * 100).view(10, 10)
        feat2[:,:,i] = torch.arange(i * 100 + 10, (i + 1) * 100 + 10).view(10, 10)

    curr_flow = torch.empty((10, 10, 2))
    curr_flow[:, :, :] = torch.tensor([1, 0])

    lookup = Lookup(feat1=feat1, feat2=feat2, curr_flow=curr_flow, r = 1)

    motion_lookup = lookup.forward()
    print(motion_lookup)


