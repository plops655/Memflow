import torch

from Helper.debug_read_write import read_from_debug

sub_grid = read_from_debug("sub_grid")
print(torch.any(torch.isnan(sub_grid)))
