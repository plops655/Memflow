import torch
from pathlib import Path

path_to_debug_dir = Path(__file__).parent.parent/'Debug_Files'
def read_from_debug(filename):
    pt_str = filename + ".pt"
    return_tensor = torch.load(str(path_to_debug_dir/pt_str))
    return return_tensor

def write_to_debug(array, filename):
    pt_str = filename + ".pt"
    torch.save(array, str(path_to_debug_dir / pt_str))

if __name__ == '__main__':

    sample = torch.Tensor([1, 2, 3, 4, 5, 6, 7])
    # write_to_debug(sample, 'sample')
    read_tensor = read_from_debug('sample')
    print(read_tensor)