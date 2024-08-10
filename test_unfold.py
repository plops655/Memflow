import torch
import torch.nn.functional as F

# Metal Backend device or CPU device
device = "mps" if torch.backends.mps.is_available() else "cpu"

if __name__ == '__main__':

    tensor = torch.empty(4, 2, 40, 40).to(device)
    unfolded_tensor = F.unfold(input=tensor, kernel_size=3, padding=1, stride=1)
    print("torch version:", torch.__version__)