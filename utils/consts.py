import torch

# Metal Backend device or CPU device
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

# H, W for image
H = 320
W = 320

# lookup L1 distance
r = 2
len_of_lookup = 2 * r**2 + 2 * r + 1

norm_fn = 'group'

# length of memory buffert
L = 2

# length of sequences used for training
train_sz = 16 * 4
validation_sz = 6 * 4

batch_sz = 4

GRU_iterations = 10

loss_cascade_ratio = 0.85

dropout = 0.1

lr = 0.001

# Hidden Dimension for GRU

hidden_dim = 16

def set_H(H_val):
    global H
    H = H_val

def set_W(W_val):
    global W
    W = W_val