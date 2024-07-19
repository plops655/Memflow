# H, W for image
H = 0
W = 0

# lookup L1 distance
r = 2
len_of_lookup = 2 * r**2 + 2 * r + 1

norm_fn = 'group'

# length of memory buffer
L = 2

# length of sequences used for training
N = 200

batch_sz = 1

GRU_iterations = 15

loss_cascade_ratio = 0.85

dropout = 0.1

lr = 0.001

# Hidden Dimension for GRU

hidden_dim = 64

def set_H(H_val):
    global H
    H = H_val

def set_W(W_val):
    global W
    W = W_val