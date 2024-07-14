# H, W for image
H = 192
W = 168

# lookup L1 distance
r = 3
len_of_lookup = 2 * r**2 + 2 * r + 1

# length of memory buffer
L = 100

# length of sequencies used for training
N = 100

# Train parameters. N_Sintel, N_KittiFLOW, ...

# Hidden Dimension for GRU

hidden_dim = 64

def set_H(H_val):
    global H
    H = H_val

def set_W(W_val):
    global W
    W = W_val

def set_r_lookup(r_val):
    global r, len_of_lookup
    r = r_val
    len_of_lookup = 2 * r**2 + 2 * r + 1

def set_L(L_val):
    global L
    L = L_val

def set_N(N_val):
    global N
    N = N_val
