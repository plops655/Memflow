import torch
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the deviced
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

device = torch.device(device)
x = torch.rand(size=(3, 4)).to(device)

print(x)