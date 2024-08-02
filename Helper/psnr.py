import torch
import numpy as np

from utils.consts import H, W

def psnr(ref: torch.Tensor, meas: torch.Tensor, maxVal=255):
    assert torch.Tensor.shape(ref) == torch.Tensor.shape(meas), "Test video must match measured video dimensions"
    dif = (ref - meas).ravel()
    mse = torch.linalg.norm(dif)**2/torch.prod(ref.shape)
    psnr = 10*torch.log10(maxVal**2.0/mse)
    return psnr


def compute_psnr(I_dec, I_ref):
    # Input:  an array, I_dec, representing a decoded image in range [0.0,255.0]
    #         an array, I_ref, representing a reference image in range [0.0,255.0]
    # Output: a float, PSNR, representing the PSNR of the decoded image w.r.t. the reference image (in dB)

    # Your code here:

    MSE = 0.0
    MSE += torch.sum((I_dec - I_ref) ** 2)

    MSE = float(MSE / (3 * H * W))

    PSNR = 10.0 * float(np.log10(255.0 ** 2 / MSE))

    return PSNR