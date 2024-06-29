import torch
import torch.nn as nn
import numpy
import os
from pathlib import Path

from SKBlock import *

if __name__ == "__main__":

    directory = Path(__file__).parent / "SK_Block_test_outfiles/"

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    def find_num_out_channels(in_channels):
        if in_channels < 4:
            return 16
        return in_channels * 2

    shapes = [(3,7,7), (16,10,10), (32,20,10), (64,19,71), (128, 7, 23)]
    blocks = [DirectBlock, ParallelBlock, FunnelBlock, ConicalBlock]

    large_kernel = (7,7)
    small_kernel = (3,3)
    stride = 1

    for shape in shapes:
        test_tensor = torch.randn(shape)
        in_channels = test_tensor.shape[-3]
        out_channels = find_num_out_channels(in_channels)
        for Block in blocks:

            sk_block = Block(large_kernel=large_kernel, small_kernel=small_kernel, stride=stride, out_channels=out_channels)
            transformed_tensor = sk_block.forward(test_tensor)

            block_name = Block.name
            in_fname = str(directory) + "/" + block_name + ",in_shape:" + str(shape) + ",out_shape:" + str(transformed_tensor.shape) + ",in"
            out_fname = str(directory) + "/" + block_name + ",in_shape:" + str(shape) + ",out_shape:" + str(transformed_tensor.shape) + ",out"

            # assert H, W not affected
            try:
                assert transformed_tensor.shape[-2:] == test_tensor.shape[-2:], f"H, W not same for output shape: {transformed_tensor.shape}, input shape: {shape}"
            except AssertionError as e:
                if not os.path.exists(in_fname + ",HW.pt"):
                    with open(in_fname + ",HW.pt", "w+"):
                        pass
                if not os.path.exists(out_fname + ",HW.pt"):
                    with open(out_fname + ",HW.pt", "w+"):
                        pass
                torch.save(test_tensor, in_fname)
                torch.save(transformed_tensor, out_fname)

            # assert desired number of out_dimensions
            try:
                assert transformed_tensor.shape[-3] == test_tensor.shape[-3], f"channels not same for output: {transformed_tensor.shape[-3]}, input: {test_tensor.shape[-3]}"
            except AssertionError as e:
                if not os.path.exists(in_fname + ",channels.pt"):
                    with open(in_fname + ",channels.pt", "w+"):
                        pass
                if not os.path.exists(out_fname + ",channels.pt"):
                    with open(out_fname + ",channels.pt", "w+"):
                        pass
                torch.save(test_tensor, in_fname)
                torch.save(transformed_tensor, out_fname)

            # assert number of dimensions of transformed_tensor is same as test_tensor

            try:
                assert len(transformed_tensor.shape) == len(test_tensor.shape), f"number of dimensions not same for output: {len(transformed_tensor.shape)}, input: {len(shape)}"
            except AssertionError as e:
                if not os.path.exists(in_fname + ",numDimensions.pt"):
                    with open(in_fname + ",numDimensions.pt", "w+"):
                        pass
                if not os.path.exists(out_fname + ",numDimensions.pt"):
                    with open(out_fname + ",numDimensions.pt", "w+"):
                        pass
                torch.save(test_tensor, in_fname)
                torch.save(transformed_tensor, out_fname)


