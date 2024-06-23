import torch
import torch.nn as nn

from utilities import pad_for_conv2d, Res_Layer

class Context_Encoder(nn.Module):

    def __init__(self, frame):
        super(Context_Encoder, self).__init__()
        H, W = frame.shape[0:2]
        self.frame = frame

        self.rl1 = Res_Layer(H=H, W=W, D=3, out_dim1=64, out_dim2=64)
        self.rl2 = Res_Layer(H=H, W=W, D=64, out_dim1=128, out_dim2=128)
        self.rl3 = Res_Layer(H=H, W=W, D=128, out_dim1=192, out_dim2=192)

        ksz = (3, 3)
        padding = pad_for_conv2d(H, W, ksz)

        self.conv_layer = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=ksz, padding=padding)

    def forward(self, small_stride, large_stride):
        # -> ResUnit(64) x 2 -> ResUnit(128) x 2 -> ResUnit(192) x 2 -> Conv3x3(256)
        # Inputs: f1 ~ H x W x 3 frame

        permuted_frame = torch.permute(self.frame, (2, 0, 1)).unsqueeze(0)
        model = (self.rl1, self.rl2, self.rl3)

        # execute model and permute back to form H x W x D

        out = torch.permute(model(permuted_frame).squeeze(0), (1, 2, 0))

        return out