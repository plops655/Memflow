import torch.nn as nn

from core.SKBlock import ConicalBlock

class Flow_Predictor(nn.Module):

    def __init__(self):
        super(Flow_Predictor, self).__init__()

        self.predictor = ConicalBlock(large_kernel=(7,7), small_kernel=(3,3), stride=1, in_channels=64, out_channels=2,
                                      norm_fn=None)

    def forward(self, x):

        x = self.predictor.forward(x)
        return x