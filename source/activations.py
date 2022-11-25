import torch
import torch.nn as nn

class SoLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input*torch.exp(input)
    