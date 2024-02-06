import torch
import torch.nn as nn

""" Relative Positional Encoding """
class RelativePositionalEncoding(nn.Module):
    def __init__(self):
        super(RelativePositionalEncoding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return

""" MHA with RPE """
class MultiHeadedAttentionRPE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

""" Conformer Block """
class ConformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return