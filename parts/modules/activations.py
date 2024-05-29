import torch
import torch.nn as nn
from torch import Tensor

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x including B (learnable param) """
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()