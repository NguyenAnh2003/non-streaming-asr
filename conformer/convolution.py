import torch
import torch.nn as nn

class PointWise1DConv(nn.Module):
    def __init__(self):
        """"""
        super(PointWise1DConv, self).__init__()
        self.conv = nn.Conv1d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class DepthWise1DConv(nn.Module):
    def __init__(self):
        super(DepthWise1DConv, self).__init__()
        self.dw_conv = nn.Conv1d()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)