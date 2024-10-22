import torch
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, module: nn.Module = None, 
                 residual_half_step: float = 0.5):
        super(ResidualConnection, self).__init__()
        self.module = module
        self.half_step = residual_half_step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ :return identity + res_step * module_output """
        identity = x
        module_output = self.module(x)
        return identity + (self.half_step * module_output)