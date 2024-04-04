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
        print(f"Identity: {identity.shape} Module: {module_output.shape} {self.module}")
        return identity + (self.half_step * module_output)

if __name__ == "__main__":
    object = ResidualConnection(nn.Linear(in_features=200, out_features=10), residual_half_step=0.5)
    print(f"Residual output: {object}")