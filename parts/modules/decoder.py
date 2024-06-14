import torch
from typing import List
from torch.nn import Module
from omegaconf import OmegaConf, DictConfig

class ASRDecoder(Module):
    def __init__(self, conf: DictConfig) -> None:
        super().__init__()
        self.conf = OmegaConf.create(conf)

    def forward(self, x: torch.Tensor):
        return x

