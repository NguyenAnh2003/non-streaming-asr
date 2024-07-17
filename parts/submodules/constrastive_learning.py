import torch
import torch.nn as nn


class ConstrastiveLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
