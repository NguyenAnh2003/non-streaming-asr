import torch
import torch.nn as nn

def get_model_params(model: nn.Module):
    params = [p.nelement() for p in model.parameters()]
    return params