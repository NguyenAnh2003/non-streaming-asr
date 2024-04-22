import torch
import torch.nn as nn

def get_model_params(model: nn.Module):
    params = [p.nelement() for p in model.parameters()]
    # for i in model.parameters():
    #     print(i.shape)
    return params

if __name__ == "__main__":
    get_model_params()
    pass