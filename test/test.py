import torch
x1 = torch.randn(16, 144, 74)
x2 = torch.randn(16, 144, 47)
x = x1 + x2
print(f"cc: {x.shape}")