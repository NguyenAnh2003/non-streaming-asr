import torch
x1 = torch.randn(16, 74, 144)
x2 = torch.randn(16, 37, 144)
x = x1 + x2
print(f"cc: {x.shape}")