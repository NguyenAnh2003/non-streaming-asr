import torch

# Create two tensors
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# Perform matrix multiplication using einsum
result = torch.einsum('ij,jk->ik', A, B)
print(result)