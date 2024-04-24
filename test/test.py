import torch
from conformer.convolution import ConvolutionModule, DepthWise1DConv

encoder_dim = 144
# batch_size, n_frames, mel bins

# sample input
x = torch.randn(16, 144, 300)
print(f"Input shape: {x.shape}")

conv_module = DepthWise1DConv(in_channels=encoder_dim,
                                out_channels=encoder_dim,
                                kernel_size=31,
                                padding=15,
                                stride=1)

# conv module
conv_out = conv_module(x)
print(f"Conv out: {conv_out.shape}")