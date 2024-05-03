import torch
import torch.nn as nn
from conformer.convolution import DepthWise1DConv
from torchaudio.models import Conformer

if __name__ == "__main__":
    x = torch.randn(16, 144, 256)
    dw = DepthWise1DConv(in_channels=144, out_channels=144, 
                         kernel_size=9, padding=((9 - 1) // 2))
    out = dw(x)
    print(out.shape) 