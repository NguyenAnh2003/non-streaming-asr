import torch
import torch.nn as nn
from .activations import Swish

class PointWise1DConv(nn.Module):
    def __init__(self):
        """"""
        super(PointWise1DConv, self).__init__()
        self.conv = nn.Conv1d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class DepthWise1DConv(nn.Module):
    def __init__(self):
        super(DepthWise1DConv, self).__init__()
        self.dw_conv = nn.Conv1d()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)

class SubsamplingConv(nn.Module):
    def __init__(self):
        super(SubsamplingConv, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvolutionModule(nn.Module):
    """ implemented Conv module sequentially """
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int, padding: int, bias: bool):
        super().__init__()
        """ 
        the point wise conv -> depth wise conv called separable convolution
        follow this guide https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728 
        """
        self.norm_layer = nn.LayerNorm() # normalize with LayerNorm

        self.point_wise1 = PointWise1DConv(in_channels=in_channels, stride=stride,
                                           padding=padding, bias=bias) # customized Pointwise Conv

        self.glu_activation = nn.GLU() # customized GLU

        """ Depthwise Conv 1D """
        self.dw_conv = DepthWise1DConv()

        """ this batch norm layer stand behind the depth wise conv (1D) """
        self.batch_norm = nn.BatchNorm1d()

        """ Swish activation """
        self.swish = Swish()

        self.point_wise2 = PointWise1DConv() #

        self.dropout = nn.Dropout(p=0.1)

        """ sequence of entire convolution """
        self.conv_module = nn.Sequential(
            self.norm_layer, self.point_wise1, self.glu_activation,
            self.dw_conv, self.batch_norm, self.swish, self.point_wise2,
            self.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ the forward will be present as skip connection """
        identity = x # define identity contain x (input)
        conv_output = self.conv_module(x)
        return identity + conv_output