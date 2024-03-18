import torch
import torch.nn as nn
from activations import Swish

class PointWise1DConv(nn.Module):
    def __init__(self, in_channels: int = 0, out_channels: int = 1, 
                kernel_size: int = 3, stride: int = 1, padding: int = 1,
                bias: bool = True):
        """ point wise conv """
        super(PointWise1DConv, self).__init__()
        self.pconv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pconv(x)

class DepthWise1DConv(nn.Module):
    
    # audio just have 1 channel
    def __init__(self, in_channels: int , out_channels: int, 
                kernel_size: int, stride: int = 1, padding: int = 1,
                bias: bool = True):
        super(DepthWise1DConv, self).__init__()
        # depth wise -> groups
        self.dw_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, groups=in_channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)

class ConvSubSampling(nn.Module):

    # basic CNN bloack
    # Conv2D sub sampling implemented follow this guide
    # https://www.tutorialexample.com/understand-convolution-subsampling-module-in-conformer-deep-learning-tutorial/
    def __init__(self, in_channels: int, out_channels: int, 
                kernel_size: int = 5, stride: int = 1, padding: int = 0):
        super(ConvSubSampling, self).__init__()
        self.chain = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x - tensor[batch_size, n_frames, fbanks]
        return self.chain(x)


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

        self.glu_activation = nn.GLU()

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
    
if __name__ == "__main__":
    subsampling = ConvSubSampling(in_channels=1, out_channels=16, kernel_size=5)
    x = torch.randint()