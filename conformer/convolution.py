import torch
import torch.nn as nn
from .activations import Swish

class PointWise1DConv(nn.Module):
    def __init__(self, in_channels: int = 0, out_channels: int = 1, 
                kernel_size: int = 1, stride: int = 1, padding: int = 1,
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
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                kernel_size: int = 1, stride: int = 1, padding: int = 1,
                bias: bool = True):
        super(DepthWise1DConv, self).__init__()
        # depth wise -> groups
        self.dw_conv = nn.Conv1d(in_channels=in_channels, groups=in_channels, 
                                out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)

class ConvSubSampling(nn.Module):

    # Conv2D sub sampling implemented follow this guide
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                kernel_size: int = 3, 
                stride: int = 1,
                padding: int = 1):
        super(ConvSubSampling, self).__init__()
        self.chain = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, 
                      stride=stride, 
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, 
                      stride=stride, 
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in - (batch_size, channels (1), n_frames, fbanks)
        x = x.unsqueeze(1)
        # conv subsampling
        x = self.chain(x) # convovle the input

        # process product dimension - (batch_size, fbanks, n_frames)
        batch_size, channel, n_frames, fbanks = x.size()
        x = x.permute(0, 2, 1, 3)
        
        # (batch_size, times, channels*fbanks)
        out = x.contiguous().view(batch_size, n_frames, -1)
        return out


class ConvolutionModule(nn.Module):
    """ implemented Conv module sequentially """
    def __init__(self, in_channels: int, 
                 out_channels: int,
                 stride: int = 1, 
                 padding: int = 0, 
                 kernel_size: int = 1,
                 bias: bool = True):
        super().__init__()

        # in_channels can be considered as encoder_dim
        self.norm_layer = nn.LayerNorm(normalized_shape=in_channels) # normalize with LayerNorm

        self.point_wise1 = PointWise1DConv(in_channels=in_channels, 
                                           out_channels=out_channels*2, # duoble out channels
                                           kernel_size=kernel_size,
                                           stride=stride, 
                                           padding=padding, 
                                           bias=bias) # customized Pointwise Conv

        self.glu_activation = nn.GLU(dim=1)

        """ Depthwise Conv 1D """
        self.dw_conv = DepthWise1DConv(in_channels=out_channels,
                                       out_channels=out_channels, 
                                       kernel_size=kernel_size, 
                                       padding=((kernel_size-1) // 2),
                                       bias=bias)

        """ this batch norm layer stand behind the depth wise conv (1D) """
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

        """ Swish activation """
        self.swish = Swish()

        self.point_wise2 = PointWise1DConv(in_channels=out_channels, 
                                           out_channels=out_channels, 
                                           kernel_size=kernel_size,
                                           stride=1, 
                                           padding=padding,
                                           bias=True) #

        self.dropout = nn.Dropout(p=0.1)

        """ sequence of entire convolution """
        self.conv_module = nn.Sequential(
            self.point_wise1,
            self.glu_activation,
            self.dw_conv, 
            self.batch_norm, 
            self.swish, 
            self.point_wise2,
            self.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_layer(x)
        x = x.contiguous().transpose(1, 2) # transpose (B, D, L)
        """ the forward will be present as skip connection """
        conv_output = self.conv_module(x)
        return conv_output.contiguous().transpose(1, 2)
    
if __name__ == "__main__":
    # print(f"Params: {_params}")
    # conv subsampling
    encoder_dim = 144
    # batch_size, n_frames, mel bins
    
    # sample input
    x = torch.randn(16, 144, 300)
    print(f"Input shape: {x.shape}")
    
    conv_module = ConvolutionModule(in_channels=encoder_dim,
                                    out_channels=encoder_dim,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1)

    # conv module
    conv_out = conv_module(x)
    print(f"Conv out: {conv_out.shape}")

    # squeeze