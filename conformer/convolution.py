import torch
import torch.nn as nn
from activations import Swish
from torchaudio.models import conformer


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
    def __init__(self, in_channels: int, out_channels: int, 
                kernel_size: int = 3, stride: int = 2, padding: int = 0):
        super(ConvSubSampling, self).__init__()
        # stride = 2 -> expirement: using max pooling layer
        self.chain = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x - tensor(batch_size, channels (1), n_frames, fbanks) - input
        batch_size, channels, times, banks = x.size()
        x = x.unsqueeze_(1) # add dimension channel
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, times, -1)
        return self.chain(x)


class ConvolutionModule(nn.Module):
    """ implemented Conv module sequentially """
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()

        self.norm_layer = nn.LayerNorm(normalized_shape=in_channels) # normalize with LayerNorm

        self.point_wise1 = PointWise1DConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                           stride=stride, padding=padding, bias=bias) # customized Pointwise Conv

        self.glu_activation = nn.GLU()

        """ Depthwise Conv 1D """
        self.dw_conv = DepthWise1DConv(in_channels=out_channels, out_channels=out_channels, 
                                       kernel_size=1, padding=padding, bias=bias)

        """ this batch norm layer stand behind the depth wise conv (1D) """
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

        """ Swish activation """
        self.swish = Swish()

        self.point_wise2 = PointWise1DConv(in_channels=out_channels, out_channels=out_channels, kernel_size=1,
                                           stride=1, padding=0, bias=True) #

        self.dropout = nn.Dropout(p=0.1)

        """ sequence of entire convolution """
        self.conv_module = nn.Sequential(
            # self.norm_layer, 
            self.point_wise1, self.glu_activation,
            self.dw_conv, self.batch_norm, self.swish, self.point_wise2,
            self.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ the forward will be present as skip connection """
        identity = x # define identity contain x (input)
        conv_output = self.conv_module(x)
        return identity + conv_output
    
if __name__ == "__main__":
    # print(f"Params: {_params}")
    # conv subsampling
    encoder_dim = 144
    subsampling = ConvSubSampling(in_channels=1, out_channels=encoder_dim,
                                  kernel_size=3, padding=0, stride=2)
    # batch_size, n_frames, mel bins
    
    # sample input
    x = torch.randn(16, 81, 300)
    print(f"In Shape: {x.shape}")
    sub_result = subsampling(x)
    print(f"ConvSubsampling result: {sub_result.shape}")
    # dimension extraction
    batch_size, channels, banks, times = sub_result.size()
    # conv module
    # conv subsampling -> linear -> conv module
    conv_module = ConvolutionModule(in_channels=64, out_channels=128, stride=1,
                                    padding=0, bias=True)
    # print(f"Conv module dict: {conv_module}")

    # sample chain
    sub_result = sub_result.contiguous().view(batch_size, times, -1)
    print(f"Reshaped tensor: {sub_result.shape}")