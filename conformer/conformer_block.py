import torch
from torch import Tensor
from typing import Tuple, Optional
import torch.nn as nn
from torch.nn import MultiheadAttention
from .modules import ResidualConnection
from .feed_forward import FeedForwardNet
from .convolution import ConvolutionModule

class RelativeMultiHeadedAttention(MultiheadAttention):
    """ implement MHA with Relative Position """

    def __init__(self, embed_dim, num_heads: int = 4):
        """
        :param embed_dim: embedding dimension (input is embedding)
        :param num_heads: number of heads
        """
        super().__init__(embed_dim, num_heads)
        """ u and v are trainable params proposed in Transformer-XL """
        self.u = nn.Parameter()
        self.v = nn.Parameter()
        self.embed_dim = embed_dim
        self.num_heads = num_heads # inherit number of heads


class ConformerBlock(nn.Module):
    """
    This is following Conformer architecture and Feed forward will follow Macaron Net
    module have feed forward module -> half step = 0.5
    multi head attention -> half step = 1
    convolution module -> half step = 1
    """

    def __init__(self, 
                 dropout: float = 0.1, 
                 attention_heads: int = 4, 
                 encoder_dim: int = 144,
                 pw_ksize: int = 1,
                 dw_ksize: int = 31,
                 conv_model_stride: int = 1):
        super().__init__()
        
        # Feed forward net sanwiching acting like point-wise ff network
        """ 1/2 Feed forward """
        self.ff1 = ResidualConnection(module=FeedForwardNet(in_feats=encoder_dim, 
                                                            out_feats=encoder_dim),
                                      residual_half_step=0.5)

        """ Multi-head Attention with APE """
        self.mha = MultiheadAttention(
            num_heads=attention_heads, # default attention heads are 4
            embed_dim=encoder_dim, # embedding dimenssion
            dropout=dropout)

        """ Convolution Module """
        self.conv_module = ResidualConnection(
            module=ConvolutionModule(
                in_channels=encoder_dim,
                out_channels=encoder_dim,
                pointwise_kernel_size=pw_ksize,
                depthwise_kernel_size=dw_ksize,
                stride=conv_model_stride),
            residual_half_step=1.0)

        """ 1/2 Feed forward """
        self.ff2 = ResidualConnection(module=FeedForwardNet(in_feats=encoder_dim, 
                                                            out_feats=encoder_dim),
                                      residual_half_step=0.5)

        """ LayerNorm """
        self.layer_norm = nn.LayerNorm(normalized_shape=encoder_dim)
        
        self.chain = nn.Sequential(self.ff1, self.ff2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ff module - sandwich
        x = self.ff1(x)

        # MHA process
        identity = x
        x = self.layer_norm(x) # layernorm before MHA
        out, _ = self.mha(x, x, x) # Q, K, V
        out = identity + (1.*out)

        # get last hidden state and feed to conv module
        out = self.conv_module(out)
        # out = out.transpose(1, 2) # transpose (batch_size, times, encoder_dim)

        # ff module - sandwich
        out = self.ff2(out)
        
        # normalize distribution of output
        out = self.layer_norm(out)
        
        return out


if __name__ == "__main__":
    encoder_dim = 144
    # batch_size, times, banks*channels
    x = torch.randn(16, 74, 144)
    # batch_size, times, feats = x.size()