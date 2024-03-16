import torch
from torch import Tensor
from typing import Tuple, Optional
import torch.nn as nn
from torch.nn import MultiheadAttention
from modules import ResidualConnection
from feed_forward import FeedForwardNet
from convolution import ConvolutionModule

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

    def __init__(self, dropout: float = 0.1, attention_heads: int = 4):
        super().__init__()
        """ 1/2 Feed forward """
        self.ff1 = ResidualConnection(module=FeedForwardNet(),
                                      residual_half_step=0.5)

        """ Multi-head Attention with APE """
        self.mha = ResidualConnection(module=MultiheadAttention(
            num_heads=attention_heads, # default attention heads are 4
            dropout=dropout), residual_half_step=1.0)

        """ Convolution Module """
        self.conv_module = ResidualConnection(module=ConvolutionModule(),
                                              residual_half_step=1.0)

        """ 1/2 Feed forward """
        self.ff2 = ResidualConnection(module=ConvolutionModule(),
                                      residual_half_step=0.5)

        """ LayerNorm """
        self.layer_norm = nn.LayerNorm()

        """ Conformer block with LayerNorm and Chain """
        self.conformer_block = nn.Sequential(self.ff1, self.mha, self.conv_module, self.ff2, self.layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conformer_block(x)


if __name__ == "__main__":
    pass