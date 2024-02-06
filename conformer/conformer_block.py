import torch
import torch.nn as nn
from modules import ResidualConnection
from feed_forward import FeedForwardNet
from convolution import ConvolutionModule

class RelativePositionalEncoding(nn.Module):
    def __init__(self):
        super(RelativePositionalEncoding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return

class MultiHeadedAttentionRPE(nn.Module):
    def __init__(self, attention_heads: int = 4):
        super().__init__()

    def forward(self, x):
        return x

class ConformerBlock(nn.Module):
    """
    This is following Conformer architecture and Feed forward will follow Macaron Net
    module have feed forward module -> half step = 0.5
    multi head attention -> half step = 1
    convolution module -> half step = 1
    """
    def __init__(self, half_step_residual: bool = True, attention_heads: int = 4):
        super().__init__()
        """ 1/2 Feed forward """
        self.ff1 = ResidualConnection(module=FeedForwardNet(), residual_half_step=0.5)

        """ Multi-head Attention with RPE """
        self.mha = ResidualConnection(module=MultiHeadedAttentionRPE(
            attention_heads=attention_heads
        ), residual_half_step=1.0)

        """ Convolution Module """
        self.conv_module = ResidualConnection(module=ConvolutionModule(), residual_half_step=1.0)

        """ 1/2 Feed forward """
        self.ff2 = ResidualConnection(module=ConvolutionModule(), residual_half_step=0.5)

        """ LayerNorm """
        self.layer_norm = nn.LayerNorm()

        """ Conformer block with LayerNorm and Chain """
        self.conformer_block = nn.Sequential(self.ff1, self.mha, self.conv_module, self.ff2, self.layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conformer_block(x)