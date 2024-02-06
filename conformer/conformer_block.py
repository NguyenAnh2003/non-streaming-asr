import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from modules import ResidualConnection
from feed_forward import FeedForwardNet
from convolution import ConvolutionModule

class MultiHeadedAttentionRPE(nn.Module):
    def __init__(self, num_heads: int = 4):
        super().__init__()
        """ implement MHA with Relative Position """

    def forward(self, x):
        return x

class ConformerBlock(nn.Module):
    """
    This is following Conformer architecture and Feed forward will follow Macaron Net
    module have feed forward module -> half step = 0.5
    multi head attention -> half step = 1
    convolution module -> half step = 1
    """
    def __init__(self, d_model: int, dropout: float = 0.1, half_step_residual: bool = True, attention_heads: int = 4):
        super().__init__()
        """ 1/2 Feed forward """
        self.ff1 = ResidualConnection(module=FeedForwardNet(),
                                      residual_half_step=0.5)

        """ Multi-head Attention with APE """
        self.mha = ResidualConnection(module=MultiheadAttention(
            num_heads=attention_heads,
            dropout=dropout
        ), residual_half_step=1.0)

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
    mha = MultiheadAttention(embed_dim=300, num_heads=4, dropout=0.1)
    print(f"MHA : {mha}")