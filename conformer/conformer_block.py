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

    def __init__(self, 
                 in_feats: int,
                 out_feats: int,
                 dropout: float = 0.1, 
                 attention_heads: int = 4, 
                 embed_dim: int = 144,
                 padding: int = 2,
                 stride: int = 1):
        super().__init__()
        
        # Feed forward net sanwiching acting like point-wise ff network
        """ 1/2 Feed forward """
        self.ff1 = ResidualConnection(module=FeedForwardNet(in_feats=in_feats, 
                                                            out_feats=out_feats),
                                      residual_half_step=0.5)

        """ Multi-head Attention with APE """
        self.mha = MultiheadAttention(
            num_heads=attention_heads, # default attention heads are 4
            embed_dim=embed_dim, # embedding dimenssion
            dropout=dropout)

        """ Convolution Module """
        self.conv_module = ConvolutionModule(in_channels=encoder_dim,
                                             out_channels=encoder_dim,
                                             stride=stride,
                                             padding=padding)

        """ 1/2 Feed forward """
        self.ff2 = ResidualConnection(module=FeedForwardNet(in_feats=out_feats, 
                                                            out_feats=in_feats),
                                      residual_half_step=0.5)

        """ LayerNorm """
        self.layer_norm = nn.LayerNorm(normalized_shape=in_feats)
        
        self.chain = nn.Sequential(self.ff1, self.ff2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ff module - sandwich
        x = self.ff1(x)

        # MHA process
        identity = x
        out, _ = self.mha(x, x, x) # Q, K, V
        out = identity + (1.*out)
        out = out.transpose(1, 2) # transpose (batch_size, encoder_dim, times)

        # get last hidden state and feed to conv module
        conv_identity = out # (batch_size, encoder_dim, times)
        print(f"Conv identity: {conv_identity.shape}")

        out = self.conv_module(out)
        out += conv_identity 

        print(f"Conv shape: {out.shape}")

        out = out.transpose(1, 2) # transpose (batch_size, times, encoder_dim)
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

    # linear
    # ff_net1 = FeedForwardNet(in_feats=encoder_dim, out_feats=encoder_dim)
    # ff_out1 = ff_net1(x)
    # print(f"Feed forward module out: {ff_out1.shape}")
    
    # MHA
    # mha = nn.MultiheadAttention(num_heads=4, 
    #                             embed_dim=encoder_dim, 
    #                             dropout=0.1, 
    #                             batch_first=True)
    # out, _ = mha(ff_out1, ff_out1, ff_out1)
    # print(f"MHA out: {out.shape} Transpose: {out.transpose(1, 2).shape}")
    
    # Conv module
    # conv_module = ConvolutionModule(in_channels=encoder_dim, 
    #                                 out_channels=encoder_dim, 
    #                                 stride=1, 
    #                                 padding=0, 
    #                                 bias=True)

    # out conv
    # out_conv = conv_module(out.transpose(1, 2))
    # print(f"Conv module out: {out_conv.shape}")
    
    # ff module 2
    # ff_net2 = FeedForwardNet(in_feats=encoder_dim, out_feats=encoder_dim*2)
    # ff_out2 = ff_net2(out_conv)
    # print(f"FF module 2 out: {ff_out2.shape}")
    
    # conformer encoder
    embed_dim = 144
    encoder = ConformerBlock(in_feats=encoder_dim, 
                             out_feats=encoder_dim,
                             embed_dim=encoder_dim,
                             stride=1,
                             padding=2)
    en_out = encoder(x)
    print(f"Encoder out: {en_out.shape}")