import torch
import torch.nn as nn
from ..activations import Swish
from ..res_connection import ResidualConnection

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, 
                 expansion_factor: int = 4,
                 dropout: float = 0.1, 
                 bias: bool = True):
        super().__init__() # inherit Module
        """
        :param in_feats:
        :param out_feats: 
        This FF network consists of LayerNorm -> Linear -> Dropout -> Linear -> Swish
        """
        # layer norm
        self.norm_layer = nn.LayerNorm(normalized_shape=in_feats, 
                                       eps=1e-05) # config LayerNorm

        # Swish activation function
        self.swish = Swish() #

        # Alternative activation
        self.relu = nn.ReLU() # alternative for Swish
        
        # silu
        self.silu = nn.SiLU()

        # -- --- ---- --- --- ---- -- PointWise FeedForward appear in Transformer https://arxiv.org/abs/1706.03762
        # config in feats and out feats of sub-linear 1 network
        self.sub_linear1 = nn.Linear(in_features=in_feats,
                                     out_features=out_feats*4,
                                     bias=bias, dtype=torch.float32)

        # config dropout for common usage in FF block
        self.dropout = nn.Dropout(p=dropout)  # common dropout

        # config in feats and out feats of sub-linear 2 network
        self.sub_linear2 = nn.Linear(in_features=out_feats*4,
                                     out_features=in_feats,
                                     bias=bias)  # final Linear layer
        # -- --- ---- --- --- ---- -- PointWise FeedForward

        # combine all these block to form a sequence FF
        self.chain = nn.Sequential(
            self.norm_layer,
            self.sub_linear1,
            self.silu,
            self.dropout,
            self.sub_linear2,
        )

    def forward(self, x):
        """ input is weights from Linear layer after Dropout """
        return self.chain(x)  # return output of FF network