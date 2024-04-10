import torch
import torch.nn as nn
from .activations import Swish
from .modules import ResidualConnection

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, 
                 dropout: float = 0.1, 
                 bias: bool = True):
        super().__init__() # inherit Module
        """
        :param in_feats:
        :param out_feats: 
        This FF network consists of LayerNorm -> Linear -> Dropout -> Linear -> Swish
        """
        # layer norm
        self.norm_layer = nn.LayerNorm(normalized_shape=in_feats) # config LayerNorm

        # Swish activation function
        self.swish = Swish() #

        # Alternative activation
        self.relu = nn.ReLU() # alternative for Swish

        # -- --- ---- --- --- ---- -- PointWise FeedForward appear in Transformer https://arxiv.org/abs/1706.03762
        # config in feats and out feats of sub-linear 1 network
        self.sub_linear1 = nn.Linear(in_features=in_feats,
                                     out_features=out_feats, bias=bias)

        # config dropout for common usage in FF block
        self.dropout = nn.Dropout(p=dropout)  # common dropout

        # config in feats and out feats of sub-linear 2 network
        self.sub_linear2 = nn.Linear(in_features=out_feats, 
                                     out_features=in_feats,
                                     bias=bias)  # final Linear layer
        # -- --- ---- --- --- ---- -- PointWise FeedForward

        # combine all these block to form a sequence FF
        self.chain = nn.Sequential(
            self.norm_layer,
            self.sub_linear1,
            self.swish,
            self.dropout,
            self.sub_linear2,
            self.dropout
        )

    def forward(self, x):
        """ input is weights from Linear layer after Dropout """
        return self.chain(x)  # return output of FF network

if __name__ == "__main__":
    ff = FeedForwardNet(300, 100)
    print(f"Feed forward net: {ff}")
    x = torch.randint(0, 100, (81, 300)).float()
    print(f"Shape: {x.shape}")
    print(f"result: {ff(x).shape}")
    
    # with residual connection
    ffr = ResidualConnection(module=ff, residual_half_step=0.5)
    ffr2 = ResidualConnection(module=ff, residual_half_step=0.5)
    print(f"ff residual connection result: {ffr(x).shape}")