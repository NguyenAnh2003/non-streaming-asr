import torch
import torch.nn as nn
from activations import Swish

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__() # inherit Module
        """
        :param input_size: number of weight after dropout
        This FF network consists of LayerNorm -> Linear -> Dropout -> Linear -> Swish
        """

        # LayerNorm explained: https://www.pinecone.io/learn/batch-layer-normalization/
        self.norm_layer = nn.LayerNorm() # config LayerNorm

        # config in feats and out feats of sub-linear 1 network
        self.sub_linear1 = nn.Linear(in_features=input_dim,
                                     out_features=hidden_dim, bias=True)

        # config dropout for common usage in FF block
        self.dropout = nn.Dropout(p=dropout)  # common dropout

        # config in feats and out feats of sub-linear 2 network
        self.sub_linear2 = nn.Linear(in_features=hidden_dim, out_features=input_dim,
                                     bias=True)  # final Linear layer

        # Swish activation function
        self.swish = Swish()

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
    ff = FeedForwardNet()