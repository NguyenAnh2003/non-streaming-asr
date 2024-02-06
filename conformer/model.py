import torch
import torch.nn as nn
from .convolution import SubsamplingConv
from .conformer_block import ConformerBlock

class Decoder(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(bidirectional=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.lstm(x)
        return self.softmax(x) # perform soft max on output

class Conformer(nn.Module):
    def __init__(self, dropout: float = 0.1):
        """ convolution subsampling """
        self.conv_subsampling = SubsamplingConv() # config

        """ linear """
        self.linear = nn.Linear()

        """ dropout """
        self.dropout = nn.Dropout(p=dropout)

        """ conformer encoder """
        self.conformer_encoder = ConformerBlock() # encoder

        """ decoder """
        self.decoder = Decoder() #

        """ model chain """
        self.chain = nn.Sequential(self.conv_subsampling, self.linear, self.dropout, self.conformer_encoder, self.decoder)

    def forward(self, x):
        return self.chain(x)