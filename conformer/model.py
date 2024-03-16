import torch
import torch.nn as nn
from convolution import ConvSubSampling
from conformer_block import ConformerBlock
import torchaudio.models.conformer

class Decoder(nn.Module):
    def __init__(self, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=bidirectional) # suggest using MHA -> increase params

    def forward(self, x):
        return self.lstm(x) # perform soft max on output


class Conformer(nn.Module):
    """ Encoder Conformer """

    def __init__(self, dropout: float = 0.1):
        """ convolution subsampling """
        super().__init__()
        self.conv_subsampling = ConvSubSampling()  # config

        """ linear """
        self.linear = nn.Linear()

        """ dropout """
        self.dropout = nn.Dropout(p=dropout)

        """ conformer encoder """
        self.conformer_encoder = ConformerBlock()  # encoder

        """ decoder """
        self.decoder = Decoder(bidirectional=True)  #

        """ softmax """
        self.softmax = nn.Softmax()

        """ model chain """
        self.encoder_chain = nn.Sequential(self.conv_subsampling, self.linear,
                                           self.dropout, self.conformer_encoder)

    def forward(self, x):
        hidden_state = self.encoder_chain(x) # get relation ship between audio frame
        output = self.decoder(hidden_state)
        return self.softmax(output) # normalize output to probability with softmax