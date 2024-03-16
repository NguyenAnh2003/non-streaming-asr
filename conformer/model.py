import torch
import torch.nn as nn
from convolution import ConvSubSampling
from conformer_block import ConformerBlock
import torchaudio.models.conformer

class DecoderLSTM(nn.Module):
    def __init__(self, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=bidirectional) # suggest using MHA -> increase params

    def forward(self, x):
        return self.lstm(x) # perform soft max on output


class Conformer(nn.Module):
    """ Encoder Conformer """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int,
                 dropout: float = 0.1):
        """ convolution subsampling """
        super().__init__()
        self.conv_subsampling = ConvSubSampling(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding)  # config

        self.flatten = nn.Flatten()

        # from conv to linear the feature must be flatten
        self.linear = nn.Linear(in_features=out_channels)

        # dropout
        self.dropout = nn.Dropout(p=dropout)

        # conformer block - encoder
        self.conformer_encoder = ConformerBlock()  # encoder

        # lstm decoder
        self.decoder = DecoderLSTM(bidirectional=True)  #

        # softmax
        self.softmax = nn.Softmax()

        # log-softmax
        self.log_softmax = nn.LogSoftmax()

        # model
        self.encoder_chain = nn.Sequential(self.conv_subsampling, self.flatten,
                                           self.linear, self.dropout,
                                           self.conformer_encoder)

    def forward(self, x):
        hidden_state = self.encoder_chain(x) # get relation ship between audio frame
        output = self.decoder(hidden_state)
        return self.softmax(output) # normalize output to probability with softmax