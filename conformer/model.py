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


class SpeechModel(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int,
                 dropout: float = 0.1, num_layers: int = 1):
        super().__init__()
        """ :param num_layers -> number of conformer encoders. """
        
        # usually audio have only 1 channel -> in_channel : 1
        self.conv_subsampling = ConvSubSampling(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding)  # config

        self.flatten = nn.Flatten()

        # from conv to linear the feature must be flatten
        """ linear """
        self.linear = nn.Linear(in_features=out_channels)

        """ dropout """
        self.dropout = nn.Dropout(p=dropout)

        """ conformer encoder with layers """
        self.conformer_encoder_layers = nn.ModuleList([
            ConformerBlock() for _ in range(num_layers)]) #

        """ decoder """
        self.decoder = DecoderLSTM(bidirectional=True)  #

        """ softmax """
        self.softmax = nn.Softmax()

        """ log softmax """
        # self.log_softmax = nn.LogSoftmax()

        """ model chain """
        self.encoder_chain = nn.Sequential(self.conv_subsampling, self.flatten,
                                           self.linear, self.dropout,
                                           self.conformer_encoder)

    def forward(self, x):
        # forward encoder
        hidden_state = self.encoder_chain(x) # get relation ship between audio frame
        # forward decoder
        output = self.decoder(hidden_state)
        return self.softmax(output) # normalize output to probability with softmax

if __name__ == "__main__":
    x = torch.randn(16, 1, 340, 81)
    model = SpeechModel(in_channels=1)