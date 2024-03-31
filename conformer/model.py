import torch
import torch.nn as nn
from convolution import ConvSubSampling
from conformer_block import ConformerBlock
import torchaudio.models.conformer

class DecoderLSTM(nn.Module):
    def __init__(self, bidirectional: bool = True):
        super().__init__()
        # batch_first -> in & out (batch, seq, feature)
        self.lstm = nn.LSTM(bidirectional=bidirectional, batch_first=True) # suggest using MHA -> increase params

    def forward(self, x):
        return self.lstm(x) # perform soft max on output


class SpeechModel(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int,
                 dropout: float = 0.1, num_layers: int = 1, 
                 encoder_dim: int = 144):
        super().__init__()
        """ :param num_layers -> number of conformer encoders. """
        
        # usually audio have only 1 channel -> in_channel : 1
        self.conv_subsampling = ConvSubSampling(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding)  # config

        # from conv to linear the feature must be flatten
        """ linear """
        # in_feats must be out_channels of CNN, 16 as considered out channels
        self.linear = nn.Linear(in_features=out_channels, out_features=out_channels,
                                bias=True, dtype=torch.float32)

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
        
        # encoder chain -> linear -> dropout -> conformer encoder blocks
        self.input_projection = nn.Sequential(self.linear, self.dropout)

    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        # pipeline -> conv_subsampling -> flatten -> linear -> dropout -> conformer encoder
        x = self.conv_subsampling(x)
        x = self.input_projection(x)

        # output for each conformer block
        for layer in self.conformer_encoder_layers:
            output = layer(x)
        
        return output

    def forward(self, x):
        # forward encoder
        hidden_state = self._forward_encoder(x) # get relation ship between audio frame
        # forward decoder
        output = self.decoder(hidden_state)
        return self.softmax(output) # normalize output to probability with softmax

if __name__ == "__main__":
    x = torch.randn(16, 81, 300)    
    encoder_dim = 144
    subsampling = ConvSubSampling(in_channels=1, out_channels=encoder_dim,
                                  kernel_size=3, padding=0, stride=2)
    # batch_size, n_frames, mel bins
    
    # sample input
    print(f"In Shape: {x.shape}")
    sub_result = subsampling(x)
    print(f"ConvSubsampling result: {sub_result.shape}")
    # dimension extraction
    batch_size, channels, banks, times = sub_result.size()
    # sample chain
    sub_result = sub_result.contiguous().view(batch_size, times, -1)
    print(f"Reshaped tensor: {sub_result.shape}")
    
    # linear
    linear = nn.Linear(in_features=banks*channels, out_features=encoder_dim, bias=True)

    out = linear(sub_result)
    print(f"Linear: {out.shape}")