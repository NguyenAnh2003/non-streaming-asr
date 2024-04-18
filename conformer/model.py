import torch
import torch.nn as nn
from .convolution import ConvSubSampling
from .conformer_block import ConformerBlock

class DecoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 bias: bool, bidirectional: bool = True,
                 batch_first: bool = True,
                 d_model: int = 144,
                 num_classes: int = 0):
        super().__init__()
        # batch_first -> in & out (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            bias=bias,
                            batch_first=batch_first,
                            bidirectional=bidirectional)

        self.output_projection = nn.Linear(in_features=d_model*2,
                                           out_features=num_classes,
                                           bias=bias)

    def forward(self, x):
        out, _ = self.lstm(x) # taking output - prediction from LSTM
        output = self.output_projection(out) # projection the output to num classes
        return output


class SpeechModel(nn.Module):

    def __init__(self,
                 input_dims: int,
                 in_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 num_layers: int = 16,
                 n_heads: int = 4,
                 encoder_dim: int = 144,
                 decoder_dim: int = 144,
                 subsample_stride: int = 2,
                 normal_stride: int = 1,
                 ):
        super().__init__()
        """ 
        :param encoder_dim: encoder dimension can be used for out_channels output, model output
        :param in_channels: n_mels channels (default: 81)
        """

        # usually audio have only 1 channel -> in_channel : 1
        self.conv_subsampling = ConvSubSampling(in_channels=in_channels,
                                                out_channels=encoder_dim,
                                                kernel_size=kernel_size,
                                                stride=subsample_stride,
                                                padding=padding)  # config

        # from conv to linear the feature must be flatten
        """ linear """
        # in_feats must be out_channels of CNN, 16 as considered out channels
        self.linear = nn.Linear(in_features=encoder_dim*((input_dims // 4) - 1),
                                out_features=encoder_dim,
                                bias=True,
                                dtype=torch.float32)

        """ dropout """
        self.dropout = nn.Dropout(p=dropout)

        """ conformer encoder with layers """
        self.conformer_encoder_layers = nn.ModuleList([
            ConformerBlock(in_feats=encoder_dim,
                           out_feats=encoder_dim,
                           encoder_dim=encoder_dim,
                           attention_heads=n_heads,
                           conv_model_stride=normal_stride
                           ) for _ in range(num_layers)])  #

        """ decoder """
        self.decoder = DecoderLSTM(input_size=decoder_dim,
                                   hidden_size=decoder_dim,
                                   num_classes=num_classes,
                                   bias=True, bidirectional=True,
                                   batch_first=True)  #

        """ softmax """
        self.softmax = nn.Softmax(dim=1)  # softmax on n_frames

        """ log softmax """
        # self.log_softmax = nn.LogSoftmax()

        # encoder chain -> linear -> dropout -> conformer encoder blocks
        self.input_projection = nn.Sequential(self.linear, self.dropout)

    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        # pipeline -> conv_subsampling -> flatten -> linear -> dropout -> conformer encoder
        x = self.conv_subsampling(x)
        print(f"Conv subsampling: {x.shape}")

        output = self.input_projection(x)
        print(f"Input projection: {output.shape}")

        # output for each conformer block
        for layer in self.conformer_encoder_layers:
            output = layer(output)

        print(f"Conformer block: {output.shape}")

        return output

    def forward(self, x):
        # forward encoder
        hidden_state = self._forward_encoder(x)  # get relation ship between audio frame
        # forward decoder
        out= self.decoder(hidden_state)
        return out  # normalize output to probability with softmax


if __name__ == "__main__":
    x = torch.randn(16, 81, 300)
    in_channels = 81
    encoder_dim = 512

    # model
    speech_model = SpeechModel(in_channels=in_channels,
                               encoder_dim=encoder_dim,
                               kernel_size=3, padding=0,
                               stride=1, num_layers=4)

    print(f"Model out: {speech_model(x).shape}")
