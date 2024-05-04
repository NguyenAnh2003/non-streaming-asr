import torch
import torch.nn as nn
from .convolution import ConvSubSampling
from .conformer_block import ConformerBlock
from torchaudio.models import Conformer

class DecoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 bias: bool, bidirectional: bool = True,
                 batch_first: bool = True,
                 d_model: int = 144,
                 num_classes: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.output_projection = nn.Linear(in_features=d_model,
                                           out_features=num_classes,
                                           bias=bias)

    def forward(self, x):
        out = self.dropout(x)
        output = self.output_projection(out) # projection the output to num classes
        return output


class SpeechModel(nn.Module):

    def __init__(self,
                 input_dims: int,
                 in_channels: int,
                 sub_kernel_size: int,
                 pw_kernel_size: int,
                 dw_kernel_size: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 padding: int = 1,
                 num_layers: int = 16,
                 n_heads: int = 4,
                 encoder_dim: int = 144,
                 decoder_dim: int = 144,
                 subsample_stride: int = 2,
                 normal_stride: int = 1,
                 expansion_factor: int = 4,
                 apply_conv_first: bool = True,
                 ):
        super().__init__()
        """ 
        :param encoder_dim: encoder dimension can be used for out_channels output, model output
        :param in_channels: n_mels channels (default: 81)
        """

        # usually audio have only 1 channel -> in_channel : 1
        self.conv_subsampling = ConvSubSampling(in_channels=in_channels,
                                                out_channels=encoder_dim,
                                                kernel_size=sub_kernel_size,
                                                stride=subsample_stride,
                                                padding=padding)  # config

        # from conv to linear the feature must be flatten
        """ linear """
        # in_feats must be out_channels of CNN, 16 as considered out channels
        self.linear = nn.Linear(
            # in_features=encoder_dim*input_dims,
            in_features=(input_dims  * encoder_dim),
            out_features=encoder_dim,
            bias=True,
            dtype=torch.float32)

        # encoder chain -> linear -> dropout -> conformer encoder blocks
        self.input_projection = nn.Sequential(self.linear, nn.Dropout(p=0.1))



        """ dropout """
        self.dropout = nn.Dropout(p=dropout)

        """ conformer encoder with layers """
        self.conformer_encoder_layers = nn.ModuleList([
            ConformerBlock(encoder_dim=encoder_dim,
                           attention_heads=n_heads,
                           pw_ksize = pw_kernel_size,
                           dw_ksize = dw_kernel_size,
                           conv_model_stride=normal_stride,
                           expansion_factor=expansion_factor,
                           apply_conv_first=apply_conv_first
                           ) for _ in range(num_layers)])  #

        self.conformer_torchaudio = Conformer(input_dim=144, num_heads=4,
                                              ffn_dim=144*4, num_layers=16,
                                              depthwise_conv_kernel_size=31,
                                              convolution_first=True)

        """ decoder """
        # self.decoder = DecoderLSTM(input_size=decoder_dim,
        #                            hidden_size=decoder_dim,
        #                            num_classes=num_classes,
        #                            bias=True, bidirectional=True,
        #                            batch_first=True)  #

        self.last_linear = nn.Linear(in_features=encoder_dim, out_features=num_classes)

        self.classifier = nn.Sequential(self.last_linear, self.dropout)

        """ log softmax """
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def calc_length(self, lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
        add_pad: float = all_paddings - kernel_size
        one: float = 1.0
        for i in range(repeat_num):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
            if ceil_mode:
                lengths = torch.ceil(lengths)
            else:
                lengths = torch.floor(lengths)
        return lengths.to(dtype=torch.int)

    def _forward_encoder(self, x: torch.Tensor, lengths: torch.Tensor):
        # calculate lengths
        # out_lengths = self.calc_length(lengths, all_paddings=2, kernel_size=3,
        #                            stride=2, ceil_mode=False, repeat_num=2)

        # pipeline -> conv_subsampling -> flatten -> linear -> dropout -> conformer encoder
        x = self.conv_subsampling(x)

        output = self.input_projection(x)

        # output, lengths = self.conformer_torchaudio(output, lengths)

        # output for each conformer block
        for layer in self.conformer_encoder_layers:
            output = layer(output)
            
        return output, lengths

    def forward(self, x, lengths):
        # forward encoder
        hidden_state, lengths = self._forward_encoder(x, lengths)  # get relation ship between audio frame
        # forward decoder
        out = self.classifier(hidden_state)
        out = out.contiguous().transpose(0, 1)

        # output (log_probs, prediction(argmax), lengths)
        log_probs = self.log_softmax(out)
        _, prediction = torch.max(log_probs, dim=-1)
        return log_probs, prediction, lengths  # normalize output to probability with softmax


if __name__ == "__main__":
    x = torch.randn(16, 81, 300)
    in_channels = 81
    encoder_dim = 512
