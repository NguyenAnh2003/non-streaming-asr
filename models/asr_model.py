from utils.utils import get_configs
import nemo.collections.asr as nemo_asr
import torch.nn as nn
import torch
from torch import Tensor
from typing import List

def get_pretrained_model(model_name):
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)
    return asr_model


class ASRModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(100, 300, bias=True)

        self.chain = nn.Sequential(self.encoder, self.linear)

    def forward(self, x: Tensor, lengths: List):
        out = self.encoder(x)
        return out
def main():
    params = get_configs("../configs/asr_model_ctc_bpe.yaml")

    pretrained_model_name = "nvidia/stt_en_conformer_ctc_large"
    pretrained_model = get_pretrained_model(pretrained_model_name)
    p_encoder = pretrained_model.encoder # init pretrained encoder
    asr_model = ASRModel(p_encoder)
    print(f"Custom model: {asr_model}")


if __name__ == "__main__":
    main()