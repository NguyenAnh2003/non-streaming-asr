from utils.utils import get_configs
import nemo.collections.asr as nemo_asr
import torch.nn as nn
import torch
from torch import Tensor
from typing import List
from functools import cache



class ASRModel(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.linear = nn.Linear(100, 300, bias=True)
        self.encoder = self.get_pretrained_encoder(pretrained_name)

        self.chain = nn.Sequential(self.encoder, self.linear)

    @cache
    def get_pretrained_encoder(model_name):
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)
        return (asr_model, asr_model.encoder)

    def forward(self, x: Tensor, lengths: List):
        out = self.encoder(x)
        return out

def main():
    params = get_configs("../configs/asr_model_ctc_bpe.yaml")

    pretrained_model_name = "nvidia/stt_en_conformer_ctc_large"
    asr_model = ASRModel(pretrained_name=pretrained_model_name)
    print(f"Custom model: {asr_model}")


if __name__ == "__main__":
    main()