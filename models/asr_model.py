from utils.utils import get_configs
import nemo.collections.asr as nemo_asr
import torch.nn as nn
import torch
from torch import Tensor
from typing import List
from functools import cache, lru_cache


class ASRModel(nn.Module):
    def __init__(self, pretrained_name, d_model, num_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes, bias=True)
        _, self.pencoder = self.get_pretrained_encoder(pretrained_name)
        self.chain = nn.Sequential(self.pencoder, self.linear)

    @lru_cache(maxsize=1)
    def get_pretrained_encoder(model_name):
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)
        return (asr_model, asr_model.encoder)

    def forward(self, x: Tensor, lengths: List):
        out = self.encoder(x)
        return out

def main():
    params = get_configs("../configs/asr_model_ctc_bpe.yaml")

    pretrained_model_name = "nvidia/stt_en_conformer_ctc_large"
    
    asr_model = ASRModel(pretrained_name=pretrained_model_name, 
                         d_model=512, 
                         num_classes=128)

    print(f"Custom model: {asr_model}")


if __name__ == "__main__":
    main()