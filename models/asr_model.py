from utils.utils import get_configs
import nemo.collections.asr as nemo_asr
from transformers import AutoModel
import torch.nn as nn
import torch
from torch import Tensor
from pytorch_lightning import LightningModule
from typing import List


class ASRModel(LightningModule):

    # the model will be built based on LightNing Module
    # the encoder will utilize the pre-trained model (which can be fine tuned on VN dataset)
    # use nemo is an arg that considered use nemo toolkit or transformer to get pretrained model
    # freeze encoder will be utilized
    def __init__(
        self,
        pretrained_name,
        d_model,
        num_classes,
        use_nemo: bool,
        is_freeze_encoder: bool,
    ):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes, bias=True)
        self.pencoder = self.get_pretrained_encoder(pretrained_name)
        self.use_nemo = use_nemo
        self.is_freeze_encoder = is_freeze_encoder

        self.chain = nn.Sequential(self.pencoder, self.linear)

    def get_pretrained_encoder(self, model_name):

        # considering use nemo toolkit or transformer
        if self.use_nemo:
            asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name=model_name
            )
        else:
            asr_model = AutoModel.from_pretrained(model_name)

        return asr_model.encoder

    def forward(self, x: Tensor, lengths: List):
        out = self.encoder(x)
        return out


def main():
    params = get_configs("../configs/asr_model_ctc_bpe.yaml")

    pretrained_model_name = "nvidia/stt_en_conformer_ctc_large"

    asr_model = ASRModel(
        pretrained_name=pretrained_model_name,
        d_model=512,
        num_classes=128,
        use_nemo=True,
        is_freeze_encoder=True,
    )

    print(f"Custom model: {asr_model}")


if __name__ == "__main__":
    main()
