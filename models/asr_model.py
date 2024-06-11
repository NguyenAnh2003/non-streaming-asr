from utils.utils import get_configs
import nemo.collections.asr as nemo_asr
from transformers import AutoModel
import torch.nn as nn
import torch
from torch import Tensor
from pytorch_lightning import LightningModule
from typing import List
from torch.optim import Adam
from omegaconf import OmegaConf, DictConfig

class ASRModel(LightningModule):

    # the model will be built based on LightNing Module
    # the encoder will utilize the pre-trained model (which can be fine tuned on VN dataset)
    # use nemo is an arg that considered use nemo toolkit or transformer to get pretrained model
    # freeze encoder will be utilized
    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = OmegaConf.create(conf)
        self.pencoder = self.get_pretrained_encoder()

        self.model = nn.Sequential(self.pencoder, self.linear)

    def get_pretrained_encoder(self):
        model_name = self.conf.pretrained_model
        # considering use nemo toolkit or transformer
        if self.use_nemo:
            asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name=model_name
            )
        else:
            asr_model = AutoModel.from_pretrained(model_name)

        return asr_model.encoder

    def forward(self, x: Tensor, lengths: List, targets):
        """
        Args:
            x (Tensor): input with tensor
            lengths (List): length input
            targets (_type_): considered as labels
        """
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        #
        inputs, target = batch
        output = self(inputs, target)
        # define loss
        loss = 0
        return loss


def main():
    params = get_configs("../configs/asr_model_ctc_bpe.yaml")

    params["model"]["pretrained_model"] = "nvidia/stt_en_conformer_ctc_large"

    asr_model = ASRModel(params)

    print(f"Custom model: {asr_model}")


if __name__ == "__main__":
    main()
