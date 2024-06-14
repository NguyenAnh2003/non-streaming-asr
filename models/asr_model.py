from parts.utils.utils import get_configs
import nemo.collections.asr as nemo_asr
from transformers import AutoModel
import torch
from torch import Tensor
from pytorch_lightning import LightningModule
from typing import List
from omegaconf import OmegaConf, DictConfig


class ASRModel(LightningModule):

    # the model will be built based on LightNing Module
    # the encoder will utilize the pre-trained model (which can be fine tuned on VN dataset)
    # use nemo is an arg that considered use nemo toolkit or transformer to get pretrained model
    # freeze encoder will be utilized

    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = OmegaConf.create(conf)

        if self.conf.model.use_pretrained == True:
            self.pretrained_encoder = self.get_pretrained_encoder()

        # self.decoder = ASRDecoder() # decoder?

    def get_pretrained_encoder(self):
        # considering use nemo toolkit or transformer
        if self.conf.model.use_nemo:
            asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name=self.conf.model.pretrained_model
            )
        else:
            asr_model = AutoModel.from_pretrained(self.conf.model.pretrained_model)

        return asr_model.encoder

    def forward(self, x: Tensor, lengths: List, targets):
        """
        Args:
            x (Tensor): input with tensor
            lengths (List): length input
            targets (_type_): considered as labels
        """
        out = self.pretrained_encoder(x)
        return out

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.pretrained_encoder(inputs, target)
        loss = 0
        return loss
    
    def configure_optimizers(self):
        return torch.nn.CTCLoss()

def main():
    params = get_configs("../configs/asr_model_with_pretrained_ctc_bpe.yaml")

    params["model"]["pretrained_model"] = "vinai/PhoWhisper-base"

    asr_model = ASRModel(params)

    print(f"Custom model: {asr_model}")


if __name__ == "__main__":
    main()
