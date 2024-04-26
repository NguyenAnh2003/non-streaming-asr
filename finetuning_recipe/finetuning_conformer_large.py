import pytorch_lightning as pl
from omegaconf import DictConfig
import nemo.collections.asr as nemo_asr
from utils.utils import get_configs

def main(MODEL_NAME: str, params):
  # get pretrained model
  conformer_large = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=MODEL_NAME)

  # optimizer

  # dataloader
  conformer_large.setup_training_data(train_data_config=params['model']['train_ds'])
  conformer_large.setup_validation_data(val_data_config=params['model']['validation_ds'])

  trainer = pl.Trainer(accelerator="gpu", max_epochs=50)
  trainer.fit(conformer_large)



if __name__ == "__main__":
  path = "../data_manipulation/librispeech/train-custom-clean"
  params = get_configs("../configs/conformer_ctc_bpe.yaml")
  MODEL_LARGE = "nvidia/stt_en_conformer_ctc_large"

  # dataloader
  params['model']['train_ds']['manifest_filepath'] = "../data_manipulation/"
  params['model']['validation_ds']['manifest_filepath'] = "../data_manipulation/"
  params['model']['test_ds']['manifest_filepath'] = "../data_manipulation/"

  main(MODEL_NAME=MODEL_LARGE, params=params)

  print(params)



