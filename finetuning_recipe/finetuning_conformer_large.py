import pytorch_lightning as pl
from omegaconf import DictConfig
import nemo.collections.asr as nemo_asr
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import get_configs

def main(MODEL_NAME: str, params):
  print("Loading pretrained model")
  # get pretrained model
  conformer_large = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=MODEL_NAME)

  # optimizer

  print("Setup dataset")
  # dataloader
  conformer_large.setup_training_data(train_data_config=params['model']['train_ds'])
  conformer_large.setup_validation_data(val_data_config=params['model']['validation_ds'])

  print("Tensorboard...")
  # logger setup
  logger = TensorBoardLogger(save_dir="../logger/logs", version=1, name=MODEL_NAME)

  print("Prepare trainer")
  trainer = pl.Trainer(accelerator="gpu", max_epochs=50, 
                       logger=logger, log_every_n_steps=100,
                       enable_checkpointing=True, 
                       inference_mode=False)
  print("Training....")
  trainer.fit(conformer_large)

  # trainer.validate(model=conformer_large,)
  
  # save model
  conformer_large.save_to(f"../saved_model/{MODEL_NAME}")
  print("Saved model ... DONE")



if __name__ == "__main__":
  SAMPLE_RATE = 16000
  path = "../data_manipulation/librispeech/train-custom-clean"
  params = get_configs("../configs/conformer_ctc_bpe.yaml")
  MODEL_LARGE = "stt_en_conformer_ctc_large_ls"

  # dataloader
  params['model']['train_ds']['sample_rate'] = SAMPLE_RATE
  params['model']['validation_ds']['sample_rate'] = SAMPLE_RATE
  params['model']['train_ds']['manifest_filepath'] = "../data_manipulation/train-manifest.json"
  params['model']['validation_ds']['manifest_filepath'] = "../data_manipulation/dev-manifest.json"
  # params['model']['test_ds']['manifest_filepath'] = "../data_manipulation/test-manifest.json"

  main(MODEL_NAME=MODEL_LARGE, params=params)


