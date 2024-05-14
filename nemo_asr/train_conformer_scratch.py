from omegaconf import DictConfig
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import get_configs


def main(MODEL_NAME: str, params):
  
  print("Tensorboard...")
  # logger setup
  logger = TensorBoardLogger(save_dir="../logger/logs", version=1, name=MODEL_NAME)


  print(f"Prepare trainer")
  trainer = pl.Trainer(accelerator="gpu", max_epochs=50, 
                       logger=logger, log_every_n_steps=100,
                       enable_checkpointing=True, 
                       inference_mode=False)
  
  conformer_large = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

  print("Setup dataset")
  # dataloader
  # conformer_large.setup_training_data(train_data_config=params['model']['train_ds'])
  # conformer_large.setup_validation_data(val_data_config=params['model']['validation_ds'])

  print("Training....")
  trainer.fit(conformer_large)
  
  # save model
  # conformer_large.save_to(f"../saved_model/{MODEL_NAME}")
  print("Saved model ... DONE")
  
if __name__ == "__main__":
  SAMPLE_RATE = 16000
  path = "../data_manipulation/librispeech/augmented-train"
  params = get_configs("../configs/conformer_ctc_char.yaml")
  MODEL_LARGE = "stt_en_conformer_ctc_large_ls"
  SAVED_MODEL = "stt_en_conformer_ctc_large_customs_ls.nemo"


  # dataloader
  params['model']['preprocessor']['sample_rate'] = SAMPLE_RATE
  params['model']['encoder']['feat_in'] = 80
  params['model']['encoder']['d_model'] = 512
  params['model']['optim']['sched']['d_model'] = 512
  params['model']['train_ds']['labels'] = params['model']['labels']
  params['model']['validation_ds']['labels'] = params['model']['labels']
  params['model']['test_ds']['labels'] = params['model']['labels']
  params['model']['decoder']['vocabulary'] = params['model']['labels']
  params['model']['decoder']['num_classes'] = 28
  params['model']['decoder']['feat_in'] = 512
  params['model']['optim']['weight_decay'] = 1e-3
  params['model']['optim']['sched']['min_lr'] = 1e-6
  params['model']['train_ds']['sample_rate'] = SAMPLE_RATE
  params['model']['validation_ds']['sample_rate'] = SAMPLE_RATE
  params['model']['test_ds']['sample_rate'] = SAMPLE_RATE
  params['model']['train_ds']['manifest_filepath'] = "../data_manipulation/metadata/manifests/train-aug-manifest.json"
  params['model']['validation_ds']['manifest_filepath'] = "../data_manipulation/metadata/manifests/dev-aug-manifest.json"
  params['model']['test_ds']['manifest_filepath'] = "../data_manipulation/metadata/manifests/test-aug-manifest.json"

  main(MODEL_NAME=MODEL_LARGE, params=params)