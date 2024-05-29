import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr
from pytorch_lightning.loggers import TensorBoardLogger
from nemo.collections.asr.metrics.wer import WER
from utils.utils import get_configs

def main(MODEL_NAME: str, params):
  print("Loading pretrained model")
  # get pretrained model
  asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=MODEL_NAME)
  asr_model.encoder.freeze() # freezing encoder of ASR model

  # optimizer

  print("Setup dataset")
  # dataloader
  asr_model.setup_training_data(train_data_config=params['model']['train_ds'])
  asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])

  print("Tensorboard...")
  # logger setup
  logger = TensorBoardLogger(save_dir="../logger/logs", version=1, name=MODEL_NAME)

  print("Prepare trainer")
  trainer = pl.Trainer(accelerator="gpu", max_epochs=50, 
                       logger=logger, log_every_n_steps=100,
                       enable_checkpointing=True, 
                       inference_mode=False)
  print("Training....")
  # trainer.fit(asr_model)

  # trainer.validate(model=asr_model,)
  
  # save model
  # asr_model.save_to(f"../saved_model/{MODEL_NAME}")
  print("Saved model ... DONE")

def test(MODEL_NAME, params):
  # prepare model
  asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    restore_path=f"../saved_model/{MODEL_NAME}")

  print(f"Prepare testing model: {MODEL_NAME}")
  asr_model.setup_test_data(test_data_config=params['model']['test_ds'])
  asr_model.cuda()
  asr_model.eval()
  
  wer_nums = []
  wer_denoms = [] # label tokens
  
  for test_batch in asr_model.test_dataloader():
    test_batch = [x.cuda() for x in test_batch]
    targets = test_batch[2] #
    targets_size = test_batch[3] #
    in_size = test_batch[1] #

    print(f"Targets length: {targets_size} "
          f" In size: {in_size}")

    log_probs, encoded_len, greedy_predictions = asr_model(
      input_signal=test_batch[0], input_signal_length=test_batch[1]
    )
    # Notice the model has a helper object to compute WER
    asr_model.wer.update(predictions=greedy_predictions, predictions_lengths=None, targets=targets,
                               targets_lengths=targets_lengths)
    _, wer_num, wer_denom = asr_model.wer.compute()
    asr_model.wer.reset()
    wer_nums.append(wer_num.detach().cpu().numpy())
    wer_denoms.append(wer_denom.detach().cpu().numpy())

    # Release tensors from GPU memory
    del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions

  # We need to sum all numerators and denominators first. Then divide.
  print(f"WER = {sum(wer_nums) / sum(wer_denoms)}")

def inference(array, MODEL_NAME):
  asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    restore_path=f"../saved_model/{MODEL_NAME}")
  asr_model.cuda()
  result = asr_model.transcribe([array])
  print(result)

if __name__ == "__main__":
  SAMPLE_RATE = 16000
  path = "../data_manipulation/librispeech/augmented-train"
  params = get_configs("../configs/conformer_ctc_bpe.yaml")
  MODEL_LARGE = "nvidia/stt_en_conformer_ctc_large"
  SAVED_MODEL = "stt_en_conformer_ctc_large_customs_ls.nemo"
  FCONFORMER_LARGE = "nvidia/stt_en_fastconformer_ctc_large"

  # dataloader
  params['model']['train_ds']['sample_rate'] = SAMPLE_RATE
  params['model']['validation_ds']['sample_rate'] = SAMPLE_RATE
  params['model']['test_ds']['sample_rate'] = SAMPLE_RATE
  params['model']['train_ds']['manifest_filepath'] = "../data_manipulation/metadata/manifests/train-clean-manifest.json"
  params['model']['validation_ds']['manifest_filepath'] = "../data_manipulation/metadata/manifests/dev-clean-manifest.json"
  params['model']['test_ds']['manifest_filepath'] = "../data_manipulation/metadata/manifests/test-aug-manifest.json"

  # main(MODEL_NAME=MODEL_LARGE, params=params)
  test(SAVED_MODEL, params)
  # inference("../data_manipulation/examples/kkk.flac", SAVED_MODEL)