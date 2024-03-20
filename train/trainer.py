from utils.utils import get_configs
from data_manipulation.dataloader import DevSet, TrainSet, TrainLoader, DevLoader, LibriSpeechVocabRAW
from torch.optim import Adam
from conformer.model import Conformer
from train_utils import train_one_epoch, eval_one_epoch
from utils.utils import get_configs
from jiwer import wer
import torch.nn as nn
import time
from logger.my_logger import setup_logger

# train logger
_train_logger = setup_logger(path="../logs/train.log", location="trainer")


# model conformer
model = Conformer()

# params
_train_params = get_configs("../configs/train_params.yaml")
_model_params = get_configs("../configs/model_params.yaml")

# optimizer Adam
optimizer = Adam(model.parameters)

# loss function - ctc-loss 
criterion = nn.CTCLoss(blank=28) #

# necess params
_EPOCHS = _train_params['epochs']
BATCH_SIZE = _train_params['batch_size']
SHUFFLE = _train_params['shuffle']

# init dataloader
libri_vocab = LibriSpeechVocabRAW() # librispeech vocab

train_dataset = TrainSet(vocab=libri_vocab, csv_file="../metadata-train-clean.csv", root_dir="../librispeech/train-custom-clean")
train_dataloader = TrainLoader(dataset=train_dataset, bactch_size=BATCH_SIZE, shuffle=SHUFFLE)

dev_dataset = DevSet(vocab=libri_vocab, csv_file="../metadata-train-clean.csv", root_dir="../librispeech/train-custom-clean")
dev_dataloader = DevLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

def trainer():
  train_losses = []
  val_losses = []
  start_time = time.time()
  for epoch in range(_EPOCHS):
    # train mode
    model.train(True)

    # average loss
    train_avg_loss = train_one_epoch(train_loader=train_dataloader, model=model,
                               optimizer=optimizer, loss_fn=criterion)

    # append avg loss
    train_losses.append(train_avg_loss)

    # model validation
    model.eval(True)
    val_avg_loss = eval_one_epoch(val_loader=dev_dataloader, model=model, loss_fn=criterion)

    val_losses.append(val_avg_loss)
    
    # logger
    _train_logger.log(_train_logger.INFO, f"EPOCH: {epoch+1} TRAIN LOSS: {train_avg_loss} 
                      DEV LOSS: {val_avg_loss}")

  # logging summary
  _train_logger.log(_train_logger.INFO, f"EPOCHES: {_EPOCHS} TOTAL TRAIN LOSS: {min(train_losses)} 
                    TOTAL DEV LOSS: {min(val_losses)}")

if __name__ == "__main__":
  trainer()