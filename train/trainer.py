import torch.cuda
from utils.utils import get_configs
from data_manipulation.dataloader import DevSet, TrainSet, TrainLoader, DevLoader, LibriSpeechVocabRAW
from torch.optim import Adam
from train_utils import train_one_epoch, eval_one_epoch, setup_speech_model
from utils.utils import get_configs
from jiwer import wer
import torch.nn as nn
import time
from logger.my_logger import setup_logger
from utils.utils import get_executing_time

# train logger
_train_logger = setup_logger(path="../logs/train.log", location="trainer")

# params
train_params = get_configs("../configs/train_params.yaml")
model_params = get_configs("../configs/model_params.yaml")

# model params
encoder_dim = model_params['encoder_dim']
in_channels = model_params['channels']

# model conformer
model = setup_speech_model()

# necess params
EPOCHS = train_params['epochs']
BATCH_SIZE = train_params['batch_size']
LR = train_params['learning_rate']
DATASET_NAME = train_params['dataset_name']
SHUFFLE = train_params['shuffle']

# optimizer Adam
optimizer = Adam(model.parameters(), lr=LR)

# loss function - ctc-loss 
criterion = nn.CTCLoss(blank=28) #

# init dataloader
libri_vocab = LibriSpeechVocabRAW(vocab_file_path="../data_manipulation/vocab.txt") # librispeech vocab

train_dataset = TrainSet(vocab=libri_vocab, csv_file="../data_manipulation/metadata-train-clean.csv",
                         root_dir="../data_manipulation/librispeech/train-custom-clean")
train_dataloader = TrainLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

dev_dataset = DevSet(vocab=libri_vocab, csv_file="../data_manipulation/metadata-train-clean.csv",
                     root_dir="../data_manipulation/librispeech/train-custom-clean")
dev_dataloader = DevLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

def trainer(exp_name: str):
  # wandb config
  configs = dict(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LR,
    dataset=DATASET_NAME,
    architecture=exp_name
  )

  # init var
  train_losses = []
  val_losses = []
  start_time = time.time()

  # wandb init
  # wandb.init(project="S2T",
  #            name=exp_name,
  #            config=configs)
  
  # wandb watch model grads
  # wandb.watch(models=model, criterion=criterion, log="all", log_freq=10)

  # iterate epochs
  for epoch in range(EPOCHS):

    # move model to cuda
    if torch.cuda.is_available():
        model.cuda()

    # train mode
    model.train(True)

    # average loss
    # train_avg_loss = train_one_epoch(train_loader=train_dataloader,
    #                                  model=model,
    #                                  optimizer=optimizer,
    #                                  loss_fn=criterion)

    train_one_epoch(train_loader=train_dataloader,
                                     model=model,
                                     optimizer=optimizer,
                                     loss_fn=criterion)

    # append avg loss
    # train_losses.append(train_avg_loss)

    # model validation
    # model.eval()
    # val_avg_loss = eval_one_epoch(val_loader=dev_dataloader,
    #                               model=model,
    #                               loss_fn=criterion)
    #
    # val_losses.append(val_avg_loss)
    #
    # # logger
    # _train_logger.log(_train_logger.INFO, f"EPOCH: {epoch+1} TRAIN LOSS: {train_avg_loss} DEV LOSS: {val_avg_loss}")
    #
    # # console log
    # print(f"EPOCH: {epoch+1} TRAIN LOSS: {train_avg_loss} DEV LOSS: {val_avg_loss} TIME: {get_executing_time(start_time=start_time)}")
    #
    # # wandb logging
    # train_logging(model_name=exp_name,
    #               train_loss=train_avg_loss,
    #               dev_loss=val_avg_loss, epoch=epoch)

  # terminate wandb
  # wandb.finish()
  
  trained_time = get_executing_time(start_time)
  print(f"EPOCHES: {EPOCHS} TRAIN LOSS: {min(train_losses)} DEV LOSS: {min(val_losses)} Time: {trained_time}")
  # logging summary
  _train_logger.log(_train_logger.INFO, f"EPOCHES: {EPOCHS} TOTAL TRAIN LOSS: {min(train_losses)} TOTAL DEV LOSS: {min(val_losses)}")

if __name__ == "__main__":
  EXP_NAME = train_params['model_name']
  trainer(EXP_NAME)