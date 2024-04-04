import wandb
import os
from dotenv import load_dotenv

# load env var
load_dotenv()

def melspec_wandb(audio_array, rate: int):
  wandb.Audio(data_or_path=audio_array,
              sample_rate=rate)

def train_logging(dev_loss: float, 
                  train_loss: float, 
                  dev_acc: float, 
                  train_acc: float,
                  epoch: int):
    
  # logging dev/train loss
  wandb.log({"train/loss_per_epoch": train_loss, 
             "train/acc_per_epoch": train_acc,
             "train/epoch": epoch})
  
  # logging dev/train accuracy
  wandb.log({"dev/loss_per_epoch": dev_loss, 
             "dev/acc_per_epoch": dev_acc,
             "dev/epoch": epoch})