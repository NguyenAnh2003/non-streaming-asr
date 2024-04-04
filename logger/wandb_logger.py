import wandb
import os
from dotenv import load_dotenv

# load env var
load_dotenv()

def train_logging(dev_loss: float, 
                  train_loss: float, 
                  dev_acc: float, 
                  train_acc: float):
    
  # logging dev/train loss
  wandb.log({"train/loss_per_epoch": train_loss, "train/acc_per_epoch": train_acc})
  
  # logging dev/train accuracy
  wandb.log({"dev/loss_per_epoch": dev_loss, "dev/acc_per_epoch": dev_acc})