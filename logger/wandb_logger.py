import wandb
import os
from dotenv import load_dotenv

# load env var
load_dotenv()

def train_logging(model_name: str, 
                  dev_loss: float, 
                  train_loss: float, 
                  dev_acc: float, 
                  train_acc: float):
    
  # logging dev/train loss
  wandb.log({"train loss/epoch": train_loss, "dev loss/epoch": dev_loss})
  
  # logging dev/train accuracy
  wandb.log({"train acc/epoch": train_acc, "dev acc/epoch": dev_acc})