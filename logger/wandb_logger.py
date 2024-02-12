import wandb
import os
from dotenv import load_dotenv

# load env var
load_dotenv()

# login wandb
wandb.login(key=os.getenv("WANDB_KEY"))