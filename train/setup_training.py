import torch
from torch.utils.data import DataLoader

def train_one_epoch(train_loader: DataLoader):
    train_losses = []
    """ setup train data_manipulation loader for 1 epoch """
    for step, (log_mel, transcript) in enumerate(train_loader):
        # batch_log_mel, batch transcript
        # get input from batch

        pass
    pass

def eval_one_epoch(val_loader: DataLoader):
    """ setup validation data_manipulation loader for 1 epoch """
    for step, batch in enumerate(val_loader):
        # get inputs from batch
        pass
    pass