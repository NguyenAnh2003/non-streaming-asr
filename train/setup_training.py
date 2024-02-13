import torch
from torch.utils.data import DataLoader


def train_one_epoch(train_loader: DataLoader):
    """ setup train data loader for 1 epoch """
    for step, batch in enumerate(train_loader):
        # get input from batch
        pass
    pass

def eval_one_epoch(val_loader: DataLoader):
    """ setup validation data loader for 1 epoch """
    for step, batch in enumerate(val_loader):
        # get inputs from batch
        pass
    pass