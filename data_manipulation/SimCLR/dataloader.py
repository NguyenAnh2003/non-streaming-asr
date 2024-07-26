from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf, DictConfig
from typing import Any
import csv
import pandas as pd


class SimCLRDataset(Dataset):
    def __init__(self, conf: DictConfig = None, transforms=None) -> None:
        super().__init__()
        self.conf = OmegaConf.create(conf)
        self.transforms = transforms
        self.dataset = self._create_hf_dataset()

    def __getitem__(self, index):
        # input contain audio path
        # this transforms function will process with audio array
        # then transforms to log melspec and aug this input
        log_mel = self.transforms(index)
        return log_mel

    def _create_hf_dataset(self):
        # data path included in conf (OmegaConf)
        dataset = HFDataset.from_json(self.conf.model.train.train_path)
        return dataset

    def __len__(self):
        return


class SimCLRDataloader(DataLoader):
    def __init__(self):
        super().__init__()
