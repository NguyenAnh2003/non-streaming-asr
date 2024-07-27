from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf, DictConfig
from typing import Any
import csv
import pandas as pd
import os


class SimCLRDataset(Dataset):
    def __init__(self, conf: DictConfig = None, transforms=None) -> None:
        super().__init__()
        # conf OmegaConf create
        # transforms function for augment audio signal
        self.conf = OmegaConf.create(conf)
        self.transforms = transforms
        self.dataset = self._create_hf_dataset()
        self.root_dir = self.conf.trainer.root_dir  # setup root dir

    def __getitem__(self, index):
        # input contain audio path
        # this transforms function will process with audio array
        # then transforms to melspec and aug this input
        audio_path = self.dataset[index]["audio_filepath"]
        audio_path = os.path.join(self.root_dir, audio_path)
        transcript = self.dataset[index]["text"]

        if self.transforms:
            augmented_data = self.transforms(audio_path)  # input (audio path)

        return augmented_data, transcript

    def _create_hf_dataset(self):
        # data path included in conf (OmegaConf)
        # read json file
        dataset = HFDataset.from_json(self.conf.trainer.train_path)
        return dataset


class SimCLRDataloader(DataLoader):
    def __init__(self):
        super().__init__()
