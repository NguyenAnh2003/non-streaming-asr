from torch.utils.data import DataLoader, Dataset, default_collate
from feats_extraction.log_mel import audio_transforms
from utils.utils import get_configs
import torchaudio
from logger.my_logger import setup_logger
import pandas as pd
import os
from typing import Tuple

logger = setup_logger(path="../logger/logs/dataset.log", location="dataloader")
logger.getLogger(__name__)

# custom data_manipulation set
class TrainSet(Dataset):

    def __init__(self, csv_file, root_dir: str = "./", config_path: str = "../configs/audio_extraction.yaml"):
        super(TrainSet, self).__init__()
        """ define init """
        self.params = get_configs(config_path)  # defined params
        self.audio_samples = pd.read_csv(csv_file) # dataset defined as csv file
        self.root_dir = root_dir # ./

    def __getitem__(self, index):
        """ return log mel spectrogram, and transcript """

        # load audio to array and sample
        sample_path, sample_transcript = self._get_audio_sample(index)
        array, rate = torchaudio.load(sample_path)

        # transform audio to mel spec
        log_mel = audio_transforms(array=array, params=self.params)

        # return log_mel and transcript
        return log_mel, sample_transcript

    def _get_audio_sample(self, index) -> Tuple[str, str]:
        """ process audio path
        :param index -> audio sample
        :return path with audio sample .flac
        """
        sample_path = os.path.join(self.root_dir, self.audio_samples.iloc[index, 0])  # audio path for each sample index
        audio_absolute_path = f"{sample_path}.flac" # process result
        audio_transcript = self.audio_samples.iloc[index, 1]
        return audio_absolute_path, audio_transcript

    def __len__(self) -> int:
        return len(self.audio_samples)


class DevSet(Dataset):
    def __init__(self):
        """ define init """

    def __getitem__(self, item):
        """ return log mel spectrogram, and transcript """

        # load audio to array and sample

        # transform audio to mel spec

        return


# custom dataloader
class TrainLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """ Train loader init """
        super(TrainLoader, self).__init__(*args, **kwargs)
        self.shuffle = kwargs['shuffle']
        self.collate_fn = self.collate_custom_fn


    def collate_custom_fn(self, batch):
        for step, (audio_path, audio_transcript) in enumerate(batch):
            return audio_path, audio_transcript


class DevLoader(DataLoader):
    def __init__(self, dataset):
        """ Dev loader init """
        super().__init__(dataset)
        self.dataset = dataset # validation dataset

# check
if __name__ == "__main__":
    train_set = TrainSet(csv_file="./train_samples.csv", root_dir="./librispeech/train-custom-clean")
    data_loader = TrainLoader(dataset=train_set, batch_size=4, shuffle=False, collate_fn=default_collate)
    for step, (log_mel, transcript) in enumerate(data_loader):
        print(f"Audio: {log_mel} Transcript: {transcript}")