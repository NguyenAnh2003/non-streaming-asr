from torch.utils.data import DataLoader, Dataset, default_collate
from feats_extraction.log_mel import audio_transforms
from utils.utils import get_configs
import torchaudio
from logger.my_logger import setup_logger
import pandas as pd
import os

logger = setup_logger(path="../logger/logs/dataset.log", location="dataloader")
logger.getLogger(__name__)

# custom data_manipulation set
class TrainSet(Dataset):

    def __init__(self, csv_file, root_dir: str = "./", config_path: str = "../configs/audio_extraction.yaml"):
        """ define init """
        self.params = get_configs(config_path)  # defined params
        self.audio_samples = pd.read_csv(csv_file) # dataset defined as csv file
        self.root_dir = root_dir # ./

    def __getitem__(self, index):
        logger.log(level=logger.DEBUG, msg="return log mel and transcript of each data_manipulation point")
        """ return log mel spectrogram, and transcript """

        # load audio to array and sample
        sample_path = self._get_audio_sample_path(index)
        audio_transcript = self.audio_samples.iloc[index, 1]
        array, rate = torchaudio.load(sample_path)

        # transform audio to mel spec
        log_mel = audio_transforms(array=array, params=self.params)

        # return log_mel and transcript
        return log_mel, str(audio_transcript)

    def _get_audio_sample_path(self, index):
        """ process audio path
        :param index -> audio sample
        :return path with audio sample .flac
        """
        sample_path = os.path.join(self.root_dir, self.audio_samples.iloc[index, 0])  # audio path for each sample index
        return f"{sample_path}.flac"

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


""" define data_manipulation loader """


# custom dataloader
class TrainLoader(DataLoader):
    def __init__(self):
        """ Train loader init """


class DevLoader(DataLoader):
    def __init__(self):
        """ Dev loader init """


""" check """
if __name__ == "__main__":
    train_set = TrainSet(csv_file="./train_samples.csv", root_dir="./librispeech/train-custom-clean")
    data_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False, collate_fn=default_collate)
    for step, (log_mel, transcript) in enumerate(data_loader):
        print(f"Audio log mel: {log_mel.shape} Transcript: {type(transcript)}")