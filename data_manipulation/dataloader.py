from torch.utils.data import DataLoader, Dataset
from feats_extraction.log_mel import audio_transforms
from utils.utils import get_configs
import torchaudio
from logger.my_logger import setup_logger
import pandas as pd

logger = setup_logger(path="../logger/logs/dataset.log", location="dataloader")
logger.getLogger(__name__)

# custom data_manipulation set
class TrainSet(Dataset):

    def __init__(self, csv_file, config_path: str = "../configs/audio_extraction.yaml"):
        """ define init """
        self.params = get_configs(config_path)  # defined params
        self.train_csv = pd.read_csv(csv_file)

    def __getitem__(self, index):
        logger.log(level=logger.DEBUG, msg="-- return log mel and transcript of each data_manipulation point --")
        """ return log mel spectrogram, and transcript """

        # load audio to array and sample
        sample_path = ""  # audio path for each sample index
        array, rate = torchaudio.load(sample_path)

        # transform audio to mel spec
        log_mel = audio_transforms(array=array, params=self.params)

        # return log_mel and transcript
        return log_mel

    def _get_audio_sample_path(self, index):
        """ process audio path """
        return


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
    pass
