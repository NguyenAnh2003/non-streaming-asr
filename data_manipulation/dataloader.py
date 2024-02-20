import torch
from torch.utils.data import DataLoader, Dataset # torch Dataset
from datasets import Dataset as HuggingFaceDataset # huggingface Dataset
from feats_extraction.log_mel import audio_transforms
from utils.utils import get_configs
import torchaudio
from logger.my_logger import setup_logger
import pandas as pd
import os
from typing import Tuple, Dict

from torchtext.vocab import Vocab

logger = setup_logger(path="../logger/logs/dataset.log", location="dataloader")
logger.getLogger(__name__)


def _create_huggingface_dataset(csv_path: str):
    train_csv = pd.read_csv(csv_path)
    dataset = HuggingFaceDataset.from_pandas(train_csv)
    return dataset

# vocab
class LibriSpeechVocabRAW:
    # language librispeech vocab file: https://openslr.trmal.net/resources/11/librispeech-vocab.txt

    def __init__(self, vocab_file_path: str = "./vocab.txt"):
        # vocab file
        self.vocab_file = vocab_file_path
        self.word2index = {}
        self.index2word = {}
        self.index_of_word = 1 # default index for a word
        self._process_vocab()

    def _process_vocab(self):
        with open(self.vocab_file, 'r', encoding='utf-8') as vb_file:
            for line in vb_file:
                # assign word to index and index to word (line.replace("\n", "") represent for a line -> 1 word 1 line)
                self.word2index[line.replace("\n", "")] = self.index_of_word
                self.index2word[self.index_of_word] = line.replace("\n", "")
                self.index_of_word += 1 # increase index

# custom data_manipulation set
class TrainSet(Dataset):

    def __init__(self, vocab, csv_file, root_dir: str = "./", config_path: str = "../configs/audio_extraction.yaml"):
        super(TrainSet, self).__init__()
        """ define init """
        self.params = get_configs(config_path)  # defined params
        self.audio_samples = pd.read_csv(csv_file)  # dataset defined as csv file
        self.root_dir = root_dir  # ./
        self.vocab = vocab
        self.hg_dataset = _create_huggingface_dataset(csv_path=csv_file)

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
        sample_path = os.path.join(self.root_dir, self.hg_dataset[index]['audio_id'])  # audio path for each sample index
        audio_absolute_path = f"{sample_path}.flac"  # process result
        audio_transcript = self.hg_dataset[index]['transcript']
        return audio_absolute_path, audio_transcript

    def __len__(self) -> int:
        return len(self.audio_samples)


class DevSet(Dataset):
    def __init__(self, csv_file, root_dir: str = "./", config_path: str = "../configs/audio_extraction.yaml"):
        super(DevSet, self).__init__()
        """ define init """
        self.params = get_configs(config_path)  # defined params
        self.audio_samples = pd.read_csv(csv_file)  # dataset defined as csv file
        self.root_dir = root_dir  # ./

    def __getitem__(self, index):
        """ return log mel spectrogram, and transcript """

        # load audio to array and sample
        sample_path, sample_transcript = self.__get_audio_sample(index)
        array, rate = torchaudio.load(sample_path)

        # transform audio to mel spec
        log_mel = audio_transforms(array=array, params=self.params)

        # return log_mel and transcript
        return log_mel, sample_transcript

    def __get_audio_sample(self, index) -> Tuple[str, str]:
        """ process audio path
        :param index -> audio sample
        :return path with audio sample .flac
        """
        sample_path = os.path.join(self.root_dir, self.audio_samples.iloc[index, 0])  # audio path for each sample index
        audio_absolute_path = f"{sample_path}.flac"  # process result
        audio_transcript = self.audio_samples.iloc[index, 1]
        return audio_absolute_path, audio_transcript

    def _process_sample_transcript(self, index):
        """
        :param index: index for each sample
        the function used for process audio transcript from str to int
        list[index]
        """

    def __len__(self) -> int:
        return len(self.audio_samples)


# custom dataloader
class TrainLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """ Train loader init """
        super(TrainLoader, self).__init__(*args, **kwargs)
        self.shuffle = kwargs['shuffle']
        # self.collate_fn = self.collate_custom_fn

    def collate_custom_fn(self, batch):
        # https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
        pass
        # for step, (audio_path, audio_transcript) in enumerate(batch):
        # process each sample in 1 batch
        # return audio_path, audio_transcript


class DevLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """ Dev loader init """
        super().__init__(*args, **kwargs)


# check
if __name__ == "__main__":
    librispeech_vocab = LibriSpeechVocabRAW()
    train_set = TrainSet(vocab= librispeech_vocab, csv_file="./train_samples.csv", root_dir="./librispeech/train-custom-clean")
    data_loader = TrainLoader(dataset=train_set, batch_size=1, shuffle=False)
    for step, (log_mel, transcript) in enumerate(data_loader):
        print(f"Audio: {log_mel} Transcript: {transcript}")
