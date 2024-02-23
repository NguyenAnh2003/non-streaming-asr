import torch
from torch.utils.data import DataLoader, Dataset # torch Dataset
from datasets import Dataset as HuggingFaceDataset # huggingface Dataset
from feats_extraction.log_mel import audio_transforms
from utils.utils import get_configs
import torchaudio
from logger.my_logger import setup_logger
import pandas as pd
import os
from typing import Tuple, List
from tqdm import tqdm
import numpy as np

from torchtext.vocab import Vocab

logger = setup_logger(path="../logger/logs/dataset.log", location="dataloader")
logger.getLogger(__name__)

_FILTER_BANKS = 81

# vocab
class LibriSpeechVocabRAW:
    # language librispeech vocab file: https://openslr.trmal.net/resources/11/librispeech-vocab.txt

    def __init__(self, vocab_file_path: str = "./vocab.txt"):
        # vocab file
        self.vocab_file = vocab_file_path
        self.word2index = {"PAD": 0, "UNF": 1} # update code
        self.index2word = {0: "PAD", 1: "UNF"} # update code
        self.index_of_word = 2 # default index for a word
        self.__process_vocab()

    def __process_vocab(self):
        with open(self.vocab_file, 'r', encoding='utf-8') as vb_file:
            for line in vb_file:
                # assign word to index and index to word (line.replace("\n", "") represent for a line -> 1 word 1 line)
                self.word2index[line.replace("\n", "")] = self.index_of_word
                self.index2word[self.index_of_word] = line.replace("\n", "")
                self.index_of_word += 1 # increase index

# dataset
class TrainSet(Dataset):

    def __init__(self, vocab, csv_file, root_dir: str = "./", config_path: str = "../configs/audio_extraction.yaml"):
        super(TrainSet, self).__init__()
        """ define init """
        self.params = get_configs(config_path)  # defined params
        self.audio_samples = pd.read_csv(csv_file)  # dataset defined as csv file
        self.root_dir = root_dir  # ./
        self.vocab = vocab
        # init hugging face dataset
        self.hg_dataset = self.__create_huggingface_dataset(csv_path=csv_file)
        # map each transcript from str to index(int) in word2index dict
        self.hg_dataset = self.hg_dataset.map(self.__process_sample_transcript)

    def __getitem__(self, index):
        """ return log mel spectrogram, and transcript """

        # load audio to array and sample
        sample_path, sample_transcript = self.__get_audio_sample(index)
        array, rate = torchaudio.load(sample_path)
        # transform audio to mel spec
        log_mel = audio_transforms(array=array, params=self.params)

        # return log_mel and transcript
        return log_mel, torch.tensor(sample_transcript)

    def __get_audio_sample(self, index) -> Tuple[str, List[int]]:
        """ process audio path
        :param index -> audio sample
        :return path with audio sample .flac
        """
        sample_path = os.path.join(self.root_dir, self.hg_dataset[index]['audio_id'])  # audio path for each sample index
        audio_absolute_path = f"{sample_path}.flac"  # process result
        audio_transcript = self.hg_dataset[index]['transcript']

        return audio_absolute_path, audio_transcript

    @staticmethod
    def __create_huggingface_dataset(csv_path: str):
        train_csv = pd.read_csv(csv_path)
        dataset = HuggingFaceDataset.from_pandas(train_csv)
        return dataset

    def __process_sample_transcript(self, batch: Dataset):
        """ function receive batch and mapp each transcript to index in Vocab """
        batch["transcript"] = batch["transcript"].split()
        batch["transcript"] = [*map(self.vocab.word2index.get, batch["transcript"])]
        batch["transcript"] = [int(1 if value is None else value) for value in batch["transcript"]] # update code
        return batch

    def __len__(self) -> int:
        return len(self.audio_samples)


class DevSet(Dataset):
    def __init__(self, vocab, csv_file, root_dir: str = "./", config_path: str = "../configs/audio_extraction.yaml"):
        super(TrainSet, self).__init__()
        """ define init """
        self.params = get_configs(config_path)  # defined params
        self.audio_samples = pd.read_csv(csv_file)  # dataset defined as csv file
        self.root_dir = root_dir  # ./
        self.vocab = vocab
        # init hugging face dataset
        self.hg_dataset = self.__create_huggingface_dataset(csv_path=csv_file)
        # map each transcript from str to index(int) in word2index dict
        self.hg_dataset = self.hg_dataset.map(self.__process_sample_transcript)

    def __getitem__(self, index):
        """ return log mel spectrogram, and transcript """

        # load audio to array and sample
        sample_path, sample_transcript = self.__get_audio_sample(index)
        array, rate = torchaudio.load(sample_path)
        # transform audio to mel spec
        log_mel = audio_transforms(array=array, params=self.params)

        # return log_mel and transcript
        return log_mel, torch.tensor(sample_transcript)

    def __get_audio_sample(self, index) -> Tuple[str, List[int]]:
        """ process audio path
        :param index -> audio sample
        :return path with audio sample .flac
        """
        sample_path = os.path.join(self.root_dir,
                                   self.hg_dataset[index]['audio_id'])  # audio path for each sample index
        audio_absolute_path = f"{sample_path}.flac"  # process result
        audio_transcript = self.hg_dataset[index]['transcript']

        return audio_absolute_path, audio_transcript

    @staticmethod
    def __create_huggingface_dataset(csv_path: str):
        train_csv = pd.read_csv(csv_path)
        dataset = HuggingFaceDataset.from_pandas(train_csv)
        return dataset

    def __process_sample_transcript(self, batch: Dataset):
        """ function receive batch and mapp each transcript to index in Vocab """
        batch["transcript"] = batch["transcript"].split()
        batch["transcript"] = [*map(self.vocab.word2index.get, batch["transcript"])]
        batch["transcript"] = [int(1 if value is None else value) for value in batch["transcript"]]  # update code
        return batch

    def __len__(self) -> int:
        return len(self.audio_samples)


# custom dataloader
class TrainLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """ Train loader init """
        super(TrainLoader, self).__init__(*args, **kwargs)
        self.shuffle = kwargs['shuffle']
        self.collate_fn = self.collate_custom_fn

    def collate_custom_fn(self, batch):
        # https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders

        batch_size = len(batch) # create temp batch_size

        # max_frames - each one: tensor([n_frames, banks])
        max_frames = max(x[0].size(0) for x in batch) # get max n_frames per batch

        # max_len_transcript - each one: len(transcript)
        max_len_transcript = max(len(x[1]) for x in batch)

        # create empty tensor with batch_size, max_frames and banks
        batch_logmel = torch.zeros(batch_size, max_frames, _FILTER_BANKS, dtype=torch.float32)

        batch_transcript = torch.zeros(batch_size, max_len_transcript, dtype=torch.int)

        for step, (log_mel, transcript) in enumerate(batch):
            # process each single sample and add to batch

            batch_logmel[step].narrow(0, 0, log_mel.size(0)).copy_(log_mel)
            batch_transcript[step].narrow(0, 0, len(transcript)).copy_(transcript)
            # return log_mel, transcript

        return batch_logmel, batch_transcript



class DevLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """ Train loader init """
        super(DevLoader, self).__init__(*args, **kwargs)
        self.shuffle = kwargs['shuffle']
        self.collate_fn = self.collate_custom_fn

    def collate_custom_fn(self, batch):
        batch_size = len(batch)  # create temp batch_size

        # max_frames - each one: tensor([n_frames, banks])
        max_frames = max(x[0].size(0) for x in batch)  # get max n_frames per batch

        # max_len_transcript - each one: len(transcript)
        max_len_transcript = max(len(x[1]) for x in batch)

        # create empty tensor with batch_size, max_frames and banks
        batch_logmel = torch.zeros(batch_size, max_frames, _FILTER_BANKS, dtype=torch.float32)

        batch_transcript = torch.zeros(batch_size, max_len_transcript, dtype=torch.int)

        for step, (log_mel, transcript) in enumerate(batch):
            # process each single sample and add to batch

            batch_logmel[step].narrow(0, 0, log_mel.size(0)).copy_(log_mel)
            batch_transcript[step].narrow(0, 0, len(transcript)).copy_(transcript)
            # return log_mel, transcript

        return batch_logmel, batch_transcript


# check
if __name__ == "__main__":
    librispeech_vocab = LibriSpeechVocabRAW()

    # aaa
    train_set = TrainSet(vocab= librispeech_vocab, csv_file="dev-clean.csv", root_dir="librispeech/dev-custom-clean")
    # for step in range(train_set.__len__()):
    #     print(f"Audio: {train_set[step][0].shape} Transcript: {train_set[step][1]}")

    data_loader = TrainLoader(dataset=train_set, batch_size=4, shuffle=False)
    for step, (log_mel, transcript) in tqdm(enumerate(data_loader)):
        pass
        print(f"Audio: {log_mel} Shape{log_mel.shape} "
              f"Transcript: {transcript}")

    # logMel = torch.randn([1, 81, 204])
    # max_frames = 400
    # batch_logmel = torch.zeros(4, max_frames, 81, dtype=torch.float32)
    # # narrow: input, dim, start, length
    # batch_logmel[2].narrow(0, 0, batch_logmel.size(1)).copy_(logMel)
    # print(batch_logmel)

    print("DONE")