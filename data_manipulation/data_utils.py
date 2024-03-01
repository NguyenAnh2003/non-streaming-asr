import torch
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
import os
import csv
import pandas as pd
from logger.my_logger import setup_logger
import shutil
from feats_extraction.log_mel import audio_transforms
from typing import List, Tuple
import torchaudio
import torch
from tqdm import tqdm
from utils.utils import get_configs

_params = get_configs("../configs/audio_processing.yaml")

_logger = setup_logger("../logger/logs/write_csv.log", location="data utils")
_logger.getLogger(__name__)

URL = "dev-clean"

FOLDER_IN_ARCHIVE = "LibriSpeech"

SAMPLE_RATE = 16000

_DATA_SUBSETS = [
    "dev-clean", #
    "dev-other",
    "test-clean", #
    "test-other",
    "train-clean-100", #
    "train-clean-360", #
    "train-other-500", #
]

_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",  # noqa: E501
    "http://www.openslr.org/resources/12/dev-other.tar.gz": "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",  # noqa: E501
    "http://www.openslr.org/resources/12/test-clean.tar.gz": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",  # noqa: E501
    "http://www.openslr.org/resources/12/test-other.tar.gz": "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz": "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",  # noqa: E501
    "http://www.openslr.org/resources/12/train-other-500.tar.gz": "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2",  # noqa: E501
}

# LibriSpeech data utils
def download_libirspeech_dataset(root: str = f"D:\\", url: str = URL):
    """ auto download librispeech dataset """
    # base url
    base_url = "http://www.openslr.org/resources/12/"
    # extension file
    extension = ".tar.gz"

    # filename with url(openslr url) + extension file
    filename = url + extension
    archive = os.path.join(root, filename) # binding archive folder
    download_url = os.path.join(base_url, filename)
    if not os.path.isfile(archive):
        checksum = _CHECKSUMS.get(download_url, None) #?
        download_url_to_file(download_url, archive) # download tar file
    _extract_tar(archive) # extract archive Libirspeech folder?

def _process_librispeech_dataset(metadata_file_path):
    """ metadata from txt to csv
    audio_id, transcript
    """
    metadata_dict = {}
    with open(metadata_file_path, 'r') as file:
        for line in file:
            # Split each line into audio_id and transcript using space as a delimiter
            parts = line.strip().split(' ', 1)

            # Ensure that the line has at least two parts (audio_id and transcript)
            if len(parts) == 2:
                audio_id, transcript = parts
                metadata_dict[audio_id] = transcript
    # logging
    _logger.log(_logger.INFO, "GET METADATA DICT")
    return metadata_dict

def write_metadata_txt_2_csv(csv_path: str):
    # define metadata
    _logger.log(_logger.INFO, "DEFINE METADATA")
    metadata_dict = _process_librispeech_dataset("librispeech/train-transcripts.txt")

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        # define csv writer
        csv_writer = csv.writer(csv_file)

        # write row
        csv_writer.writerow(['audio_id', 'transcript'])

        for audio_id, transcript in metadata_dict.items():
            csv_writer.writerow([audio_id, transcript])

    _logger.log(_logger.INFO,  "WRITE CSV COMPETE")

def combine_audio_from_folders():
    source_folder_path = "./librispeech/train-clean-100/train-clean-100"
    destination_folder_path = "librispeech/train-custom-clean"

    # list all child folder in parent
    for folder_name in os.listdir(source_folder_path):
        # binding child folder with parent folder
        child_folder_path = os.path.join(source_folder_path, folder_name)

        if os.path.isdir(child_folder_path):
            # get all files in one child folder
            for folder_child_name in os.listdir(child_folder_path):
                child2_folder_path = os.path.join(child_folder_path, folder_child_name)
                if os.path.isdir(child2_folder_path):
                    for filename in os.listdir(child2_folder_path):
                        file_path = os.path.join(child2_folder_path, filename)
                        # print(f"From: {child2_folder_path} File: {filename}")
                        # copy and move to destination
                        shutil.copy(file_path, destination_folder_path)

def get_number_audio_samples():
    source_path = "librispeech/train-custom-clean"
    amounts = []
    for filename in os.listdir(source_path):
        if filename.endswith(".flac"):
            file_path = os.path.join(source_path, filename)
            if os.path.isfile(file_path):
                amounts.append(file_path)
    return len(amounts)

# LibriSpeech utils
def concat_transcripts_txt_file() -> None:
    # source path
    source_path = "./librispeech/train_libri_transcripts"
    # combined transcript txt file
    combined_txt_path = "librispeech/train-transcripts.txt"

    # open combined file
    with open(combined_txt_path, 'w', encoding='utf-8') as dest_file:

        # traverse each file
        for filename in os.listdir(source_path):
            file_path = os.path.join(source_path, filename)

            # open file and write to dest_file
            with open(file_path, 'r', encoding='utf-8') as infile:
                dest_file.write(infile.read())

# get longest audio
def _get_long_audio(source_path: str = "./librispeech/train-custom-clean") -> List[Tuple[torch.Tensor, str]]:
    # shape [n_frames, banks] get long audio from 900 n_frames
    laus = []
    print("Getting long audio")
    for filename in tqdm(os.listdir(source_path)):
        # file path can be represented for filename
        file_path = os.path.join(source_path, filename)

        # audio array load
        audio_array, _ = torchaudio.load(file_path)

        # get log mel shape
        logmel_sample = audio_transforms(array=audio_array, params=_params)

        # log mel shape [n_frames, banks]
        if logmel_sample.size(0) > 900:
            laus.append((logmel_sample.shape, file_path))

    print("DONE Getting long audio")
    return laus


if __name__ == "__main__":
    # flow process librispeech data folder
    # 1. combine all file in folder
    # 2. filter .flac and txt file
    # 3. combine content of txt file
    # 4. concat content of txt file
    # 5. write metadata from txt 2 csv file

    # dev-clean 2703; train-clean 28539
    # concat_transcripts_txt_file()
    # print(f"Number of samples in train-custom-clean: {get_number_audio_samples()}")
    #
    # source_path = "librispeech/train-transcripts.txt"
    # amounts = []
    # with open(source_path, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         amounts.append(line)
    # print(f"Length: {len(amounts)}")

    # write metadata from txt to csv
    # write_metadata_txt_2_csv("metadata-train-clean.csv")

    # double check csv file
    # train_df = pd.read_csv("metadata-train-clean.csv")
    # print(f"NUmber of samples in train-csv: {train_df.shape[0]}")
    #
    # source_path = "librispeech/train-custom-clean"
    # dest_path = "librispeech/train_libri_transcripts"
    # for filename in os.listdir(source_path):
    #     if filename.endswith(".txt"):
    #         file_path = os.path.join(source_path, filename)
    #         if os.path.isfile(file_path):
    #             shutil.move(file_path, dest_path)


    # write_metadata_txt_2_csv("./metadata-dev-clean.csv")

    # long audios
    las = _get_long_audio(source_path="./librispeech/train-custom-clean")
    print(f"List of long audio: {las}")

    print("DONE")