import os

from omegaconf import OmegaConf, DictConfig
import torch
import torchaudio
import librosa
import torchaudio.functional as F
import torchaudio.transforms as T
import math
import pandas as pd
import json
import csv


class DataProcessingPipeline:
    def __init__(self, conf: DictConfig = None):
        self.conf = OmegaConf.create(conf)

    def audio_transforms(self, sample_array):
        """
        :param array: audio array get by torchaudio
        :param params: params config in yaml file
        :return: log mel spectrogram
        """

        # define mel spec transform function
        F_mel = T.MelSpectrogram(
            sample_rate=self.conf.processor.sample_rate,
            n_fft=self.conf.processor.n_fft,
            win_length=self.conf.processor.win_length,
            hop_length=self.conf.processor.hop_length,
            window_fn=eval(self.conf.processor.window_fn),
            center=self.conf.processor.center,
            pad_mode=self.conf.processor.pad_mode,
            power=self.conf.processor.power,
            norm=self.conf.processor.norm,
            n_mels=self.conf.processor.n_mels,
            mel_scale=self.conf.processor.mel_scale,
        )

        # get mel spectrogram
        mel_spectrogram = F_mel(sample_array)

        # log mel spectrogram
        log_melspectrogram = F.amplitude_to_DB(
            mel_spectrogram,
            multiplier=10,
            amin=1e-10,
            db_multiplier=math.log10(max(1e-10, 1)),
        )

        # adjust output
        # standard output: [bannks, n_frames (times)]
        # the banks can be represent the in channels for CNN can be considered as standard channels,
        # the n_frames cannot be channels because the data in not consistent in distribution in n_frames (time).
        return log_melspectrogram.squeeze(0).contiguous().transpose(0, 1)

    def add_noise2audio(self, sample_array: torch.Tensor, noise_array: torch.Tensor):
        """
        :param sample_array: torch.Tensor,
        :param noise_array
        :return augmented audio with noise
        """
        # work with noise have tensor([>= 1, ...n_frames]) 2 channels - audio with 2 channels can be considered as stereo sound
        noise_array = noise_array[0, : sample_array.size(1)]  # take n_frames -> vector

        noise_array = noise_array.unsqueeze(0)  # turn back to matrix reduce 1 channel

        # process noise_array
        scaled_noise_arr = noise_array[
            :, : sample_array.size(1)
        ]  # noise array must be tensor([1, ... n_frames])

        snr_dbs = torch.tensor([20, 10, 3])

        # augmented_audio
        augmented_audio = F.add_noise(
            waveform=sample_array, noise=scaled_noise_arr, snr=snr_dbs
        )

        return augmented_audio

    def _audio_pitch_shift(sample_array: torch.Tensor, params):
        # hanning function config window_fn
        params["window_fn"] = torch.hann_window  #

        #
        PichShift_F = T.PitchShift(
            sample_rate=params["sample_rate"],
            n_steps=params["pshift_steps"],
            bins_per_octave=params["pshift_bins_per_octave"],
            n_fft=params["n_fft"],
            win_length=params["win_length"],
            hop_length=params["hop_length"],
            window_fn=params["window_fn"],
        )  #

        # get pitch shifted audio
        pshifted_audio = PichShift_F(sample_array)

        #
        return pshifted_audio

    def _trim_audio(audio_array, params):
        """Trim audio with Librosa
        :param audio_array
        :param params: configs in yaml file
        :return: trimmed audio array
        """
        trimmed_audio, _ = librosa.effects.trim(
            y=audio_array,
            top_db=params["top_db"],
            frame_length=params["win_length"],
            hop_length=params["hop_length"],
        )

        # return trimmed audio -> audio array
        return trimmed_audio

    def tolower(self, transcript):
        transcript = transcript.lower()
        return transcript

    def get_duration(self, path):
        duration = librosa.core.get_duration(path=path, sr=16000)
        return duration

    def build_metadata_manifests(self, path: str, outpath: str):
        # example 
        # build_data_manifest("./metadata/ls/test-other.csv", "./metadata/ls/test-other-manifest.json")
        reader = csv.reader(open(path, "r", encoding="utf-8"))
        next(reader)
        with open(outpath, "w") as fout:
            for line in reader:
                try:
                    duration = float(line[2])
                    meta_data = {
                        "audio_filepath": line[-1],
                        "duration": duration,
                        "text": line[1],
                    }
                    json.dump(meta_data, fout)
                    fout.write("\n")
                except ValueError:
                    raise ValueError("Cannot convert str to float")


class LibriSpeechDataProcessingPipeline:
    def __init__(self, conf: DictConfig):
        super().__init__(DataProcessingPipeline)
        self.conf = OmegaConf.create(conf)

    def prepare_ds_nemo(self,
                        path,
                        root_dir = "./librispeech/test-custom-other/",
                        dest_dir = "../data_manipulation/librispeech/test-custom-other/",
                        metadata_path = "./metadata/ls/test-other.csv"):
        # process each sample
        df = pd.read_csv(path)

        # preprocess durations
        durations = []
        audio_paths = []
        for audio in df["audio_id"]:
            audio_path = root_dir + audio + ".flac"
            a2 = dest_dir + audio + ".flac"
            duration = self.get_duration(audio_path)
            durations.append(duration)
            audio_paths.append(a2)
        df["duration"] = durations
        df["path"] = audio_paths

        # preprocess audio transcripts
        lower_transcripts = []
        for audio_transcript in df["transcript"]:
            lower_transcripts.append(audio_transcript.lower())
        df["transcript"] = lower_transcripts
        df.to_csv(metadata_path, index=False)

    def build_data_manifest(path: str, outpath: str):
        reader = csv.reader(open(path, "r", encoding="utf-8"))
        next(reader)
        with open(outpath, 'w') as fout:
            for line in reader:
                try:
                    duration = float(line[2])
                    meta_data = {
                        "audio_filepath": line[-1],
                        "duration": duration,
                        "text": line[1],
                    }
                    json.dump(meta_data, fout)
                    fout.write("\n")
                except ValueError:
                    raise ValueError("Cannot convert str to float")


class LibriSpeechDataProcessingPipeline(DataProcessingPipeline):
    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = OmegaConf.create(conf)


class VNDataProcessingPipeline(DataProcessingPipeline):
    def __init__(self, conf: DictConfig = None):
        super().__init__()
        self.conf = OmegaConf.create(conf)

    def prepare_ds_csv(self, root_dir: str, dest_path):
        with open(dest_path, 'w', newline='', encoding='utf-8') as meta_file:
            csv_writer = csv.writer(meta_file) # writer CSV
            csv_writer.writerow(['audio_id', 'path', 'transcript', 'duration'])

            # filter txt file
            for filename in os.listdir(root_dir):
                if filename.endswith("txt"):
                    txt_path = os.path.join(root_dir, filename)
                    wav_path = txt_path[:-4] + ".wav"
                    with open(txt_path, 'r', encoding="utf-8") as text_file:
                        transript = text_file.read()
                        duration = self.get_duration(wav_path)
                        csv_writer.writerow([filename[:-4], wav_path, transript, duration])
                        # print(f"id: {filename[:-4]} path: {wav_path} duration: {duration} transcript: {transript}")
        print("DONE")

if __name__ == "__main__":
    data_pipe = VNDataProcessingPipeline()
    data_pipe.prepare_ds_csv(root_dir="/media/vkuai/NTFS_Data/nguyenanh_dataset/speech/vlsp_2020",
                             dest_path="./metadata/vn/vlsp-2020-100h.csv")