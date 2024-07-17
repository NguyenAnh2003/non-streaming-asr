import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
import augment
import random
import torchaudio.transforms as T


class AudioAugmentation:
    # transform function contains
    # pitch shift, reverbration, adding noise
    # advanced techniques - time masking, freq masking
    def __init__(self, rate=16_000) -> None:
        self.rate = rate

        # transform function input should be mel spectrogram
        self.transforms_waveform = [
            self.pitch_shift,  # shifting audio pitch
            self.reverb_audio,  # reverb audio with room echo sound
            self.add_noise,
        ]

        self.transforms_log_melspectrogram = [self.time_mask, self.freq_mask]

    def transforms_wavform_2_log_melspec(self, audio_array):
        mel_transform = T.MelSpectrogram()
        mel_spec = mel_transform(audio_array)
        return mel_spec

    def audio_augment(self, audio_array):
        # random audio aug function
        random_transform_wavform_func = random.choice(self.transforms_waveform)
        audio_array = random_transform_wavform_func(audio_array)

        # convert to mel-spectrogram
        log_mel_spec = self.transforms_wavform_2_log_melspec(audio_array)
        # random aug log mel spectrogram
        log_mel_spec_func = random.choice(self.transforms_log_melspectrogram)
        log_mel_spec = log_mel_spec_func(log_mel_spec)

        return log_mel_spec

    def reverb_audio(self):
        reverbered_audio = (
            augment.EffectChain()
            .reverb(100, 80, 90)
            .channels(1)
            .apply(self.array, src_info={"rate": self.rate})
        )

        return reverbered_audio

    def pitch_shift(self):
        pitch_shifted_audio = (
            augment.EffectChain()
            .pitch(200)
            .rate(self.rate)
            .apply(self.array, src_info={"rate": self.rate})
        )
        return pitch_shifted_audio

    def time_mask(self):
        pass

    def freq_mask(self):
        pass

    def add_noise(self):
        pass

    def __call__(self, input):
        x1 = self.audio_augment(input)
        x2 = self.audio_augment(input)
        return x1, x2


class ProjectionHeadNetwork(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, bias: bool = True):
        super.__init__()
        self.fc1 = nn.Linear(in_features=in_feats, out_features=hidden_dim, bias=bias)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=out_feats, bias=bias)

        self.chain = nn.Sequential(self.fc1, self.relu, self.fc2)

    def forward(self, x):
        out = self.chain(x)
        return out


class ConstrastiveLoss(nn.Module):
    # implement based SimCLR framework
    # the working principle input -> split 2 sample and augment another one ->
    # perform with f(.) (can be transformer encoder) -> perform representation
    # projection head g(.)
    # calculate the cosine similarity
    def __init__(self, conf: DictConfig = None, temperature=0.7) -> None:
        super().__init__()
        self.conf = OmegaConf.create(conf)
        self.temperature = temperature  # self.conf.model.temperature

    def forward(self, x, y):
        pass
