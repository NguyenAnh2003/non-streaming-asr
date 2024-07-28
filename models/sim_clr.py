import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
import augment
import random
import torchaudio.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, lr_scheduler
from transformers import AutoModel, WhisperTokenizer, WhisperFeatureExtractor
import torchaudio
import numpy as np
from lightly.loss import NTXentLoss # https://docs.lightly.ai/self-supervised-learning/examples/simclr.html


class BasicASRModel(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf  # already create as DictConfig with OmegaConf
        self.encoder = self._pretrained_encoder()

        # acting as projection head
        # https://arxiv.org/pdf/2212.11491 dive into projection head
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.conf.model.mlp.in_feats,
                out_features=self.conf.model.mlp.out_feats,
                bias=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                in_features=self.conf.model.mlp.out_feats, out_features=128, bias=True
            ),
        )

        self.softmax = nn.Softmax(dim=-1)  # softmax the last dim

    def _pretrained_encoder(self):
        asr_model = AutoModel.from_pretrained(self.conf.model.pretrained_model)
        encoder = asr_model.encoder

        # freeze encoder
        if self.conf.model.freeze == True:
            for params in encoder.parameters():
                params.requires_grad = False

        return encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder(x)
        out = self.mlp(encoder_out)
        out = self.softmax(out)
        return out


class SpectrogramToDB(object):
    """Turns a spectrogram from the power/amplitude scale to the decibel scale.
    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip. This method is sourced from an earlier release of torchaudio and
    is no longer present in current versions.
    Args:
        stype (str): scale of input spectrogram ("power" or "magnitude").  The
            power being the elementwise square of the magnitude. default: "power"
        top_db (float, optional): minimum negative cut-off in decibels.  A reasonable number
            is 80.
    """

    def __init__(self, stype="power", top_db=None):
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = np.log10(np.maximum(self.amin, self.ref_value))

    def __call__(self, spec):
        # numerically stable implementation from librosa
        # https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html
        spec_db = self.multiplier * torch.log10(torch.clamp(spec, min=self.amin))
        spec_db -= self.multiplier * self.db_multiplier

        if self.top_db is not None:
            spec_db = torch.max(
                spec_db, spec_db.new_full((1,), spec_db.max() - self.top_db)
            )
        return spec_db


class AudioTransforms:
    # transform function contains
    # pitch shift, reverbration, adding noise
    # advanced techniques - time masking, freq masking
    def __init__(self, rate=16_000) -> None:
        self.rate = rate
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-base"
        )

        # transform function input should be mel spectrogram
        self.augment_waveform = [
            self.pitch_shift,  # shifting audio pitch
            self.reverb_audio,  # reverb audio with room echo sound
        ]

        self.augment_melspec = [self.time_mask, self.freq_mask]

    def _load_audio_signal(self, audio_path):
        audio_array, _ = torchaudio.load(audio_path)
        return audio_array

    def transforms_wavform_2_melspec(self, audio_array):
        # melspec transform function
        # using WhisperFeatureExtractor to compute mel-spectrogram
        mel_spec = self.whisper_feature_extractor(
            audio_array, self.rate, return_tensors="pt"
        ).input_features  #
        mel_spec = SpectrogramToDB(stype="magnitude", top_db=8000)(mel_spec)
        return mel_spec

    def _feature_extraction_original(self, audio_path):
        # melspec transform function
        audio_array = self._load_audio_signal(audio_path)
        audio_array = audio_array.squeeze(0)  # remove batch dim
        mel_spec = self.transforms_wavform_2_melspec(audio_array)
        print(f"mel: {mel_spec.shape}")
        return mel_spec

    def audio_augment(self, audio_path):
        audio_array = self._load_audio_signal(audio_path)
        # random audio aug function
        for funcs in self.augment_waveform:
            audio_array = funcs(audio_array)

        audio_array = audio_array.squeeze(0)  # remove batch dim

        # convert to mel-spectrogram
        mel_spec = self.transforms_wavform_2_melspec(audio_array)
        # random aug mel spectrogram
        for funcs in self.augment_melspec:
            mel_spec = funcs(mel_spec)

        return mel_spec

    def reverb_audio(self, audio_array):
        reverbered_audio = (
            augment.EffectChain()
            .reverb(100, 80, 90)
            .channels(1)
            .apply(audio_array, src_info={"rate": self.rate})
        )

        return reverbered_audio

    def pitch_shift(self, audio_array):
        pitch_shifted_audio = (
            augment.EffectChain()
            .pitch(200)
            .rate(self.rate)
            .apply(audio_array, src_info={"rate": self.rate})
        )
        return pitch_shifted_audio

    def time_mask(self, spec, T=40, num_masks=1, replace_with_zero=False):
        cloned = spec.clone()
        len_spectro = cloned.shape[2]

        for i in range(0, num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if replace_with_zero:
                cloned[0][:, t_zero:mask_end] = 0
            else:
                cloned[0][:, t_zero:mask_end] = cloned.mean()
        return cloned

    def freq_mask(self, spec, F=30, num_masks=2, replace_with_zero=False):
        cloned = spec.clone()
        num_mel_channels = cloned.shape[1]

        for i in range(0, num_masks):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            if replace_with_zero:
                cloned[0][f_zero:mask_end] = 0
            else:
                cloned[0][f_zero:mask_end] = cloned.mean()

        return cloned

    def add_noise(self):
        pass

    def __call__(self, input):
        # input is audio path
        x1 = self.audio_augment(input)  # keep one original
        x2 = self.audio_augment(input)
        return x1, x2


class SimCLR(pl.LightningModule):
    def __init__(self, conf: DictConfig = None) -> None:
        super().__init__()
        self.conf = OmegaConf.create(conf)
        self.asr_model = BasicASRModel(conf)
        self.temperature = 0.55
        self.nt_xent_loss = NTXentLoss()

    def configure_optimizers(self):
        optim = Adam(
            self.model.parameters(),
            lr=self.conf.model.train.lr,
            weight_decay=self.conf.model.train.weight_decay,
        )

        scheduler = lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=self.conf.model.train.max_epochs,
            eta_min=self.conf.model.train.lr / 50,
        )

        return [optim], [scheduler]

    def loss(self, batch, mode="train"):
        # integrate with speech model (asr model)
        melspecs, _ = batch  # get melspec augmented

        xi, xj = melspecs
        zi = self.asr_model(xi)
        zj = self.asr_model(xj)
        # loss
        sim_matrix = self._calc_sim(zi, zj) / self.temperature
        sim_matrix = torch.exp(sim_matrix)

        return loss

    def info_nce_loss(self, batch, mode="train"):
        melspecs, _ = batch  # get melspec augmented

        xi, xj = melspecs
        zi = self.asr_model(xi)
        zj = self.asr_model(xj)


        self.log("train_loss")
        return

    def _calc_sim(self, zi, zj):
        representations = torch.cat((zi, zj), dim=0)
        sim_matrx = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
        )
        return sim_matrx

    def training_step(self, batch, batch_idx):
        # return loss each step - lightning module include backward
        # process not need to pay attention
        return self.loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.loss(batch, mode="val")
