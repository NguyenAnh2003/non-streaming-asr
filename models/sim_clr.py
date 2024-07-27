import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
import augment
import random
import torchaudio.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, lr_scheduler
from models.asr_model import ASRModel
from transformers import AutoModel
import torchaudio
import numpy as np


class BasicASRModel(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf  # already create as DictConfig with OmegaConf
        self.encoder = self._pretrained_encoder()
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

    def forward(self, x: torch.Tensor):
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
        mel_transform = T.MelSpectrogram(sample_rate=self.rate, 
                                         n_mels=80, 
                                         n_fft=640, 
                                         win_length=640, 
                                         hop_length=321, 
                                         f_min=-80, 
                                         f_max=8000, 
                                         pad=0)
        
        mel_spec = mel_transform(audio_array)
        mel_spec = SpectrogramToDB(stype='magnitude', top_db=8000)(mel_spec)
        return mel_spec

    def _feature_extraction_original(self, audio_path):
        # melspec transform function
        audio_array = self._load_audio_signal(audio_path)
        mel_spec = self.transforms_wavform_2_melspec(audio_array)
        return mel_spec

    def audio_augment(self, audio_path):
        audio_array = self._load_audio_signal(audio_path)
        # random audio aug function
        for funcs in self.augment_waveform:
            audio_array = funcs(audio_array)

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
            if (t_zero == t_zero + t): return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
            else: cloned[0][:,t_zero:mask_end] = cloned.mean()
        return cloned

    def freq_mask(self, spec, F=30, num_masks=2, replace_with_zero=False):
        cloned = spec.clone()
        num_mel_channels = cloned.shape[1]
        
        for i in range(0, num_masks):        
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f) 
            if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
            else: cloned[0][f_zero:mask_end] = cloned.mean()
        
        return cloned

    def add_noise(self):
        pass

    def __call__(self, input):
        # input is audio path
        x1 = self._feature_extraction_original(input) # keep one original
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

    def calc_similarity_batch(self, x1, x2):
        representations = torch.cat([x1, x2], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

    def forward(self, x1, x2):
        # dim input?
        zi = F.normalize(x1, p=2, dim=1)
        zj = F.normalize(x2, p=2, dim=1)

        # shape: (batch, seq, dim)
        sim_matrix = self.calc_similarity_batch(zi, zj)

        sim_ij = torch.diag(sim_matrix, self.conf.model.train.batch_size)
        sim_ji = torch.diag(sim_matrix, -self.conf.model.train.batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)


class SimCLR(pl.LightningModule):
    def __init__(self, conf: DictConfig = None) -> None:
        super().__init__()
        self.conf = OmegaConf.create(conf)
        self.model = ASRModel(conf)
        self.loss_func = ConstrastiveLoss()

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

    def nt_xent_loss(self, batch, mode="train"):
        # integrate with speech model (asr model)
        audio, _ = batch

        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass
