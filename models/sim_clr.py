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


class AudioTransforms:
    # transform function contains
    # pitch shift, reverbration, adding noise
    # advanced techniques - time masking, freq masking
    def __init__(self, rate=16_000) -> None:
        self.rate = rate

        # transform function input should be mel spectrogram
        self.transforms_waveform = [
            self.pitch_shift,  # shifting audio pitch
            self.reverb_audio,  # reverb audio with room echo sound
        ]

        self.transforms_melspectrogram = [self.time_mask, self.freq_mask]

    def transforms_wavform_2_melspec(self, audio_array):
        mel_transform = T.MelSpectrogram()
        mel_spec = mel_transform(audio_array)
        return mel_spec

    def _load_audio_signal(self, audio_path):
        audio_array, _ = torchaudio.load(audio_path)
        return audio_array

    def audio_augment(self, audio_path):
        audio_array = self._load_audio_signal(audio_path)
        # random audio aug function
        random_transform_wavform_func = random.choice(self.transforms_waveform)
        audio_array = random_transform_wavform_func(audio_array)

        # convert to mel-spectrogram
        log_mel_spec = self.transforms_wavform_2_melspec(audio_array)
        # random aug log mel spectrogram
        log_mel_spec_func = random.choice(self.transforms_melspectrogram)
        log_mel_spec = log_mel_spec_func(log_mel_spec)

        return log_mel_spec

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

    def time_mask(self):
        pass

    def freq_mask(self):
        pass

    def add_noise(self):
        pass

    def __call__(self, input):
        # input is audio_path
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
