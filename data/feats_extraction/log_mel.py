import math

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

""" Taking log mel spectrogram of audio """
def audio_transforms(array, params):
    """
    :param array: audio array get by torchaudio
    :param params: params config in yaml file
    :return: log mel spectrogram
    """
    F_mel = T.MelSpectrogram(
        sample_rate=params["sample_rate"],
        n_fft=params["n_fft"],
        win_length=params["win_length"],
        hop_length=params["hop_length"],
        window_fn=eval(params["window_fn"]),
        center=params["center"],
        pad_mode=params["pad_mode"],
        power=params["power"],
        norm=params["norm"],
        n_mels=params["n_mels"],
        mel_scale=params["mel_scale"],
    )

    # get mel spectrogram
    mel_spectrogram = F_mel(array)

    # log mel spectrogram
    log_melspectrogram = F.amplitude_to_DB(mel_spectrogram, multiplier=10, amin=1e-10,
                                           db_multiplier=math.log10(max(1e-10, 1)))

    # adjust output
    return log_melspectrogram