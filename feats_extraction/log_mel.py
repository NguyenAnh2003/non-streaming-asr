import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

def audio_transforms(array, params):
    """ Taking log mel spectrogram of audio """

    return