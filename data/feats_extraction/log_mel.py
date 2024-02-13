import math
from utils.utils import get_configs
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from utils.visualizer import plot_melspectrogram

""" Taking log mel spectrogram of audio """
def audio_transforms(array, params):
    """
    :param array: audio array get by torchaudio
    :param params: params config in yaml file
    :return: log mel spectrogram
    """

    # define mel spec transform function
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

if __name__ == "__main__":
    filepath = "../examples/test.wav"
    array, _ = torchaudio.load(filepath)
    params = get_configs("../../configs/audio_extraction.yaml")
    log_melspec = audio_transforms(array, params)
    print(f"Log mel: {log_melspec} Shape: {log_melspec.shape}")
    log_mel_plot = torch.squeeze(log_melspec)
    plot_melspectrogram(specgram=log_mel_plot, title="Log Mel", ylabel="banks")