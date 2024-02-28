import torchaudio
import torchaudio.functional as F
import torch
import librosa
from utils.visualizer import plot_melspectrogram, plot_waveform
from utils.utils import get_configs

# including preprocessing and post-processing
_params = get_configs("../configs/audio_processing.yaml")


# Audio preprocessing
def _trim_audio(audio_array, params):
    """
    :param audio_array
    :param params: configs in yaml file
    :return: trimmed audio array
    """
    trimmed_audio, _ = librosa.effects.trim(y=audio_array, top_db=params["top_db"])

    # return trimmed audio -> audio array
    return trimmed_audio

if __name__ == "__main__":

    # audio processing Trim
    array, sr = torchaudio.load("./examples/kkk.flac")
    trimmed_array = _trim_audio(audio_array=array, params=_params)

    # plotting
    plot_waveform(waveform=array, sr=sr) # plotting audio before trimmed
    plot_waveform(waveform=trimmed_array, sr=sr)

    # mel spectrogram

    print("DONE")