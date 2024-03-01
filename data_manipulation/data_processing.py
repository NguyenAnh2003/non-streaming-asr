import torchaudio
import torchaudio.functional as F
import torch
import librosa
from utils.visualizer import plot_melspectrogram, plot_waveform
from utils.utils import get_configs
from pydub import AudioSegment

# including preprocessing and post-processing
_params = get_configs("../configs/audio_processing.yaml")


# Audio preprocessing
def _trim_audio(audio_array, params):
    """ Trim audio with Librosa
    :param audio_array
    :param params: configs in yaml file
    :return: trimmed audio array
    """
    trimmed_audio, _ = librosa.effects.trim(y=audio_array,
                                            top_db=params["top_db"],
                                            frame_length=params["win_length"],
                                            hop_length=params["hop_length"])

    # return trimmed audio -> audio array
    return trimmed_audio

def _audio_segmenter():
    segmented_audio = AudioSegment

if __name__ == "__main__":

    # audio processing Trim
    array, sr = torchaudio.load("./examples/kkk.flac")
    trimmed_array = _trim_audio(audio_array=array, params=_params)

    print(f"Shape before trimmed: {array.shape} After trimmed: {trimmed_array.shape}")

    # plotting
    plot_waveform(waveform=array, sr=sr) # plotting audio before trimmed
    plot_waveform(waveform=trimmed_array, sr=sr)

    # mel spectrogram

    print("DONE")