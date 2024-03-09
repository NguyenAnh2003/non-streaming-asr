import torchaudio
import torchaudio.functional as F
import torch
import librosa
from utils.visualizer import plot_melspectrogram, plot_waveform
from utils.utils import get_configs
from pydub import AudioSegment
import transformers

# Audio Augmentation: https://pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html

# including preprocessing and post-processing
_params = get_configs("../configs/audio_processing.yaml")

# noise file url
_NOISE_SUBSETS = [
    "",
    ""
]

# adding noise
def add_noise2audio(sample_array: torch.Tensor, noise_array: torch.Tensor):
    """ SNR explained: https://www.linkedin.com/pulse/signal-to-noise-ratio-snr-explained-leonid-ayzenshtat/
    :param sample_array: torch.Tensor,
    :param noise_array
    :return augmented audio with noise
    """
    snr_dbs = torch.tensor([20, 10, 3])
    
    # length for audio and noise -> focus on audio size
    length_augmented = sample_array.size()
    
    # augmented_audio
    augmented_audio = F.add_noise(waveform=sample_array, noise=noise_array, snr=snr_dbs,
                                  lengths=length_augmented)
    
    return augmented_audio #

# use later
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

# use later = ))
def _audio_segmentation(path: str):
    """ :param -> wav file path
    :returns audio segmented
    """
    segmented_audio = AudioSegment.from_wav(file=path)

    return segmented_audio

if __name__ == "__main__":

    # audio processing Trim
    # array, sr = torchaudio.load("./examples/kkk.flac")
    # trimmed_array = _trim_audio(audio_array=array, params=_params)

    # print(f"Shape before trimmed: {array.shape} After trimmed: {trimmed_array.shape}")

    # adding noise

    print("DONE")