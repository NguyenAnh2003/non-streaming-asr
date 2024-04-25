import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch
import librosa
from utils.visualizer import plot_melspectrogram, plot_waveform
from utils.utils import get_configs
from pydub import AudioSegment
from datasets import load_dataset
import librosa
import transformers

# including preprocessing and post-processing
_params = get_configs("../configs/audio_processing.yaml")

# adding background noise
def _add_noise2audio(sample_array: torch.Tensor, noise_array: torch.Tensor):
    """ 
    :param sample_array: torch.Tensor,
    :param noise_array
    :return augmented audio with noise
    """
    # work with noise have tensor([>= 1, ...n_frames]) 2 channels - audio with 2 channels can be considered as stereo sound
    noise_array = noise_array[0, :sample_array.size(1)] # take n_frames -> vector
    
    noise_array = noise_array.unsqueeze(0) # turn back to matrix reduce 1 channel
    
    # process noise_array
    scaled_noise_arr = noise_array[:, :sample_array.size(1)] # noise array must be tensor([1, ... n_frames])
    
    snr_dbs = torch.tensor([20, 10, 3])
    
    # augmented_audio
    augmented_audio = F.add_noise(waveform=sample_array, noise=scaled_noise_arr, snr=snr_dbs)
    
    return augmented_audio

# gaussian noise

# change pitch
def _audio_pitch_shift(sample_array: torch.Tensor, params):
    # hanning function config window_fn
    params["window_fn"] = torch.hann_window #

    # 
    PichShift_F = T.PitchShift(sample_rate=params["sample_rate"],
                        n_steps=params["pshift_steps"],
                        bins_per_octave=params["pshift_bins_per_octave"],
                        n_fft=params["n_fft"], win_length=params["win_length"],
                        hop_length=params["hop_length"],
                        window_fn=params["window_fn"]) # 

    # get pitch shifted audio
    pshifted_audio = PichShift_F(sample_array)

    # 
    return pshifted_audio    
    

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

# preprocessing with huggingface dataset
def _tolower(point):
    point['sentence'] = point['sentence'].lower()
    return point

def _get_duration(point):
    # audio path process
    def _process_audio_path(point):
        path = "./" + point['audio']['path']
        return path
    
    # ttt
    audio_path = _process_audio_path(point)
    point['duration'] = librosa.core.get_duration(path=audio_path)
    return point

def preprocess_ds(dataset):
    # process each sample
    def _inner_func(point):
        point = _tolower(point)
        point = _get_duration(point)
        return point
    
    # mapping each sample to be processed
    dataset = dataset.map(lambda x: _inner_func(x))
    return dataset


if __name__ == "__main__":
    dataset = load_dataset('vivos')
    print("DONE")