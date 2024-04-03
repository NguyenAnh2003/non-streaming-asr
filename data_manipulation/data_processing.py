import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch
import librosa
from utils.visualizer import plot_melspectrogram, plot_waveform
from utils.utils import get_configs
from pydub import AudioSegment
import transformers

# Audio Augmentation: https://pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html

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

if __name__ == "__main__":

    # audio processing Trim
    array, sr = torchaudio.load("./examples/kkk.flac")
    # trimmed_array = _trim_audio(audio_array=array, params=_params)

    # print(f"Shape before trimmed: {array.shape} After trimmed: {trimmed_array.shape}")

    # adding noise
    # n_array, _ = torchaudio.load("./noises/re_tam.wav")
    # print(n_array.shape)
    
    # augmented = _add_noise2audio(sample_array=array, noise_array=n_array) #
    # print(augmented.shape)
    
    ps_audio = _audio_pitch_shift(array, _params)
    print(ps_audio.shape)
    

    print("DONE")