import torch
import torch.nn.functional as F
import torchaudio
from utils.visualizer import plot_waveform

array, rate = torchaudio.load("../data_manipulation/noises/my_noise0.wav")
plot_waveform(waveform=array, sr=rate)