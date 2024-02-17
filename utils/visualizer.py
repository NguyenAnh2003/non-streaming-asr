import matplotlib.pyplot as plt
import torch
import librosa

# Plot waveform
def plot_waveform(waveform, sr, title="Waveform"):
    """ visualize wave form """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.set_xlabel("time")
    axes.set_ylabel("amplitude")
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)

# Plot Mel spectrogram
def plot_melspectrogram(specgram, title=None, ylabel=None):
    """ visualize mel spectrogram && log mel spectrogram """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Mel Spectrogram")
    axs.set_ylabel(ylabel=ylabel)
    axs.set_xlabel("frames")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)