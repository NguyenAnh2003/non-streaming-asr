import matplotlib.pyplot as plt
import torch
import librosa
import numpy as np
import json

def plot_bar(data):
    """
    :param data: dictionary with keys as labels and values as corresponding values
    :type data: dict
    """
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set
    WER = []
    CER = []
    MODEL_NAMES = []

    for item in data:
        for model_name, metrics in item.items():
            wer = metrics["WER"] * 100
            cer = metrics["CER"] * 100
            WER.append(wer)
            CER.append(cer)
            MODEL_NAMES.append(model_name)

    # Set position of bar on X axis
    br1 = np.arange(len(WER))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    bars1 = plt.bar(br1, WER, color='blue', width=barWidth,
                    edgecolor='grey', label='WER')
    bars2 = plt.bar(br2, CER, color='red', width=barWidth,
                    edgecolor='grey', label='CER')

    # Adding annotations
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 3), ha='center', va='bottom')

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 3), ha='center', va='bottom')

    # Adding Xticks
    plt.xlabel('Model', fontweight='bold', fontsize=15)
    plt.ylabel('Error rate', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(WER))], MODEL_NAMES)

    plt.legend()
    plt.show()

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

if __name__ == "__main__":
    data = [{"Fast Conformer": {"WER": 0.04222150048717116, "CER": 0.015581577808900198}},
            {"Finetuned Model": {"WER": 0.043520623579084115, "CER": 0.015038972275790262}},
            {"Conformer ctc small": {"WER": 0.08144355501213152, "CER": 0.03227403045923492}}]

    plot_bar(data)