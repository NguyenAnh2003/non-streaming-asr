import torchaudio

audio, sample = torchaudio.load("hehe.wav")

print(f"Audio array: {audio} Sample: {sample}")