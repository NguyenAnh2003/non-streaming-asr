import torchaudio

audio, sample = torchaudio.load("../test.flac")

print(f"Audio array: {audio} Sample: {sample}")