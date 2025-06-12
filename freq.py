import librosa

y, sr = librosa.load("/Users/tagirahmetsin/Downloads/jul.wav", sr=None)
print(f"Частота дискретизации jul.wav: {sr}")