import os
import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# Add current dir to path
sys.path.append(os.getcwd())

bass_path = "/Users/josii/Documents/music/music data/Biteki_びてき_sewerperson_-_a_hundred_and_seventy_yards_Lyrics_AMV_stems/bass.wav"

print(f"Analyzing bass frequencies for: {bass_path}")

# Load the bass stem
y, sr = librosa.load(bass_path, sr=None)

# Calculate the spectrum
S = np.abs(librosa.stft(y))
frequencies = librosa.fft_frequencies(sr=sr)

# Get average magnitude per frequency
avg_mag = np.mean(S, axis=1)

# Find peak frequency
peak_idx = np.argmax(avg_mag)
peak_freq = frequencies[peak_idx]

# Find frequency range (where magnitude is above 10% of peak)
threshold = 0.1 * np.max(avg_mag)
active_indices = np.where(avg_mag > threshold)[0]
min_freq = frequencies[active_indices[0]]
max_freq = frequencies[active_indices[-1]]

print(f"Peak Bass Frequency: {peak_freq:.2f} Hz")
print(f"Active Frequency Range: {min_freq:.2f} Hz - {max_freq:.2f} Hz")

# Check for sub-bass content (below 60Hz)
sub_bass_mag = np.sum(avg_mag[frequencies < 60])
total_mag = np.sum(avg_mag)
sub_bass_percent = (sub_bass_mag / total_mag) * 100

print(f"Sub-Bass Content (<60Hz): {sub_bass_percent:.2f}% of total bass energy")

# Key estimation from bass (chroma)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
mean_chroma = np.mean(chroma, axis=1)
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
likely_key = notes[np.argmax(mean_chroma)]

print(f"Likely Key (from bass): {likely_key}")
