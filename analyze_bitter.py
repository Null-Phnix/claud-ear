#!/usr/bin/env python3
"""Quick audio analysis for bitter.wav"""
import sys
import os
import json

AUDIO_FILE = "/Users/josii/Documents/music/music data/Biteki_びてき_homesick_-_bitter_lyrics.wav"
OUTPUT_FILE = "/tmp/bitter_analysis.json"

results = {}

# Basic file info
try:
    stat = os.stat(AUDIO_FILE)
    results["file_size_bytes"] = stat.st_size
    results["file_size_mb"] = round(stat.st_size / (1024 * 1024), 2)
except Exception as e:
    results["file_error"] = str(e)

# Try mutagen for WAV metadata
try:
    import mutagen
    from mutagen.wave import WAVE
    audio = WAVE(AUDIO_FILE)
    results["duration_seconds"] = round(audio.info.length, 2)
    results["duration_minutes"] = round(audio.info.length / 60, 2)
    results["sample_rate"] = audio.info.sample_rate
    results["channels"] = audio.info.channels
    results["bits_per_sample"] = getattr(audio.info, 'bits_per_sample', 'unknown')
except Exception as e:
    results["mutagen_error"] = str(e)

# Try librosa for tempo and key
try:
    import librosa
    import numpy as np
    y, sr = librosa.load(AUDIO_FILE, sr=None, duration=120)

    # Tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    results["estimated_bpm"] = round(float(tempo), 1)

    # Key estimation via chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    dominant_note = notes[chroma_mean.argmax()]
    results["dominant_note"] = dominant_note

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    results["spectral_centroid_mean"] = round(float(spectral_centroids.mean()), 2)

    # RMS energy
    rms = librosa.feature.rms(y=y)
    results["rms_energy_mean"] = round(float(rms.mean()), 6)
    results["rms_energy_db"] = round(float(librosa.amplitude_to_db(rms.mean())), 2)

    # Onset strength (for rhythm feel)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    results["onset_strength_mean"] = round(float(onset_env.mean()), 4)

    print("librosa analysis complete")
except Exception as e:
    results["librosa_error"] = str(e)

# Save results
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
