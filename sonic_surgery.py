import os
import sys
from pathlib import Path

# Add current dir to path
sys.path.append(os.getcwd())

from server import _download_with_ytdlp, separate_stems, DEMUCS_SR
import soundfile as sf

url = "https://www.youtube.com/watch?v=_2xZYb-lsS4"
output_dir = Path.home() / "Documents" / "music" / "music data"

print(f"Downloading {url}...")
result = _download_with_ytdlp(url, output_dir)
file_path = result["file_path"]
print(f"Downloaded to: {file_path}")

print("Separating stems (this might take a minute)...")
# Limit to 60 seconds for speed in this demo
stems = separate_stems(file_path, max_duration=60.0)

stem_dir = Path(file_path).parent / f"{Path(file_path).stem}_stems"
stem_dir.mkdir(parents=True, exist_ok=True)

saved = {}
for name, audio in stems.items():
    stem_path = stem_dir / f"{name}.wav"
    sf.write(str(stem_path), audio, DEMUCS_SR)
    saved[name] = str(stem_path)

print(f"Stems saved to: {stem_dir}")
for name, path in saved.items():
    print(f"  - {name}: {path}")
