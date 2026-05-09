#!/usr/bin/env python3
"""Transcribe all WAV files that haven't been transcribed yet. Saves lyrics to bitekivibes_lyrics_all.txt"""
import os
import sys
import json
import time
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

MUSIC_DIR = Path.home() / "Documents" / "music" / "music data"
OUTPUT_FILE = Path(__file__).parent / "all_playlist_lyrics.txt"

def get_already_transcribed():
    """Read existing lyrics files to find what's already been done."""
    done = set()
    for lyrics_file in [
        Path(__file__).parent / "all_lyrics.txt",
        Path(__file__).parent / "bitekivibes_lyrics.txt",
    ]:
        if lyrics_file.exists():
            with open(lyrics_file) as f:
                for line in f:
                    if line.startswith("=== ") and line.strip().endswith(" ==="):
                        name = line.strip().strip("= ").strip()
                        done.add(name.lower())
    return done

def wav_to_name(fname):
    """Convert WAV filename to a display name."""
    name = fname.replace('.wav', '')
    # Remove common prefixes
    for prefix in ['Biteki_びてき_', 'Biteki_']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace('_', ' ')
    return name

def main():
    print(f"=== TRANSCRIBE ALL REMAINING TRACKS ===", flush=True)
    print(f"Music dir: {MUSIC_DIR}", flush=True)
    print(f"Output: {OUTPUT_FILE}", flush=True)

    # Get all WAV files
    all_wavs = sorted([f for f in os.listdir(MUSIC_DIR) if f.endswith('.wav')])
    print(f"Total WAV files: {len(all_wavs)}", flush=True)

    # Get already transcribed
    done = get_already_transcribed()
    print(f"Already transcribed: {len(done)}", flush=True)

    # Filter to remaining
    remaining = []
    for f in all_wavs:
        name = wav_to_name(f).lower()
        # Check various forms against done set
        if name in done:
            continue
        # Also check without common suffixes
        name_clean = name.replace(' lyrics amv', '').replace(' lyrics hsr amv', '').replace(' official music video', '').replace(' official video', '').replace(' official audio', '').strip()
        if name_clean in done:
            continue
        remaining.append(f)

    print(f"Remaining to transcribe: {len(remaining)}", flush=True)

    if not remaining:
        print("Nothing to transcribe!", flush=True)
        return

    # Import server tools (this loads heavy models)
    from server import transcribe_lyrics

    all_lyrics = {}
    total = len(remaining)
    start_total = time.time()

    for i, fname in enumerate(remaining):
        path = str(MUSIC_DIR / fname)
        name = wav_to_name(fname)
        print(f"\n=== [{i+1}/{total}] Transcribing: {name} ===", flush=True)
        start = time.time()
        try:
            result = transcribe_lyrics(path)
            elapsed = time.time() - start
            print(f"Done in {elapsed:.0f}s", flush=True)
            data = json.loads(result)
            lyrics = data.get('lyrics', data.get('text', ''))
            all_lyrics[name] = lyrics
            preview = lyrics[:150].replace('\n', ' ')
            print(f"LYRICS: {preview}...", flush=True)
        except Exception as e:
            elapsed = time.time() - start
            print(f"FAILED after {elapsed:.0f}s: {e}", flush=True)
            all_lyrics[name] = f"ERROR: {str(e)}"

        # Save incrementally every 5 tracks
        if (i + 1) % 5 == 0 or (i + 1) == total:
            with open(OUTPUT_FILE, 'w') as f:
                for n, l in all_lyrics.items():
                    f.write(f"\n=== {n} ===\n")
                    f.write(l + "\n")
            print(f"[Saved {len(all_lyrics)} lyrics to {OUTPUT_FILE}]", flush=True)

    total_elapsed = time.time() - start_total
    print(f"\n{'='*60}", flush=True)
    print(f"ALL {total} TRACKS COMPLETE in {total_elapsed/60:.1f} minutes", flush=True)
    print(f"Lyrics saved to: {OUTPUT_FILE}", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == '__main__':
    main()
