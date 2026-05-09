#!/usr/bin/env python3
"""Download all tracks from multiple YouTube playlists, skipping duplicates."""
import subprocess
import sys
import json
import os
import re
import time
from pathlib import Path

DOWNLOAD_DIR = Path.home() / "Documents" / "music" / "music data"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[^\w\s\-]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:200]

def get_existing_files():
    """Get set of existing wav filenames (lowercase for comparison)."""
    return {f.lower() for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.wav')}

def get_playlist_tracks(url):
    """Get list of (id, title, duration) from a playlist."""
    result = subprocess.run(
        [sys.executable, '-m', 'yt_dlp', '--flat-playlist', '--dump-json', url],
        capture_output=True, text=True, timeout=120
    )
    tracks = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        try:
            info = json.loads(line)
            tracks.append({
                'id': info.get('id', ''),
                'title': info.get('title', 'unknown'),
                'uploader': info.get('uploader', info.get('channel', '')),
                'duration': info.get('duration', 0),
                'url': info.get('url', info.get('webpage_url', f"https://youtube.com/watch?v={info.get('id', '')}"))
            })
        except json.JSONDecodeError:
            continue
    return tracks

def download_track(track, existing_files):
    """Download a single track. Returns (success, skipped, filename)."""
    title = track['title']
    uploader = track.get('uploader', '')

    # Skip deleted/private videos
    if '[Deleted video]' in title or '[Private video]' in title:
        return False, True, title

    # Build expected filename to check for duplicates
    if uploader:
        fname = sanitize_filename(f"{uploader}_{title}")
    else:
        fname = sanitize_filename(title)

    wav_name = f"{fname}.wav".lower()

    # Check for duplicate
    if wav_name in existing_files:
        return False, True, title

    # Also check partial matches (in case uploader differs)
    title_sanitized = sanitize_filename(title).lower()
    for existing in existing_files:
        if title_sanitized in existing:
            return False, True, title

    # Download
    video_url = f"https://youtube.com/watch?v={track['id']}"
    output_template = str(DOWNLOAD_DIR / f"{fname}.%(ext)s")

    dl_cmd = [
        sys.executable, '-m', 'yt_dlp',
        '--extract-audio',
        '--audio-format', 'wav',
        '--output', output_template,
        '--no-playlist',
        '--no-overwrites',
        video_url
    ]

    try:
        result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            existing_files.add(wav_name)  # Update set for future dup checks
            return True, False, title
        else:
            return False, False, f"{title} (ERROR: {result.stderr[:100]})"
    except subprocess.TimeoutExpired:
        return False, False, f"{title} (TIMEOUT)"
    except Exception as e:
        return False, False, f"{title} ({str(e)[:100]})"

def main():
    playlists = [
        ('Playlist 1', 'https://youtube.com/playlist?list=PLMMYqGVck0eeErlZwZU3y4k5q5noPq-Od'),
        ('Playlist 2', 'https://youtube.com/playlist?list=PLMMYqGVck0eczGkF1Q-x7eBST3QBJYmMW'),
        ('Playlist 3', 'https://youtube.com/playlist?list=PLMMYqGVck0ef2wBhFxw1RBJhL7B59UHVE'),
        ('Playlist 4', 'https://youtube.com/playlist?list=PLMMYqGVck0ed0i7lf1pKhb19kJo6p2Nmk'),
        ('Playlist 5', 'https://youtube.com/playlist?list=PLMMYqGVck0efhFBU11RX-iAvcuwRZeE5Z'),
        ('Playlist 6', 'https://youtube.com/playlist?list=PLMMYqGVck0ednNZ1A9Oc75vutGIs3edbD'),
    ]

    existing = get_existing_files()
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    seen_ids = set()  # Track IDs we've already processed (cross-playlist dedup)

    start_time = time.time()

    for pname, purl in playlists:
        print(f"\n{'='*60}", flush=True)
        print(f"Processing {pname}: {purl}", flush=True)
        print(f"{'='*60}", flush=True)

        tracks = get_playlist_tracks(purl)
        print(f"Found {len(tracks)} tracks", flush=True)

        pl_downloaded = 0
        pl_skipped = 0
        pl_failed = 0

        for i, track in enumerate(tracks):
            # Cross-playlist dedup by video ID
            if track['id'] in seen_ids:
                print(f"  [{i+1}/{len(tracks)}] SKIP (already in another playlist): {track['title']}", flush=True)
                pl_skipped += 1
                continue
            seen_ids.add(track['id'])

            success, skipped, name = download_track(track, existing)

            if skipped:
                print(f"  [{i+1}/{len(tracks)}] SKIP: {name}", flush=True)
                pl_skipped += 1
            elif success:
                print(f"  [{i+1}/{len(tracks)}] DOWNLOADED: {name}", flush=True)
                pl_downloaded += 1
            else:
                print(f"  [{i+1}/{len(tracks)}] FAILED: {name}", flush=True)
                pl_failed += 1

        print(f"\n{pname} summary: {pl_downloaded} downloaded, {pl_skipped} skipped, {pl_failed} failed", flush=True)
        total_downloaded += pl_downloaded
        total_skipped += pl_skipped
        total_failed += pl_failed

    elapsed = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"ALL PLAYLISTS COMPLETE in {elapsed/60:.1f} minutes", flush=True)
    print(f"Total: {total_downloaded} downloaded, {total_skipped} skipped, {total_failed} failed", flush=True)
    print(f"Total WAV files in library: {len([f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.wav')])}", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == '__main__':
    main()
