#!/usr/bin/env python3
"""
Billboard chart discovery — downloads top charting songs into the library.
Uses the `billboard.py` package (no API key required).
"""
import sys
from pathlib import Path

from song_db import SongDB

CHARTS = ["hot-100", "r-b-hip-hop-songs", "rap-song"]
MAX_CHART_DOWNLOADS_PER_RUN = 10


def get_chart_songs(chart_name: str) -> list[dict]:
    """Return [{rank, title, artist}] for the current top songs on a Billboard chart."""
    try:
        import billboard
        chart = billboard.ChartData(chart_name)
        results = []
        for entry in chart:
            results.append({
                "rank": entry.rank,
                "title": entry.title,
                "artist": entry.artist,
            })
        return results
    except Exception as e:
        return []


def chart_discovery(db: SongDB, music_dir: Path, log=print) -> int:
    """
    Iterate CHARTS, fetch top songs, skip already-in-library titles,
    download new ones via yt-dlp.  Returns count of new songs added.
    """
    try:
        from discovery import search_youtube, download_track
    except ImportError:
        log("charts: discovery.py not available, skipping")
        return 0

    new_songs = 0
    seen_titles: set[str] = set()

    # Build a set of title fragments already on disk for quick dedup
    existing_stems = {wav.stem.lower() for wav in music_dir.glob("*.wav")}

    for chart_name in CHARTS:
        if new_songs >= MAX_CHART_DOWNLOADS_PER_RUN:
            break

        log(f"Charts: fetching '{chart_name}'")
        songs = get_chart_songs(chart_name)
        if not songs:
            log(f"Charts: no results for '{chart_name}' (billboard.py missing or network error)")
            continue

        for song in songs:
            if new_songs >= MAX_CHART_DOWNLOADS_PER_RUN:
                break

            title = song["title"]
            artist = song["artist"]
            key = f"{artist} - {title}".lower()

            if key in seen_titles:
                continue
            seen_titles.add(key)

            # Skip if a file with this title already exists in the library
            title_fragment = title.lower().replace(" ", "_")
            if any(title_fragment in stem for stem in existing_stems):
                log(f"Charts: already have '{title}' by {artist}, skipping")
                continue

            query = f"{artist} {title} official audio"
            log(f"Charts: searching YouTube for #{song['rank']} '{title}' by {artist}")

            tracks = search_youtube(query, max_results=3)
            if not tracks:
                continue

            track = tracks[0]
            vid_id = track["id"]
            if not vid_id:
                continue

            # Skip very short or very long tracks
            duration = track.get("duration", 0) or 0
            if duration > 0 and (duration < 60 or duration > 600):
                continue

            log(f"Charts: downloading '{track['title']}' by '{track['uploader']}'")
            out_path = download_track(vid_id, track["title"], track["uploader"], music_dir)

            if out_path:
                db.add_song(out_path)
                db.log_discovery(
                    query=f"billboard:{chart_name} #{song['rank']}",
                    source_artist="billboard",
                    source_song=f"{artist} - {title}",
                    downloaded_file=out_path,
                )
                existing_stems.add(Path(out_path).stem.lower())
                log(f"Charts: added {Path(out_path).name}")
                new_songs += 1

    return new_songs


if __name__ == "__main__":
    db_path = Path(__file__).parent / "agent.db"
    music_dir = Path.home() / "Documents" / "music" / "music data"
    db = SongDB(db_path)

    def log(msg):
        print(msg)

    # Print top 10 from each chart
    for chart_name in CHARTS:
        print(f"\n--- {chart_name} ---")
        songs = get_chart_songs(chart_name)
        if not songs:
            print("  (no data — is billboard.py installed?)")
        else:
            for s in songs[:10]:
                print(f"  #{s['rank']:3d}  {s['artist']} — {s['title']}")

    # Download 1 test song
    print("\n--- Test download (1 song) ---")
    MAX_CHART_DOWNLOADS_PER_RUN = 1
    count = chart_discovery(db, music_dir, log)
    print(f"Downloaded {count} new song(s)")
