#!/usr/bin/env python3
"""
Music discovery via yt-dlp YouTube/SoundCloud search + optional Spotify related artists.
Called every 10 songs analyzed to expand the library with related music.
"""
import subprocess
import sys
import json
import os
import re
from pathlib import Path

from song_db import SongDB

QUERIES_PER_ARTIST = 2
SONGS_PER_QUERY = 1
MAX_DISCOVER_PER_CYCLE = 5

_VALID_FNAME = re.compile(r'[^\w\s\-\.]')


def sanitize_filename(name: str) -> str:
    name = _VALID_FNAME.sub('', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:200]


def init_spotify():
    """
    Load Spotify credentials from environment or .env file.
    Returns a spotipy.Spotify client, or None if credentials are not configured.
    """
    # Try loading .env manually (avoids requiring python-dotenv)
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                os.environ.setdefault(key.strip(), val.strip())

    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        return None

    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        auth = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        )
        return spotipy.Spotify(auth_manager=auth)
    except ImportError:
        return None
    except Exception:
        return None


def get_spotify_related(artist: str, sp=None) -> list[str]:
    """
    Return a list of related artist names via Spotify.
    Returns [] if sp is None or the artist cannot be found.
    """
    if sp is None or not artist or artist == "Unknown":
        return []
    try:
        results = sp.search(q=f"artist:{artist}", type="artist", limit=1)
        items = results.get("artists", {}).get("items", [])
        if not items:
            return []
        artist_id = items[0]["id"]
        related = sp.artist_related_artists(artist_id)
        return [a["name"] for a in related.get("artists", [])[:5]]
    except Exception:
        return []


def get_spotify_track_data(artist: str, title: str, sp=None) -> dict | None:
    """
    Return a dict with popularity metrics for a specific track.
    Returns None if sp is None or track not found.
    """
    if sp is None:
        return None
    try:
        results = sp.search(q=f"track:{title} artist:{artist}", type="track", limit=1)
        items = results.get("tracks", {}).get("items", [])
        if not items:
            return None
        track = items[0]
        track_id = track["id"]
        features = sp.audio_features([track_id])
        feat = features[0] if features else {}
        return {
            "popularity": track.get("popularity"),
            "energy": round(feat.get("energy", 0), 2) if feat else None,
            "tempo": round(feat.get("tempo", 0), 1) if feat else None,
            "danceability": round(feat.get("danceability", 0), 2) if feat else None,
            "valence": round(feat.get("valence", 0), 2) if feat else None,
            "album": track.get("album", {}).get("name"),
            "release_date": track.get("album", {}).get("release_date"),
            "spotify_url": track.get("external_urls", {}).get("spotify"),
        }
    except Exception:
        return None


def generate_queries(artist: str, analyzed_songs: list[str] = None, related_artists: list[str] = None) -> list[str]:
    """Generate YouTube/SoundCloud search queries for discovering related music."""
    queries = []
    if not artist or artist == "Unknown":
        return queries

    # Direct artist search
    queries.append(f"{artist} music")

    # Related sound search
    queries.append(f"songs like {artist} underground")

    # Add queries for Spotify-related artists
    if related_artists:
        for related in related_artists[:2]:
            queries.append(f"{related} music")

    return queries[:QUERIES_PER_ARTIST + (2 if related_artists else 0)]


def search_youtube(query: str, max_results: int = 3) -> list[dict]:
    """Search YouTube for videos matching query. Returns list of {id, title, uploader}."""
    return _yt_search(f"ytsearch{max_results}:{query}", max_results)


def search_soundcloud(query: str, max_results: int = 3) -> list[dict]:
    """Search SoundCloud for tracks matching query. Returns list of {id, title, uploader}."""
    return _yt_search(f"scsearch{max_results}:{query}", max_results)


def _yt_search(search_string: str, max_results: int) -> list[dict]:
    """Internal helper: run yt-dlp with a search string, return parsed track list."""
    try:
        result = subprocess.run(
            [
                sys.executable, '-m', 'yt_dlp',
                search_string,
                '--flat-playlist',
                '--dump-json',
                '--no-warnings',
                '--quiet',
            ],
            capture_output=True, text=True, timeout=30
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
                    'uploader': info.get('uploader', info.get('channel', 'unknown')),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'url': info.get('url', info.get('webpage_url', '')),
                })
            except json.JSONDecodeError:
                continue
        return tracks
    except Exception:
        return []


def download_track(video_id: str, title: str, uploader: str, music_dir: Path, url: str = None) -> str | None:
    """Download a single track as WAV. Returns the output file path or None on failure."""
    if uploader and uploader != 'unknown':
        fname = sanitize_filename(f"{uploader}_{title}")
    else:
        fname = sanitize_filename(title)

    output_template = str(music_dir / f"{fname}.%(ext)s")

    # Determine the source URL (YouTube or SoundCloud)
    if url:
        video_url = url
    else:
        video_url = f"https://youtube.com/watch?v={video_id}"

    # Check if file already exists
    if (music_dir / f"{fname}.wav").exists():
        return None

    # Check partial match
    title_san = sanitize_filename(title).lower()
    for existing in music_dir.glob("*.wav"):
        if title_san and title_san in existing.stem.lower():
            return None

    try:
        result = subprocess.run(
            [
                sys.executable, '-m', 'yt_dlp',
                '--extract-audio',
                '--audio-format', 'wav',
                '--output', output_template,
                '--no-playlist',
                '--no-overwrites',
                '--quiet',
                '--no-warnings',
                video_url,
            ],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            out_path = music_dir / f"{fname}.wav"
            if out_path.exists():
                return str(out_path)
        return None
    except Exception:
        return None


def discover_and_download(db: SongDB, music_dir: Path, log=print, sp=None) -> int:
    """
    Find and download related music based on analyzed artists.
    Searches both YouTube and SoundCloud.
    If sp (Spotify client) is provided, uses related artists to expand queries.
    Returns the count of new songs added to the database.
    """
    artists = db.get_analyzed_artists()
    if not artists:
        log("Discovery: no analyzed artists yet, skipping")
        return 0

    log(f"Discovery: running for {len(artists)} artists")

    new_songs = 0
    seen_ids: set[str] = set()

    for artist in artists:
        if new_songs >= MAX_DISCOVER_PER_CYCLE:
            break

        # Get Spotify related artists to expand the query pool
        related = get_spotify_related(artist, sp)
        if related:
            log(f"Discovery: Spotify related artists for {artist!r}: {related}")

        queries = generate_queries(artist, related_artists=related)

        for query in queries:
            if new_songs >= MAX_DISCOVER_PER_CYCLE:
                break

            # Search both YouTube and SoundCloud
            for source, search_fn in [("youtube", search_youtube), ("soundcloud", search_soundcloud)]:
                if new_songs >= MAX_DISCOVER_PER_CYCLE:
                    break

                log(f"Discovery [{source}]: searching '{query}'")
                tracks = search_fn(query, max_results=SONGS_PER_QUERY + 2)

                for track in tracks[:SONGS_PER_QUERY]:
                    if new_songs >= MAX_DISCOVER_PER_CYCLE:
                        break

                    vid_id = track['id']
                    if not vid_id or vid_id in seen_ids:
                        continue
                    seen_ids.add(vid_id)

                    # Skip very short or very long tracks (not songs)
                    duration = track.get('duration', 0) or 0
                    if duration > 0 and (duration < 60 or duration > 600):
                        continue

                    log(f"Discovery: downloading '{track['title']}' by '{track['uploader']}' [{source}]")
                    out_path = download_track(
                        vid_id, track['title'], track['uploader'], music_dir,
                        url=track.get('url') if source == 'soundcloud' else None
                    )

                    if out_path:
                        db.add_song(out_path)
                        db.log_discovery(
                            query=f"{source}:{query}",
                            source_artist=artist,
                            source_song=track['title'],
                            downloaded_file=out_path
                        )
                        log(f"Discovery: added {Path(out_path).name}")
                        new_songs += 1

    return new_songs


if __name__ == "__main__":
    from pathlib import Path as P
    db_path = P(__file__).parent / "agent.db"
    music_dir = P.home() / "Documents" / "music" / "music data"
    db = SongDB(db_path)

    sp = init_spotify()
    if sp:
        print("Spotify: client initialized")
    else:
        print("Spotify: not configured (set SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET to enable)")

    def log(msg):
        print(msg)

    count = discover_and_download(db, music_dir, log, sp=sp)
    print(f"Discovered {count} new songs")
