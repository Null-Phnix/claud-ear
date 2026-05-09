#!/usr/bin/env python3
"""
Autonomous Music Intelligence Agent
Continuously analyzes songs in the library using `claude -p` subprocess calls.
"""
import subprocess
import sys
import time
import logging
import re
from pathlib import Path

import power
from song_db import SongDB

MUSIC_DIR = Path.home() / "Documents" / "music" / "music data"
ANALYSES_DIR = Path.home() / "Documents" / "music" / "analyses"
DB_PATH = Path(__file__).parent / "agent.db"
LOG_PATH = Path(__file__).parent / "agent.log"
CLAUDE_CMD = "claude"

# Suffixes to strip from YouTube video titles
_JUNK_SUFFIXES = re.compile(
    r"\s*[\-\|]?\s*(official\s*(music\s*)?video|lyric\s*video|lyrics?\s*(amv)?|"
    r"amv|music\s*video|official\s*audio|shot\s*by\s*\w+|"
    r"wshh\s*exclusive|exclusive|official\s*clip)\s*.*$",
    re.IGNORECASE
)
# Strip trailing dashes/pipes/spaces
_TRAILING_JUNK = re.compile(r"[\s\-\|]+$")

_VALID_FNAME = re.compile(r'[^\w\s\-\.]')


def sanitize_path_part(s: str) -> str:
    """Make a string safe for use as a file/dir name."""
    s = _VALID_FNAME.sub('', s)
    s = re.sub(r'\s+', '_', s.strip())
    return s[:80] or "unknown"


_TIMESTAMP_PREFIX = re.compile(r'^\d{2}(?:_\d{2}){3}_[0-9a-f]{8}_', re.IGNORECASE)


def parse_artist_title(stem: str) -> tuple[str, str]:
    """
    Parse '{uploader}_{artist}_-_{title}' filename stem into (artist, title).
    Falls back gracefully for files that don't follow the pattern.
    """
    # Strip yt-dlp timestamp/hash prefix: YY_HH_MM_SS_8hexchars_
    stem = _TIMESTAMP_PREFIX.sub('', stem)
    # Replace underscores with spaces for parsing
    readable = stem.replace('_', ' ')

    if ' - ' in readable:
        left, right = readable.split(' - ', 1)
        # Clean right side (title): strip junk suffixes, then trailing punctuation
        title = _JUNK_SUFFIXES.sub('', right)
        title = _TRAILING_JUNK.sub('', title).strip()
        if not title:
            title = right.split()[0] if right.split() else stem

        # Clean left side (artist): often "{uploader} {artist}"
        words = left.split()
        # Detect repeated name pattern: "Lil Durk Lil Durk" or "22Gz 22Gz" → deduplicate
        n = len(words)
        if n >= 2 and n % 2 == 0:
            half = words[:n // 2]
            if half == words[n // 2:]:
                words = half
        # Skip first word if 3+ words remain (likely the YouTube channel/uploader prefix)
        if len(words) >= 3:
            artist = ' '.join(words[1:])
        else:
            artist = ' '.join(words)
        artist = artist.strip() or "Unknown"
    else:
        # No ' - ' separator: use whole stem as title, artist unknown
        title = _JUNK_SUFFIXES.sub('', readable).strip() or stem
        artist = "Unknown"

    # Limit lengths
    artist = artist[:60]
    title = title[:100]
    return artist, title


def build_analysis_prompt(wav_path: Path, doc_path: Path, artist: str, title: str,
                          spotify_data: dict = None, retry_count: int = 0,
                          quality_issues: list = None) -> str:
    """Build the full self-contained prompt for claude -p."""
    spotify_block = ""
    if spotify_data:
        lines = [f"SPOTIFY DATA (pre-fetched):"]
        if spotify_data.get("popularity") is not None:
            lines.append(f"  popularity={spotify_data['popularity']}/100")
        if spotify_data.get("energy") is not None:
            lines.append(f"  energy={spotify_data['energy']}")
        if spotify_data.get("tempo") is not None:
            lines.append(f"  tempo={spotify_data['tempo']}bpm")
        if spotify_data.get("danceability") is not None:
            lines.append(f"  danceability={spotify_data['danceability']}")
        if spotify_data.get("valence") is not None:
            lines.append(f"  valence={spotify_data['valence']}")
        if spotify_data.get("album"):
            lines.append(f"  album={spotify_data['album']!r}")
        if spotify_data.get("release_date"):
            lines.append(f"  release_date={spotify_data['release_date']}")
        if spotify_data.get("spotify_url"):
            lines.append(f"  spotify_url={spotify_data['spotify_url']}")
        spotify_block = "\n" + "\n".join(lines) + "\n"

    retry_block = ""
    if retry_count > 0 and quality_issues:
        issues_str = "; ".join(quality_issues)
        retry_block = (
            f"\nRETRY ATTEMPT {retry_count} — The previous analysis was incomplete.\n"
            f"Issues found: {issues_str}\n"
            f"Use the listed tools individually. "
            f"Ensure EVERY section contains real, detailed data — "
            f"no placeholders, no empty sections.\n"
        )

    # Select the primary analysis step based on retry_count
    if retry_count == 0:
        analysis_step = (
            "1. Call the deep_listen tool on the song file (max_duration=300). "
            "This gives you tempo, key, chords, structure, lyrics, stems, and instruments."
        )
        lyrics_step = (
            "2. If lyrics from deep_listen are incomplete or missing, "
            "call transcribe_lyrics separately on the same file."
        )
        raw_data_placeholder = "[paste the full JSON output from deep_listen here]"
    elif retry_count == 1:
        analysis_step = (
            "1. Call analyze_audio on the song file for tempo, key, and spectral features. "
            "Then call detect_chords and get_song_structure separately for chord and structure data."
        )
        lyrics_step = (
            "2. Call transcribe_lyrics on the song file to get the full lyric transcription."
        )
        raw_data_placeholder = "[paste the full JSON outputs from analyze_audio, detect_chords, and get_song_structure here]"
    else:
        analysis_step = (
            "1. Call analyze_stems on the song file to get detailed per-stem analysis "
            "(vocals, drums, bass, other). This gives more accurate instrument and content data."
        )
        lyrics_step = (
            "2. Call transcribe_lyrics on the song file to get the full lyric transcription."
        )
        raw_data_placeholder = "[paste the full JSON output from analyze_stems here]"

    return f"""You are a music intelligence analyst. Complete ALL steps below autonomously without asking questions. Write the final document using the Write tool.
{retry_block}
SONG FILE: {wav_path}
SAVE DOCUMENT TO: {doc_path}
ARTIST (from filename): {artist}
TITLE (from filename): {title}{spotify_block}

STEPS — complete every step in order:

{analysis_step}

{lyrics_step}

3. Call WebSearch for "{artist} {title} streams views YouTube" to find popularity data (view counts, chart positions, reception).

4. Call WebSearch for "{artist} discography genre" for artist context and background.

5. Using ALL the data gathered, write a complete analysis document to {doc_path} using the Write tool. The document must cover:

# {title} — {artist}

## Production Snapshot
BPM, key, chord progression, song structure (intro/verse/chorus/bridge/outro with timestamps), duration, detected stems and instruments.

## Full Lyrics
Complete transcription as heard. Mark uncertain lines with [?].

## Lyrical Analysis
Themes and meaning, key lines and their significance, literary devices (metaphor, imagery, repetition), emotional arc of the narrative.

## Sonic Breakdown
Beat construction and drum pattern (kick placement, snare/clap, hi-hat patterns), bass line character, melodic elements, vocal production, mixing observations (reverb, delay, compression style), overall sound design.

## Emotional Character
Mood and energy arc throughout the song, what feelings it creates and the sonic reasons why, how it evolves from start to finish.

## Why This Works
What specific production and lyrical choices make this track effective. Concrete examples from the data.

## Popularity & Real-World Reception
Use any pre-fetched Spotify data above to anchor this section. Include Spotify popularity score, audio features, and streaming/view counts found via web search. Critical reception or cultural impact, whether it was a hit and why or why not.

## What A Producer/Songwriter Can Learn From This
At least 5 concrete, actionable takeaways: chord choices, rhythmic tricks, structural decisions, lyrical techniques, production elements worth studying.

## Related Artists & Sounds
5-8 artists with similar sonic DNA. Be specific about what makes them similar. These are used for music discovery.

## Raw Analysis Data
```json
{raw_data_placeholder}
```

IMPORTANT: You MUST write the document to {doc_path} using the Write tool. Do not stop until the file is written."""


def run_claude_analysis(wav_path: Path, doc_path: Path, artist: str, title: str, log,
                        spotify_data: dict = None, retry_count: int = 0,
                        quality_issues: list = None) -> bool:
    """
    Run `claude -p <prompt>` to analyze the song.
    Returns True if doc_path exists after the call.
    Timeout: 10 minutes per song.
    """
    prompt = build_analysis_prompt(
        wav_path, doc_path, artist, title,
        spotify_data=spotify_data,
        retry_count=retry_count,
        quality_issues=quality_issues,
    )

    # Escalate tool set on retries for better coverage
    if retry_count == 0:
        allowed_tools = "deep_listen,transcribe_lyrics,WebSearch,Write,Read"
    elif retry_count == 1:
        allowed_tools = (
            "analyze_audio,transcribe_lyrics,detect_chords,"
            "get_song_structure,WebSearch,Write,Read"
        )
    else:
        allowed_tools = (
            "analyze_stems,transcribe_lyrics,detect_chords,WebSearch,Write,Read"
        )

    cmd = [
        CLAUDE_CMD,
        "-p", prompt,
        "--allowedTools", allowed_tools,
    ]

    log(f"  Running: claude -p [prompt] for {wav_path.name}")

    try:
        result = subprocess.run(
            cmd,
            timeout=600,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent)
        )
        if result.returncode != 0:
            log(f"  claude exited with code {result.returncode}")
            if result.stderr:
                log(f"  stderr: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after 600s for {wav_path.name}")
        return False
    except Exception as e:
        log(f"  ERROR running claude: {e}")
        return False

    return doc_path.exists()


def setup_logging() -> callable:
    """Set up file + stdout logging. Returns a log() function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger("agent")

    def log(msg: str):
        logger.info(msg)

    return log


def main_loop():
    log = setup_logging()
    log("=== Music Intelligence Agent starting ===")

    db = SongDB(DB_PATH)
    scanned = db.scan_library(MUSIC_DIR)
    log(f"Library scan: {scanned} WAV files. Stats: {db.stats()}")

    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)

    # Init Spotify once at startup (graceful skip if not configured)
    sp = None
    try:
        from discovery import init_spotify
        sp = init_spotify()
        if sp:
            log("Spotify: client initialized")
        else:
            log("Spotify: not configured (set SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET to enable)")
    except ImportError:
        pass

    songs_since_discovery = 0

    # Promote old errors to needs_retry so they get a second chance
    db.promote_errors_to_retry()
    log("Promoted any previous error songs to needs_retry")

    while True:
        pending = db.get_pending_songs()

        if not pending:
            log("No pending songs. Sleeping 5 min, then rescanning...")
            time.sleep(300)
            db.scan_library(MUSIC_DIR)
            continue

        wav_path = Path(pending[0])

        if not wav_path.exists():
            log(f"File not found, skipping: {wav_path}")
            db.mark_error(str(wav_path), "file not found")
            continue

        artist, title = parse_artist_title(wav_path.stem)
        artist_dir = ANALYSES_DIR / sanitize_path_part(artist)
        doc_path = artist_dir / f"{sanitize_path_part(title)}.md"
        artist_dir.mkdir(parents=True, exist_ok=True)

        # Fetch Spotify track data before analysis (used to anchor Popularity section)
        spotify_data = None
        if sp:
            try:
                from discovery import get_spotify_track_data
                spotify_data = get_spotify_track_data(artist, title, sp)
                if spotify_data:
                    log(f"  Spotify: popularity={spotify_data.get('popularity')}, "
                        f"energy={spotify_data.get('energy')}, tempo={spotify_data.get('tempo')}bpm")
            except ImportError:
                pass

        # Fetch retry info before marking as analyzing
        retry_info = db.get_song_retry_info(str(wav_path))
        retry_count = retry_info["retry_count"]
        quality_issues = retry_info["quality_issues"]
        # For needs_retry songs, increment counter and update tool selection
        if quality_issues:
            db.increment_retry_count(str(wav_path))
            retry_count += 1
            log(f"  Retry attempt {retry_count}/3 — prior issues: {quality_issues}")

        db.mark_analyzing(str(wav_path))
        log(f"Analyzing [{db.stats()}]: {wav_path.name}")
        log(f"  Artist: {artist!r}  Title: {title!r}")
        log(f"  Output: {doc_path}")

        success = run_claude_analysis(
            wav_path, doc_path, artist, title, log,
            spotify_data=spotify_data,
            retry_count=retry_count,
            quality_issues=quality_issues,
        )

        if success:
            db.mark_done(str(wav_path), str(doc_path))
            log(f"  DONE: {doc_path.name}")
            songs_since_discovery += 1
            # Extract structured features + embedding
            features = None
            try:
                from extractor import extract_song as _extract
                song_id = db.get_song_id(str(wav_path))
                if song_id:
                    features = _extract(db, song_id, doc_path, wav_path, log_fn=log)
            except Exception as _e:
                log(f"  extractor warning: {_e}")
            # Quality scoring
            try:
                from quality import score_doc
                result = score_doc(doc_path, features)
                log(f"  Quality: {result['score']}/100 ({result['quality']})"
                    + (f" — {', '.join(result['issues'])}" if result["issues"] else ""))
                current_retry = db.get_song_retry_info(str(wav_path))["retry_count"]
                if result["quality"] == "good":
                    db.update_quality(str(wav_path), result["score"], result["issues"])
                elif current_retry < 3:
                    db.mark_needs_retry(str(wav_path), result["score"], result["issues"])
                    log(f"  Queued for retry ({current_retry + 1}/3)")
                else:
                    db.mark_abandoned(str(wav_path), result["score"], result["issues"])
                    log(f"  Abandoned after {current_retry} retries")
            except Exception as _e:
                log(f"  quality scoring warning: {_e}")
        else:
            db.mark_error(str(wav_path), "document not created")
            log(f"  ERROR: document not created for {wav_path.name}")

        # Discovery cycle every 10 songs
        if songs_since_discovery >= 10:
            try:
                from discovery import discover_and_download
                new = discover_and_download(db, MUSIC_DIR, log, sp=sp)
                log(f"Discovery cycle (YouTube+SoundCloud): {new} new songs added")
            except ImportError:
                log("discovery.py not available yet, skipping discovery cycle")

            try:
                from charts import chart_discovery
                new_chart = chart_discovery(db, MUSIC_DIR, log)
                log(f"Discovery cycle (Billboard charts): {new_chart} new songs added")
            except ImportError:
                log("charts.py not available yet, skipping chart discovery")

            songs_since_discovery = 0

        power.sleep_between_songs()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Music Intelligence Agent")
    parser.add_argument("--one", action="store_true", help="Analyze one song and exit (for testing)")
    args = parser.parse_args()

    if args.one:
        # Single-song test mode
        log = setup_logging()
        db = SongDB(DB_PATH)
        db.scan_library(MUSIC_DIR)
        pending = db.get_pending_songs()
        if not pending:
            print("No pending songs found.")
            sys.exit(0)

        wav_path = Path(pending[0])
        artist, title = parse_artist_title(wav_path.stem)
        ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
        artist_dir = ANALYSES_DIR / sanitize_path_part(artist)
        doc_path = artist_dir / f"{sanitize_path_part(title)}.md"
        artist_dir.mkdir(parents=True, exist_ok=True)

        sp = None
        spotify_data = None
        try:
            from discovery import init_spotify, get_spotify_track_data
            sp = init_spotify()
            if sp:
                log("Spotify: client initialized")
                spotify_data = get_spotify_track_data(artist, title, sp)
                if spotify_data:
                    log(f"Spotify data: {spotify_data}")
        except ImportError:
            pass

        db.mark_analyzing(str(wav_path))
        log(f"Test mode: analyzing {wav_path.name}")

        success = run_claude_analysis(wav_path, doc_path, artist, title, log, spotify_data=spotify_data)
        if success:
            db.mark_done(str(wav_path), str(doc_path))
            log(f"SUCCESS: {doc_path}")
            try:
                from extractor import extract_song as _extract
                song_id = db.get_song_id(str(wav_path))
                if song_id:
                    _extract(db, song_id, doc_path, wav_path, log_fn=log)
            except Exception as _e:
                log(f"extractor warning: {_e}")
        else:
            db.mark_error(str(wav_path), "document not created")
            log(f"FAILED: document not created")
    else:
        main_loop()
