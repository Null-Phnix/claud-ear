#!/usr/bin/env python3
"""
Autonomous Music Intelligence Agent
Continuously analyzes songs in the library using the MCP audio tools.

Supports any LLM backend via llm_backend (Ollama by default, configurable).
"""

import os
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

# Suffixes to strip from YouTube video titles
_JUNK_SUFFIXES = re.compile(
    r"\s*[\-\|]?\s*(official\s*(music\s*)?video|lyric\s*video|lyrics?\s*(amv)?|"
    r"amv|music\s*video|official\s*audio|shot\s*by\s*\w+|"
    r"wshh\s*exclusive|exclusive|official\s*clip)\s*.*$",
    re.IGNORECASE
)
_TRAILING_JUNK = re.compile(r"[\s\-\|]+$")

_VALID_FNAME = re.compile(r'[^\w\s\-\.]')


def sanitize_path_part(s: str) -> str:
    """Make a string safe for use as a file/dir name."""
    s = _VALID_FNAME.sub('', s)
    s = re.sub(r'\s+', '_', s.strip())
    return s[:80] or "unknown"


_TIMESTAMP_PREFIX = re.compile(r'^\d{2}(?:_\d{2}){3}_[0-9a-f]{8}_', re.IGNORECASE)


def parse_artist_title(stem: str) -> tuple[str, str]:
    """Parse '{uploader}_{artist}_-_{title}' filename stem into (artist, title)."""
    stem = _TIMESTAMP_PREFIX.sub('', stem)
    readable = stem.replace('_', ' ')

    if ' - ' in readable:
        left, right = readable.split(' - ', 1)
        title = _JUNK_SUFFIXES.sub('', right)
        title = _TRAILING_JUNK.sub('', title).strip()
        if not title:
            title = right.split()[0] if right.split() else stem

        words = left.split()
        n = len(words)
        if n >= 2 and n % 2 == 0:
            half = words[:n // 2]
            if half == words[n // 2:]:
                words = half
        if len(words) >= 3:
            artist = ' '.join(words[1:])
        else:
            artist = ' '.join(words)
        artist = artist.strip() or "Unknown"
    else:
        title = _JUNK_SUFFIXES.sub('', readable).strip() or stem
        artist = "Unknown"

    return artist[:60], title[:100]


def run_audio_analysis(wav_path: Path, doc_path: Path, artist: str, title: str,
                        log_fn) -> bool:
    """
    Analyze a song using the configured LLM backend.
    Uses deep_listen MCP tool to get full analysis, then writes a formatted doc.

    This calls the current agent's own MCP tools through whatever LLM backend is
    configured (Hermes Agent, Claude Code, etc.).
    """
    from llm_backend import call_llm, get_model

    log_fn(f"  Analyzing with {get_model()} via llm_backend...")
    log_fn(f"  File: {wav_path}")
    log_fn(f"  Output: {doc_path}")

    # Build analysis document using the LLM
    prompt = f"""You are a music intelligence analyst. Write a complete, detailed
analysis document for the song below. Be specific, concrete, and data-rich.
No placeholders — every section must contain real content.

SONG FILE: {wav_path}
ARTIST: {artist}
TITLE: {title}

Write the document with these sections:

# {title} — {artist}

## Production Snapshot
BPM, key, chord progression, song structure (intro/verse/chorus/bridge/outro
with timestamps), duration, detected stems and instruments.

## Full Lyrics
Complete transcription as heard. Mark uncertain lines with [?].

## Lyrical Analysis
Themes and meaning, key lines and their significance, literary devices
(metaphor, imagery, repetition), emotional arc of the narrative.

## Sonic Breakdown
Beat construction and drum pattern (kick placement, snare/clap, hi-hat patterns),
bass line character, melodic elements, vocal production, mixing observations
(reverb, delay, compression style), overall sound design.

## Emotional Character
Mood and energy arc throughout the song, what feelings it creates and the sonic
reasons why, how it evolves from start to finish.

## Why This Works
What specific production and lyrical choices make this track effective.
Concrete examples from the data.

## Popularity & Real-World Reception
Streaming/view counts, critical reception, cultural impact. Whether it was a hit
and why or why not.

## What A Producer/Songwriter Can Learn From This
At least 5 concrete, actionable takeaways: chord choices, rhythmic tricks,
structural decisions, lyrical techniques, production elements worth studying.

## Related Artists & Sounds
5-8 artists with similar sonic DNA. Be specific about what makes them similar.

IMPORTANT: Write the complete document. No shortcuts, no placeholders."""

    try:
        response = call_llm(prompt, timeout=600)

        # Write the analysis document
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(response)
        log_fn(f"  Analysis written to {doc_path}")
        return True

    except Exception as e:
        log_fn(f"  ERROR during analysis: {e}")
        return False


def setup_logging() -> callable:
    """Set up file + stdout logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger("agent")

    def log_fn(msg: str):
        logger.info(msg)

    return log_fn


def main_loop():
    log = setup_logging()
    log("=== Music Intelligence Agent starting ===")

    # Verify LLM backend connectivity
    try:
        from llm_backend import check_connection, get_model, get_provider
        if check_connection():
            log(f"LLM backend: {get_provider()} — {get_model()} — connected ✓")
        else:
            log(f"WARNING: LLM backend unreachable — agent will fail on analysis")
    except ImportError:
        log("WARNING: llm_backend not found")

    db = SongDB(DB_PATH)
    scanned = db.scan_library(MUSIC_DIR)
    log(f"Library scan: {scanned} WAV files. Stats: {db.stats()}")

    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)

    # Init Spotify (graceful skip if not configured)
    sp = None
    try:
        from discovery import init_spotify
        sp = init_spotify()
        if sp:
            log("Spotify: client initialized")
        else:
            log("Spotify: not configured (set SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET)")
    except ImportError:
        pass

    songs_since_discovery = 0
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

        db.mark_analyzing(str(wav_path))
        log(f"Analyzing [{db.stats()}]: {wav_path.name}")
        log(f"  Artist: {artist!r}  Title: {title!r}")

        success = run_audio_analysis(wav_path, doc_path, artist, title, log)

        if success:
            db.mark_done(str(wav_path), str(doc_path))
            log(f"  DONE: {doc_path.name}")
            songs_since_discovery += 1

            # Quality scoring
            try:
                from quality import score_doc
                result = score_doc(doc_path, None)
                log(f"  Quality: {result['score']}/100 ({result['quality']})"
                    + (f" — {', '.join(result['issues'])}" if result["issues"] else ""))
            except Exception as e:
                log(f"  quality scoring warning: {e}")
        else:
            db.mark_error(str(wav_path), "analysis failed")
            log(f"  ERROR: analysis failed for {wav_path.name}")

        # Discovery cycle every 10 songs
        if songs_since_discovery >= 10:
            try:
                from discovery import discover_and_download
                new = discover_and_download(db, MUSIC_DIR, log, sp=sp)
                log(f"Discovery cycle (YouTube+SoundCloud): {new} new songs added")
            except ImportError:
                log("discovery.py not available, skipping discovery")
            except Exception as e:
                log(f"Discovery error: {e}")

            try:
                from charts import chart_discovery
                new_chart = chart_discovery(db, MUSIC_DIR, log)
                log(f"Discovery cycle (Billboard charts): {new_chart} new songs added")
            except ImportError:
                log("charts.py not available, skipping chart discovery")
            except Exception as e:
                log(f"Chart discovery error: {e}")

            songs_since_discovery = 0

        power.sleep_between_songs()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Music Intelligence Agent")
    parser.add_argument("--one", action="store_true",
                        help="Analyze one song and exit")
    args = parser.parse_args()

    if args.one:
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

        db.mark_analyzing(str(wav_path))
        log(f"Test mode: analyzing {wav_path.name}")

        success = run_audio_analysis(wav_path, doc_path, artist, title, log)
        if success:
            db.mark_done(str(wav_path), str(doc_path))
            log(f"SUCCESS: {doc_path}")
        else:
            db.mark_error(str(wav_path), "analysis failed")
            log("FAILED: analysis failed")
    else:
        main_loop()
