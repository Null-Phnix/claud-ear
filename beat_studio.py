#!/usr/bin/env python3
"""
Beat Studio — Autonomous beat analysis, MIDI generation, and lyric writing.

Usage:
    uv run python beat_studio.py <beat.wav>           # process one beat
    uv run python beat_studio.py --watch <folder>     # watch folder for new beats
    uv run python beat_studio.py --watch              # watches ~/Documents/music/beat_drops/

Output per beat:
    ~/Documents/music/beat_studio/{timestamp}_{name}/
        beat_spec.json      — BPM, key, chords, mood, CLAP world
        lyrics.md           — full song lyrics written for this beat
        production_notes.md — recording direction, what makes it distinct
        chords.mid          — chord progression MIDI
        melody.mid          — melody sketch (pentatonic, trap-spaced)
        drums.mid           — trap drum pattern
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from midiutil import MIDIFile

# ── Filename parsing (mirrors agent.py parse_artist_title) ────────────────────
_JUNK_SUFFIXES = re.compile(
    r"\s*[\-\|]?\s*(official\s*(music\s*)?video|lyric\s*video|lyrics?\s*(amv)?|"
    r"amv|music\s*video|official\s*audio|official\s*visualizer|visualizer|"
    r"shot\s*by\s*\w+|wshh\s*exclusive|exclusive|official\s*clip|"
    r"prod\.?\s+by\s+\w+|dir\.?\s+by\s+\w+)\s*.*$",
    re.IGNORECASE,
)
_TRAILING_JUNK  = re.compile(r"[\s\-\|]+$")
_TIMESTAMP_PFX  = re.compile(r"^\d{2}(?:_\d{2}){3}_[0-9a-f]{8}_", re.IGNORECASE)


def _parse_filename(stem: str) -> tuple[str, str]:
    """Return (artist, title) from a yt-dlp-style filename stem."""
    stem = _TIMESTAMP_PFX.sub("", stem)
    readable = stem.replace("_", " ")
    if " - " in readable:
        left, right = readable.split(" - ", 1)
        title = _JUNK_SUFFIXES.sub("", right)
        title = _TRAILING_JUNK.sub("", title).strip() or right.split()[0]
        words = left.split()
        n = len(words)
        if n >= 2 and n % 2 == 0 and words[:n // 2] == words[n // 2:]:
            words = words[:n // 2]
        artist = " ".join(words[1:]) if len(words) >= 3 else " ".join(words)
        return artist.strip() or "Unknown", title.strip()
    title = _JUNK_SUFFIXES.sub("", readable).strip() or stem
    return "Unknown", title.strip()


# ── Paths ──────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
STUDIO_DIR = Path.home() / "Documents" / "music" / "beat_studio"
DROP_DIR   = Path.home() / "Documents" / "music" / "beat_drops"
CLAUDE_CMD = "claude"

# ── MIDI constants (General MIDI drum map) ─────────────────────────────────────
MIDI_KICK         = 36
MIDI_SNARE        = 38
MIDI_HIHAT_CLOSED = 42
MIDI_HIHAT_OPEN   = 46
MIDI_CLAP         = 39

NOTE_TO_MIDI = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
    "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71,
}

CHORD_QUALITIES = {
    "m":    [0, 3, 7],
    "":     [0, 4, 7],
    "m7":   [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11],
    "7":    [0, 4, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

PENTATONIC = {
    "major": [0, 2, 4, 7, 9],
    "minor": [0, 3, 5, 7, 10],
}


# ── Server imports (lazy so startup is fast) ───────────────────────────────────

def _import_server():
    """Import server analysis functions. Run from the project directory."""
    sys.path.insert(0, str(HERE))
    import server as _s
    return _s


# ── Audio analysis ─────────────────────────────────────────────────────────────

def analyze_beat(wav_path: Path) -> dict:
    """Run all analysis tools directly and return merged spec dict."""
    s = _import_server()
    wav = str(wav_path)

    print("    analyze_audio...")
    raw_audio = json.loads(s.analyze_audio(wav, max_duration=120))

    print("    detect_chords...")
    try:
        raw_chords = json.loads(s.detect_chords(wav, max_duration=120))
    except Exception as e:
        print(f"      (skipped — {e.__class__.__name__}: {str(e)[:60]})")
        raw_chords = {}

    print("    get_song_structure...")
    try:
        raw_structure = json.loads(s.get_song_structure(wav, max_duration=120))
    except Exception as e:
        print(f"      (skipped — {e.__class__.__name__})")
        raw_structure = {}

    print("    classify_mood...")
    try:
        raw_mood = json.loads(s.classify_mood(wav, max_duration=30))
    except Exception as e:
        print(f"      (skipped — {e.__class__.__name__})")
        raw_mood = {}

    print("    similar_songs...")
    try:
        raw_similar = json.loads(s.similar_songs(wav, n=8))
    except Exception as e:
        print(f"      (skipped — {e.__class__.__name__})")
        raw_similar = {}

    # ── Parse signal features ──────────────────────────────────────────────────
    sig = raw_audio.get("signal_analysis", {})
    bpm = sig.get("tempo_bpm", 140.0)
    key = sig.get("estimated_key", "A minor")
    key_conf = sig.get("key_confidence", 0.5)

    # Split key into note + mode (e.g. "F# minor" → "F#", "minor")
    key_parts = key.split()
    key_note  = key_parts[0] if key_parts else "A"
    mode      = key_parts[1].lower() if len(key_parts) > 1 else "minor"

    # ── Parse chords ───────────────────────────────────────────────────────────
    chord_list = []
    chords_data = raw_chords if isinstance(raw_chords, dict) else {}
    for item in chords_data.get("chords", [])[:4]:
        label = item.get("chord") or item.get("label") or ""
        if label and label.lower() not in ("n", "n/a", "unknown", ""):
            chord_list.append(label)
    if not chord_list:
        # Fallback: build simple progression from key/mode
        r = key_note
        chord_list = ([f"{r}m", f"{r}m7", f"{r}sus4", f"{r}m"]
                      if mode == "minor" else
                      [r, f"{r}maj7", f"{r}sus2", r])

    # ── Parse mood ─────────────────────────────────────────────────────────────
    mood_list = raw_mood.get("mood", [])
    primary_mood   = mood_list[0]["label"] if mood_list else "unknown"
    secondary_mood = mood_list[1]["label"] if len(mood_list) > 1 else ""

    # ── Parse CLAP neighbors ───────────────────────────────────────────────────
    neighbors = raw_similar.get("results", [])
    similar_artists = []
    for n in neighbors:
        fp = n.get("file_path", "")
        artist, _title = _parse_filename(Path(fp).stem)
        if artist and artist != "Unknown" and artist not in similar_artists:
            similar_artists.append(artist)
        if len(similar_artists) >= 5:
            break

    # ── Parse structure ────────────────────────────────────────────────────────
    raw_sections = raw_structure.get("sections", [])
    sections = []
    for sec in raw_sections:
        label = sec.get("label") or sec.get("section") or sec.get("type") or ""
        if label and label.strip():
            sections.append(label.strip())
    if not sections:
        sections = ["intro", "verse", "hook", "verse", "hook", "outro"]

    return {
        "bpm":               round(bpm, 1),
        "key":               key_note,
        "mode":              mode,
        "key_confidence":    round(key_conf, 3),
        "chord_progression": chord_list[:4],
        "energy":            round(sig.get("loudness", {}).get("rms_mean", 0.3), 3),
        "mood":              primary_mood,
        "mood_secondary":    secondary_mood,
        "sections":          sections,
        "clap_world":        _infer_world(neighbors),
        "top_similar_artists": similar_artists,
        "spectral_centroid_hz": round(sig.get("spectral", {}).get("centroid_hz", 0), 1),
        "_raw_neighbors":    [n.get("file_path", "") for n in neighbors[:5]],
    }


def _infer_world(neighbors: list) -> str:
    """Rough CLAP world label from neighbor filenames."""
    names = " ".join(n.get("file_path", "") for n in neighbors).lower()
    if "sewerperson" in names or "funeral" in names or "hiraeth" in names:
        return "World 1 — Sadcloud/Digicore"
    if "fivio" in names or "bizzy_banks" in names or "sheff_g" in names:
        return "World 2 — NY/Brooklyn Drill"
    if "don_toliver" in names or "youngboy" in names or "astrokidjay" in names:
        return "World 3 — Houston/Melodic Trap"
    if "bad_bunny" in names or "j_balvin" in names:
        return "World 6 — Latin/Reggaeton"
    if "tems" in names or "dave" in names or "central_cee" in names:
        return "World 8 — UK/Afrobeats"
    return "World 5 — Cross-World"


# ── MIDI generation ────────────────────────────────────────────────────────────

def _parse_chord(chord_str: str) -> tuple[int, list[int]]:
    m = re.match(r"^([A-G][#b]?)(.*)", chord_str.strip())
    if not m:
        return 60, [0, 3, 7]
    root_name, quality = m.group(1), m.group(2).strip()
    quality = {"min": "m", "maj": "", "minor": "m", "major": "",
               "min7": "m7", "Maj7": "maj7"}.get(quality, quality)
    return NOTE_TO_MIDI.get(root_name, 60), CHORD_QUALITIES.get(quality, [0, 3, 7])


def generate_chord_midi(chords: list, bpm: float, path: Path, bars_per_chord: int = 2) -> bool:
    midi = MIDIFile(1)
    midi.addTempo(0, 0, bpm)
    beats = bars_per_chord * 4
    t = 0
    for chord_str in chords:
        root, intervals = _parse_chord(chord_str)
        for interval in intervals:
            midi.addNote(0, 0, root + interval - 12, t, beats, 80)
        t += beats
    with open(path, "wb") as f:
        midi.writeFile(f)
    return True


def generate_melody_midi(key: str, mode: str, bpm: float, path: Path, num_bars: int = 8) -> bool:
    import random
    random.seed(42)
    midi = MIDIFile(1)
    midi.addTempo(0, 0, bpm)
    root  = NOTE_TO_MIDI.get(key, 69)
    penta = [root + i for i in PENTATONIC.get(mode, PENTATONIC["minor"])]
    notes = penta + [n + 12 for n in penta]
    t = 0.0
    while t < num_bars * 4:
        midi.addNote(0, 0, random.choice(notes[2:8]), t,
                     random.choice([0.5, 1.0, 1.5, 2.0]),
                     random.randint(60, 90))
        t += random.choice([0.5, 1.0, 1.5]) + random.choice([0.5, 1.0, 1.5, 2.0])
    with open(path, "wb") as f:
        midi.writeFile(f)
    return True


def generate_drums_midi(bpm: float, path: Path, num_bars: int = 8) -> bool:
    midi = MIDIFile(1)
    midi.addTempo(0, 0, bpm)
    step = 0.25
    for bar in range(num_bars):
        o = bar * 4
        for i in range(0, 16, 2):
            midi.addNote(0, 9, MIDI_HIHAT_CLOSED, o + i * step, step,
                         70 if i % 4 == 0 else 50)
        midi.addNote(0, 9, MIDI_KICK, o + 0,    step * 2, 100)
        midi.addNote(0, 9, MIDI_KICK, o + 2.75, step * 2, 90)
        if bar % 2 == 0:
            midi.addNote(0, 9, MIDI_KICK, o + 3.5, step, 80)
        midi.addNote(0, 9, MIDI_SNARE, o + 1, step, 95)
        midi.addNote(0, 9, MIDI_CLAP,  o + 1, step, 85)
        midi.addNote(0, 9, MIDI_SNARE, o + 3, step, 95)
        midi.addNote(0, 9, MIDI_CLAP,  o + 3, step, 85)
        if bar % 4 == 3:
            midi.addNote(0, 9, MIDI_HIHAT_OPEN, o + 3.5, step, 70)
    with open(path, "wb") as f:
        midi.writeFile(f)
    return True


# ── Claude lyric generation ────────────────────────────────────────────────────

def _clean_env() -> dict:
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    return env


def build_lyrics_prompt(spec: dict, wav_path: Path, output_dir: Path) -> str:
    lyrics_path = output_dir / "lyrics.md"
    notes_path  = output_dir / "production_notes.md"

    chords_str   = " → ".join(spec["chord_progression"])
    similar_str  = ", ".join(spec["top_similar_artists"]) or "unknown"
    sections_str = " / ".join(spec["sections"])

    return f"""You are a lyricist and music producer writing for KingJosii.

BEAT ANALYSIS:
- BPM: {spec["bpm"]}
- Key: {spec["key"]} {spec["mode"]} (confidence: {spec["key_confidence"]})
- Chord progression: {chords_str}
- Energy: {spec["energy"]}
- Primary mood: {spec["mood"]}
- Secondary mood: {spec["mood_secondary"]}
- CLAP world: {spec["clap_world"]}
- Closest sonic neighbors: {similar_str}
- Sections: {sections_str}

━━━ TASK 1: WRITE LYRICS ━━━
Write full song lyrics to: {lyrics_path}

Structure: [Intro], [Verse 1], [Hook], [Verse 2], [Hook], [Bridge], [Verse 3], [Outro]

Rules:
- Hook: 4–8 bars, one central image or phrase, highly repeatable
- Syllable density must match {spec["bpm"]} BPM — {
    "dense, rapid-fire bars" if spec["bpm"] > 130 else
    "mid-tempo flow with room to breathe" if spec["bpm"] > 100 else
    "slow, spacious delivery — let bars hang"
}
- Mood is {spec["mood"]} — lean into that fully, don't fight the beat
- Style: KingJosii — trap/hip-hop, first-person, authentic, no clichés
- Every line earns its place. No filler.
- Nearest sonic reference: {similar_str} — understand what those artists do emotionally, then make it yours

━━━ TASK 2: WRITE PRODUCTION NOTES ━━━
Write production notes to: {notes_path}

Include:
1. What makes this beat distinctive (key, mood, CLAP world, spectral character at {spec["spectral_centroid_hz"]} Hz centroid)
2. Recording direction: energy, mic distance, delivery style, where to place ad-libs
3. What NOT to do on this beat
4. Two production tweaks that would make this beat more distinctively KingJosii
5. Three song title options based on the lyrical themes you wrote

Use the Write tool for both files. Complete both tasks without stopping."""


def generate_lyrics(spec: dict, wav_path: Path, output_dir: Path) -> bool:
    prompt = build_lyrics_prompt(spec, wav_path, output_dir)
    print("  Generating lyrics + production notes...")
    try:
        result = subprocess.run(
            [CLAUDE_CMD, "-p", prompt, "--allowedTools", "Write"],
            capture_output=True, text=True, timeout=300,
            env=_clean_env(), cwd=str(HERE),
        )
        if result.returncode != 0:
            print(f"  Claude error ({result.returncode}): {result.stderr[:300]}")
            return False
        lyrics_path = output_dir / "lyrics.md"
        return lyrics_path.exists()
    except subprocess.TimeoutExpired:
        print("  Claude timed out")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


# ── Ableton integration ────────────────────────────────────────────────────────

def try_ableton(bpm: float) -> bool:
    prompt = (
        f"Check if Ableton is connected using check_ableton_connection. "
        f"If connected: call set_tempo with tempo={bpm}, then create_midi_track with name='Chords', "
        f"then create_midi_track with name='Melody', then create_midi_track with name='Drums', "
        f"then respond ABLETON_SUCCESS. If not connected respond ABLETON_OFFLINE."
    )
    try:
        result = subprocess.run(
            [CLAUDE_CMD, "-p", prompt,
             "--allowedTools", "check_ableton_connection,set_tempo,create_midi_track"],
            capture_output=True, text=True, timeout=60,
            env=_clean_env(), cwd=str(HERE),
        )
        if "ABLETON_SUCCESS" in result.stdout:
            print(f"  Dropped into Ableton @ {bpm} BPM")
            return True
    except Exception:
        pass
    print("  Ableton offline — drag MIDI files into Ableton when ready")
    return False


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_beat(wav_path: Path):
    wav_path = Path(wav_path).resolve()
    if not wav_path.exists():
        print(f"File not found: {wav_path}")
        return

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    beat_name  = re.sub(r"[^\w\-]", "_", wav_path.stem)[:40]
    output_dir = STUDIO_DIR / f"{timestamp}_{beat_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'━'*60}")
    print(f"  BEAT STUDIO: {wav_path.name}")
    print(f"  Output:      {output_dir}")
    print(f"{'━'*60}")

    # Step 1: Analyze
    print("\n[1/4] Analyzing beat...")
    try:
        spec = analyze_beat(wav_path)
    except Exception as e:
        print(f"  Analysis failed: {e}")
        return

    spec_path = output_dir / "beat_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"  {spec['key']} {spec['mode']}, {spec['bpm']} BPM, {spec['mood']}")
    print(f"  Chords: {' → '.join(spec['chord_progression'])}")
    print(f"  World:  {spec['clap_world']}")

    # Step 2: Generate MIDI
    print("\n[2/4] Generating MIDI...")
    generate_chord_midi(spec["chord_progression"], spec["bpm"], output_dir / "chords.mid")
    generate_melody_midi(spec["key"], spec["mode"], spec["bpm"], output_dir / "melody.mid")
    generate_drums_midi(spec["bpm"], output_dir / "drums.mid")
    print(f"  chords.mid, melody.mid, drums.mid — all written")

    # Step 3: Write lyrics
    print("\n[3/4] Writing lyrics...")
    generate_lyrics(spec, wav_path, output_dir)
    lyrics_ok = (output_dir / "lyrics.md").exists()
    print(f"  lyrics.md — {'written' if lyrics_ok else 'FAILED'}")

    # Step 4: Ableton
    print("\n[4/4] Checking Ableton...")
    try_ableton(spec["bpm"])

    # Summary
    print(f"\n{'━'*60}")
    print(f"  DONE: {output_dir.name}")
    for f in sorted(output_dir.iterdir()):
        if not f.name.startswith("."):
            print(f"    {f.name:<35} {f.stat().st_size:>8,} bytes")
    print(f"\n  Drop the .mid files into Ableton: {output_dir}")
    print(f"{'━'*60}\n")


def watch_folder(folder: Path):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\n{'━'*60}")
    print(f"  BEAT STUDIO — Watch Mode")
    print(f"  Drop WAV files into: {folder}")
    print(f"  Ctrl+C to stop")
    print(f"{'━'*60}\n")
    seen = set(folder.glob("*.wav"))
    while True:
        try:
            current = set(folder.glob("*.wav"))
            for wav in sorted(current - seen):
                print(f"New beat detected: {wav.name}")
                time.sleep(1)
                process_beat(wav)
            seen = current
            time.sleep(3)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


def already_processed(wav_path: Path) -> bool:
    """Check if this beat already has a studio output with lyrics."""
    stem = re.sub(r"[^\w\-]", "_", wav_path.stem)[:40]
    for d in STUDIO_DIR.glob(f"*_{stem}"):
        if (d / "lyrics.md").exists() and (d / "beat_spec.json").exists():
            return True
    return False


def process_library(music_dir: Path, limit: int = 0, skip_done: bool = True):
    """
    Run beat_studio over all WAVs in music_dir.
    Generates lyrics + production notes + MIDI for every track.
    Skips already-processed tracks by default.
    """
    music_dir = Path(music_dir).resolve()
    wavs = sorted(music_dir.glob("*.wav"))
    if not wavs:
        print(f"No WAVs found in {music_dir}")
        return

    total = len(wavs)
    if limit:
        wavs = wavs[:limit]

    print(f"\n{'━'*60}")
    print(f"  BEAT STUDIO — Library Mode")
    print(f"  Directory: {music_dir}")
    print(f"  Total tracks: {total} | Processing: {len(wavs)}")
    print(f"{'━'*60}\n")

    done = skipped = failed = 0
    for i, wav in enumerate(wavs, 1):
        if skip_done and already_processed(wav):
            print(f"[{i}/{len(wavs)}] SKIP (already done): {wav.name[:60]}")
            skipped += 1
            continue

        print(f"[{i}/{len(wavs)}] {wav.name[:60]}")
        try:
            process_beat(wav)
            done += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'━'*60}")
    print(f"  Library run complete")
    print(f"  Processed: {done} | Skipped: {skipped} | Failed: {failed}")
    print(f"  Output: {STUDIO_DIR}")
    print(f"{'━'*60}\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "--watch":
        folder = Path(sys.argv[2]) if len(sys.argv) > 2 else DROP_DIR
        watch_folder(folder)

    elif cmd == "--library":
        # Process the main music library
        music_dir = (Path(sys.argv[2]) if len(sys.argv) > 2
                     else Path.home() / "Documents" / "music" / "music data")
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        process_library(music_dir, limit=limit)

    elif cmd == "--creations":
        # Process just the music creations folder
        creations = Path.home() / "Documents" / "music" / "music creations"
        wavs = sorted(creations.rglob("*.wav"))
        print(f"Processing {len(wavs)} creation beats...")
        for wav in wavs:
            process_beat(wav)

    else:
        process_beat(Path(cmd))


if __name__ == "__main__":
    main()
