# Phase 4 — Query Layer, Dashboard, Cross-Song Intelligence & Ableton Wiring

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the growing library of analysis docs into a queryable, searchable, browsable intelligence layer with a terminal dashboard and Ableton song-matching tool.

**Architecture:** `extractor.py` parses analysis MDs → structured rows + CLAP embeddings stored in a new `song_features` SQLite table. `query.py` CLI + four new MCP tools expose structural (SQL) and semantic (cosine similarity) search. `dashboard.py` uses `rich` to show agent progress and library stats.

**Tech Stack:** SQLite (existing), numpy, librosa, transformers/CLAP (existing), rich (new)

**Critical:** No real analysis docs exist yet (229 songs pending). Use the synthetic fixture in Task 2 for all MD-parsing tests.

---

### Task 1: Add `rich` dependency + `song_features` table + DB query methods

**Files:**
- Modify: `pyproject.toml`
- Modify: `song_db.py`

**Step 1: Add `rich` to pyproject.toml**

In `pyproject.toml`, add `"rich>=13.0.0",` after the `"spotipy>=2.23.0",` line.

**Step 2: Install dependencies**

Run: `uv sync`
Expected: resolves and installs rich without errors.

**Step 3: Verify rich imports**

Run: `uv run python -c "from rich.console import Console; Console().print('[green]rich OK[/green]')"`
Expected: prints "rich OK" in green.

**Step 4: Add `song_features` table to `song_db.py`**

In `SongDB._init_db()`, extend the `executescript` to add after the `discovery_log` table:

```python
                CREATE TABLE IF NOT EXISTS song_features (
                    song_id          INTEGER PRIMARY KEY REFERENCES songs(id),
                    bpm              REAL,
                    key              TEXT,
                    chords           TEXT,
                    genres           TEXT,
                    mood             TEXT,
                    energy           TEXT,
                    instruments      TEXT,
                    similar_artists  TEXT,
                    embedding        BLOB,
                    extracted_at     TEXT
                );
```

**Step 5: Add query methods to `SongDB`**

Append these methods to `SongDB` in `song_db.py`:

```python
def get_song_id(self, file_path: str) -> int | None:
    with self._conn() as conn:
        row = conn.execute(
            "SELECT id FROM songs WHERE file_path = ?", (str(file_path),)
        ).fetchone()
    return row["id"] if row else None

def upsert_features(self, song_id: int, features: dict):
    import json
    from datetime import datetime
    with self._conn() as conn:
        conn.execute(
            """INSERT INTO song_features
               (song_id, bpm, key, chords, genres, mood, energy, instruments,
                similar_artists, embedding, extracted_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(song_id) DO UPDATE SET
                 bpm=excluded.bpm, key=excluded.key, chords=excluded.chords,
                 genres=excluded.genres, mood=excluded.mood, energy=excluded.energy,
                 instruments=excluded.instruments, similar_artists=excluded.similar_artists,
                 embedding=excluded.embedding, extracted_at=excluded.extracted_at""",
            (
                song_id,
                features.get("bpm"),
                features.get("key"),
                json.dumps(features.get("chords") or []),
                json.dumps(features.get("genres") or []),
                features.get("mood"),
                features.get("energy"),
                json.dumps(features.get("instruments") or []),
                json.dumps(features.get("similar_artists") or []),
                features.get("embedding"),
                datetime.now().isoformat(),
            )
        )

def get_songs_without_features(self) -> list[dict]:
    """Return done songs that don't have features extracted yet."""
    with self._conn() as conn:
        rows = conn.execute(
            """SELECT s.id, s.file_path, s.document_path, s.artist, s.title
               FROM songs s
               LEFT JOIN song_features sf ON s.id = sf.song_id
               WHERE s.status = 'done' AND sf.song_id IS NULL"""
        ).fetchall()
    return [dict(r) for r in rows]

def query_songs(self, bpm_min=None, bpm_max=None, key=None, genre=None,
                mood=None, energy=None, limit=20) -> list[dict]:
    """Structured query over song_features JOIN songs."""
    import json
    clauses = ["sf.song_id IS NOT NULL"]
    params = []
    if bpm_min is not None:
        clauses.append("sf.bpm >= ?"); params.append(bpm_min)
    if bpm_max is not None:
        clauses.append("sf.bpm <= ?"); params.append(bpm_max)
    if key:
        clauses.append("LOWER(sf.key) LIKE ?"); params.append(f"%{key.lower()}%")
    if mood:
        clauses.append("LOWER(sf.mood) LIKE ?"); params.append(f"%{mood.lower()}%")
    if energy:
        clauses.append("LOWER(sf.energy) = ?"); params.append(energy.lower())
    if genre:
        clauses.append("LOWER(sf.genres) LIKE ?"); params.append(f"%{genre.lower()}%")
    where = " AND ".join(clauses)
    with self._conn() as conn:
        rows = conn.execute(
            f"""SELECT s.artist, s.title, s.file_path, s.document_path,
                       sf.bpm, sf.key, sf.mood, sf.energy, sf.genres, sf.similar_artists
                FROM songs s JOIN song_features sf ON s.id = sf.song_id
                WHERE {where}
                ORDER BY sf.bpm
                LIMIT ?""",
            params + [limit]
        ).fetchall()
    return [dict(r) for r in rows]

def get_all_embeddings(self) -> list[dict]:
    """Return all songs with stored CLAP embeddings."""
    with self._conn() as conn:
        rows = conn.execute(
            """SELECT s.id as song_id, s.file_path, s.artist, s.title, sf.embedding
               FROM songs s JOIN song_features sf ON s.id = sf.song_id
               WHERE sf.embedding IS NOT NULL"""
        ).fetchall()
    return [dict(r) for r in rows]

def features_stats(self) -> dict:
    """Return aggregate stats across all song_features rows."""
    with self._conn() as conn:
        bpm_row = conn.execute(
            "SELECT MIN(bpm), MAX(bpm), AVG(bpm) FROM song_features WHERE bpm IS NOT NULL"
        ).fetchone()
        key_rows = conn.execute(
            "SELECT key, COUNT(*) as cnt FROM song_features WHERE key IS NOT NULL GROUP BY key ORDER BY cnt DESC LIMIT 10"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) as cnt FROM song_features").fetchone()["cnt"]
    return {
        "total_with_features": total,
        "bpm_min": bpm_row[0],
        "bpm_max": bpm_row[1],
        "bpm_avg": round(bpm_row[2], 1) if bpm_row[2] else None,
        "top_keys": [{"key": r["key"], "count": r["cnt"]} for r in key_rows],
    }
```

**Step 6: Verify schema + methods**

Run: `uv run python -c "
from pathlib import Path
from song_db import SongDB
db = SongDB(Path('agent.db'))
db.upsert_features(999999, {'bpm': 140.0, 'key': 'C minor', 'chords': ['Am','F'], 'genres': ['trap'], 'mood': 'dark', 'energy': 'high', 'instruments': ['808'], 'similar_artists': ['Lil Baby']})
rows = db.query_songs(bpm_min=130, bpm_max=150, key='C minor')
print('query_songs OK, rows:', len(rows))
# cleanup
import sqlite3; conn = sqlite3.connect('agent.db'); conn.execute('DELETE FROM song_features WHERE song_id=999999'); conn.commit(); conn.close()
print('PASS')
"`
Expected: `query_songs OK, rows: 1` then `PASS`.

---

### Task 2: Create `extractor.py` — MD parser

**Files:**
- Create: `extractor.py`

**Step 1: Create the file with a synthetic test fixture at the bottom**

Create `/Users/josii/Desktop/audio-mcp-review/extractor.py`:

```python
#!/usr/bin/env python3
"""
Parses analysis markdown docs → extracts structured fields into song_features table.
Also computes CLAP audio embeddings for semantic similarity search.
"""
import re
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ─── MD Section Splitter ─────────────────────────────────────────────────────

_SECTION_RE = re.compile(r'^## (.+)$', re.MULTILINE)


def _get_section(text: str, header: str) -> str:
    """Return text content of a specific ## section, or empty string."""
    pattern = re.compile(
        r'^## ' + re.escape(header) + r'\s*\n(.*?)(?=^## |\Z)',
        re.MULTILINE | re.DOTALL
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


# ─── Field Extractors ────────────────────────────────────────────────────────

def _extract_bpm(text: str) -> float | None:
    m = re.search(r'(\d{2,3}(?:\.\d)?)\s*(?:BPM|bpm)', text)
    if m:
        return float(m.group(1))
    m = re.search(r'(?:BPM|bpm|Tempo)[:\s]+(\d{2,3}(?:\.\d)?)', text)
    return float(m.group(1)) if m else None


def _extract_key(text: str) -> str | None:
    m = re.search(
        r'(?:^|[\s,\-])((?:[A-G][#b♯♭]?\s*(?:major|minor|Major|Minor)))',
        text, re.MULTILINE
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r'(?:key|Key)[:\s]+([A-G][#b♯♭]?\s*(?:major|minor)?)',
        text
    )
    return m.group(1).strip() if m else None


def _extract_chords(text: str) -> list[str]:
    m = re.search(r'[Cc]hord(?:\s+progression)?[:\s]+([^\n]{4,80})', text)
    if not m:
        return []
    raw = m.group(1)
    parts = re.split(r'[\s\-–—,/|]+', raw)
    chord_re = re.compile(r'^[A-G][#b]?(?:m|maj|min|dim|aug|sus|add|\d)*$')
    return [p for p in parts if chord_re.match(p)][:8]


def _extract_instruments(text: str) -> list[str]:
    m = re.search(
        r'(?:instruments?|stems?|detected)[:\s]+([^\n]{4,200})',
        text, re.IGNORECASE
    )
    if not m:
        return []
    raw = m.group(1)
    items = [x.strip().lower() for x in re.split(r'[,;/]+', raw) if x.strip()]
    return [i for i in items if 2 < len(i) < 40][:10]


def _extract_mood_energy(section_text: str) -> tuple[str | None, str | None]:
    """Extract mood (first sentence) and energy (low/medium/high) from Emotional Character section."""
    if not section_text:
        return None, None
    sentences = re.split(r'(?<=[.!?])\s+', section_text)
    mood = sentences[0][:80].strip() if sentences else None

    energy_keywords = {
        "high": ["high energy", "intense", "aggressive", "explosive", "frantic", "frenetic", "hard-hitting"],
        "low": ["low energy", "calm", "mellow", "soft", "gentle", "peaceful", "slow", "minimal"],
        "medium": ["medium energy", "moderate", "balanced", "mid-tempo"],
    }
    text_lower = section_text.lower()
    for level, keywords in energy_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return mood, level

    return mood, "medium"


def _extract_similar_artists(section_text: str) -> list[str]:
    """Extract artist names from Related Artists section."""
    if not section_text:
        return []
    artists = []
    # Match bold: **Artist Name**
    artists += re.findall(r'\*\*([^*\n]{2,40})\*\*', section_text)
    # Match bullet list items: "- Artist Name —" or "- Artist Name:"
    for line in section_text.splitlines():
        line = line.strip().lstrip('-•*').strip()
        m = re.match(r'^([A-Z][^\n—:]{2,30}?)(?:\s*[—:–]|\s*$)', line)
        if m:
            name = m.group(1).strip()
            if name and name not in artists and len(name) > 2:
                artists.append(name)
    seen = set()
    result = []
    for a in artists:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            result.append(a)
    return result[:8]


_GENRE_KEYWORDS = {
    "trap": ["trap", "808", "hi-hat", "adlib"],
    "drill": ["drill", "sliding", "samples"],
    "r&b": ["r&b", "rnb", "soul", "smooth"],
    "hip-hop": ["hip-hop", "hip hop", "rap", "bars", "verse", "chorus"],
    "pop": ["pop", "catchy", "hook"],
    "electronic": ["electronic", "edm", "synth", "techno", "house"],
    "jazz": ["jazz", "swing", "bebop"],
    "rock": ["rock", "guitar", "distortion"],
}


def _infer_genres(full_text: str) -> list[str]:
    text_lower = full_text.lower()
    matched = []
    for genre, keywords in _GENRE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(genre)
    return matched[:4]


# ─── Main Parse Function ─────────────────────────────────────────────────────

def parse_analysis_doc(doc_path: Path) -> dict:
    """
    Parse an analysis markdown file and return a structured features dict.
    All fields are optional — missing data returns None / empty list.
    """
    text = doc_path.read_text(encoding="utf-8", errors="replace")

    snapshot = _get_section(text, "Production Snapshot")
    emotional = _get_section(text, "Emotional Character")
    sonic = _get_section(text, "Sonic Breakdown")
    related = _get_section(text, "Related Artists & Sounds")

    bpm = _extract_bpm(snapshot)
    key = _extract_key(snapshot)
    chords = _extract_chords(snapshot)
    instruments = _extract_instruments(snapshot) or _extract_instruments(sonic)
    mood, energy = _extract_mood_energy(emotional)
    similar_artists = _extract_similar_artists(related)
    genres = _infer_genres(text)

    return {
        "bpm": bpm,
        "key": key,
        "chords": chords,
        "instruments": instruments,
        "mood": mood,
        "energy": energy,
        "similar_artists": similar_artists,
        "genres": genres,
    }


# ─── CLAP Embedding ──────────────────────────────────────────────────────────

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_SAMPLE_RATE = 48000
_clap_model = None
_clap_processor = None


def _get_clap():
    global _clap_model, _clap_processor
    if _clap_model is None:
        from transformers import ClapModel, ClapProcessor
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        _clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device).eval()
    return _clap_model, _clap_processor


def compute_embedding(wav_path: str, max_duration: float = 60.0) -> bytes | None:
    """Compute normalized CLAP embedding for a WAV file. Returns float32 bytes."""
    try:
        import librosa
        import numpy as np
        import torch
        audio, _ = librosa.load(str(wav_path), sr=CLAP_SAMPLE_RATE,
                                duration=max_duration, mono=True)
        model, processor = _get_clap()
        device = next(model.parameters()).device
        inputs = processor(audio=audio, sampling_rate=CLAP_SAMPLE_RATE,
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embed = model.get_audio_features(**inputs)
            embed = embed / embed.norm(dim=-1, keepdim=True)
        return embed.cpu().numpy().astype(np.float32).tobytes()
    except Exception as e:
        log.warning(f"compute_embedding failed for {wav_path}: {e}")
        return None


def cosine_similarity(a: bytes, b: bytes) -> float:
    """Cosine similarity between two normalized float32 embedding byte strings."""
    import numpy as np
    va = np.frombuffer(a, dtype=np.float32)
    vb = np.frombuffer(b, dtype=np.float32)
    return float(np.dot(va, vb))


# ─── Extract + Store ─────────────────────────────────────────────────────────

def extract_song(db, song_id: int, doc_path: Path, wav_path: Path,
                 skip_embedding: bool = False, log_fn=print) -> dict | None:
    """
    Parse doc_path + compute embedding for wav_path.
    Upserts features into song_features table.
    Returns features dict or None on failure.
    """
    try:
        if not doc_path.exists():
            log_fn(f"extractor: doc not found: {doc_path}")
            return None
        features = parse_analysis_doc(doc_path)
        if not skip_embedding and wav_path and wav_path.exists():
            log_fn(f"extractor: computing embedding for {wav_path.name}")
            features["embedding"] = compute_embedding(str(wav_path))
        db.upsert_features(song_id, features)
        log_fn(f"extractor: stored features for song_id={song_id} "
               f"bpm={features.get('bpm')} key={features.get('key')!r}")
        return features
    except Exception as e:
        log_fn(f"extractor: ERROR for song_id={song_id}: {e}")
        return None


def backfill(db, analyses_dir: Path, music_dir: Path, log_fn=print,
             skip_embedding: bool = False) -> int:
    """
    Process all done songs that don't have features yet.
    Returns count of songs processed.
    """
    pending = db.get_songs_without_features()
    log_fn(f"extractor backfill: {len(pending)} songs to process")
    count = 0
    for row in pending:
        song_id = row["id"]
        doc_path = Path(row["document_path"]) if row["document_path"] else None
        wav_path = Path(row["file_path"]) if row["file_path"] else None
        if not doc_path:
            continue
        result = extract_song(db, song_id, doc_path, wav_path,
                               skip_embedding=skip_embedding, log_fn=log_fn)
        if result is not None:
            count += 1
    log_fn(f"extractor backfill: done — {count}/{len(pending)} extracted")
    return count


# ─── Standalone Tests ────────────────────────────────────────────────────────

_SYNTHETIC_DOC = """# Test Song — Test Artist

## Production Snapshot
BPM: 140, Key: C minor, Chord progression: Am - F - C - G
Duration: 3:12
Detected instruments: 808 bass, hi-hat, snare, piano, synthesizer

## Full Lyrics
[test lyrics]

## Lyrical Analysis
[test analysis]

## Sonic Breakdown
Heavy 808 bass with rolling hi-hats and punchy snare on beats 2 and 4.

## Emotional Character
Dark and aggressive energy throughout, building tension from verse to chorus. High energy track with an intense, brooding mood.

## Why This Works
[why it works]

## Popularity & Real-World Reception
[popularity]

## What A Producer/Songwriter Can Learn From This
[learnings]

## Related Artists & Sounds
- **Lil Baby** — similar melodic flow and trap production
- **Gunna** — comparable auto-tune usage
- Young Thug, Roddy Ricch
"""


def _run_self_tests():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(_SYNTHETIC_DOC)
        tmp = Path(f.name)

    features = parse_analysis_doc(tmp)
    tmp.unlink()

    assert features["bpm"] == 140.0, f"bpm: {features['bpm']}"
    assert features["key"] == "C minor", f"key: {features['key']}"
    assert "Am" in features["chords"], f"chords: {features['chords']}"
    assert "808 bass" in features["instruments"], f"instruments: {features['instruments']}"
    assert features["energy"] == "high", f"energy: {features['energy']}"
    assert any("Lil Baby" in a for a in features["similar_artists"]), \
        f"similar_artists: {features['similar_artists']}"
    assert "trap" in features["genres"], f"genres: {features['genres']}"
    print("All self-tests PASSED")
    print(f"  bpm={features['bpm']}, key={features['key']!r}, "
          f"chords={features['chords']}, energy={features['energy']!r}")
    print(f"  instruments={features['instruments']}")
    print(f"  similar_artists={features['similar_artists']}")
    print(f"  genres={features['genres']}")
    return features


if __name__ == "__main__":
    import sys
    from pathlib import Path as P
    from song_db import SongDB

    print("=== extractor.py self-tests ===")
    _run_self_tests()

    db_path = P(__file__).parent / "agent.db"
    db = SongDB(db_path)
    analyses_dir = P.home() / "Documents" / "music" / "analyses"
    music_dir = P.home() / "Documents" / "music" / "music data"

    skip_embed = "--no-embed" in sys.argv
    count = backfill(db, analyses_dir, music_dir, skip_embedding=skip_embed)
    stats = db.features_stats()
    print(f"\nLibrary stats after backfill:")
    print(f"  Songs with features: {stats['total_with_features']}")
    print(f"  BPM range: {stats['bpm_min']} – {stats['bpm_max']} (avg {stats['bpm_avg']})")
    print(f"  Top keys: {stats['top_keys'][:5]}")
```

**Step 2: Run the self-tests**

Run: `uv run python extractor.py --no-embed`
Expected: "All self-tests PASSED" followed by parsed field values. Since no songs are analyzed yet, backfill will report 0 songs.

---

### Task 3: Create `query.py` — CLI search tool

**Files:**
- Create: `query.py`

**Step 1: Create `query.py`**

```python
#!/usr/bin/env python3
"""
CLI for querying the music library.

Usage:
  uv run python query.py --bpm 130-150
  uv run python query.py --key "C minor" --genre trap
  uv run python query.py --mood dark --energy high --limit 20
  uv run python query.py --similar path/to/song.wav --n 5
  uv run python query.py --insights
"""
import argparse
import json
from pathlib import Path


def print_table(rows: list[dict]):
    if not rows:
        print("No results.")
        return
    headers = ["artist", "title", "bpm", "key", "energy", "mood"]
    col_w = {h: max(len(h), max((len(str(r.get(h) or "")) for r in rows), default=0)) for h in headers}
    sep = "  "
    header_line = sep.join(h.ljust(col_w[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print(sep.join(str(r.get(h) or "").ljust(col_w[h]) for h in headers))
    print(f"\n{len(rows)} result(s)")


def cmd_query(db, args):
    bpm_min = bpm_max = None
    if args.bpm:
        parts = args.bpm.split("-")
        if len(parts) == 2:
            bpm_min, bpm_max = float(parts[0]), float(parts[1])
        else:
            bpm_min = bpm_max = float(parts[0])
    rows = db.query_songs(
        bpm_min=bpm_min, bpm_max=bpm_max,
        key=args.key, genre=args.genre,
        mood=args.mood, energy=args.energy,
        limit=args.limit,
    )
    print_table(rows)


def cmd_similar(db, wav_path: str, n: int):
    from extractor import compute_embedding, cosine_similarity
    import numpy as np

    print(f"Computing embedding for: {wav_path}")
    query_embed = compute_embedding(wav_path, max_duration=30.0)
    if query_embed is None:
        print("ERROR: could not compute embedding.")
        return

    all_embeds = db.get_all_embeddings()
    if not all_embeds:
        print("No embeddings stored yet. Run: uv run python extractor.py")
        return

    scores = []
    for row in all_embeds:
        if row["embedding"]:
            sim = cosine_similarity(query_embed, row["embedding"])
            scores.append((sim, row))

    scores.sort(key=lambda x: x[0], reverse=True)
    print(f"\nTop {n} similar songs:")
    for sim, row in scores[:n]:
        print(f"  {sim:.3f}  {row.get('artist','?')} — {row.get('title','?')}")
        print(f"         {row.get('file_path','')}")


def cmd_insights(db):
    import json
    stats = db.features_stats()
    db_stats = db.stats()

    print("=== Library Insights ===\n")
    print(f"Songs analyzed: {db_stats.get('done', 0)}")
    print(f"Songs with features extracted: {stats['total_with_features']}")
    print(f"Pending: {db_stats.get('pending', 0)}  Errors: {db_stats.get('error', 0)}\n")

    if stats["bpm_avg"]:
        print(f"BPM range: {stats['bpm_min']} – {stats['bpm_max']}  (avg {stats['bpm_avg']})\n")

    if stats["top_keys"]:
        print("Top keys:")
        for k in stats["top_keys"]:
            bar = "█" * k["count"]
            print(f"  {k['key']:20s} {bar} {k['count']}")

    # Genre breakdown from JSON columns
    from collections import Counter
    import sqlite3
    conn = sqlite3.connect(str(Path(__file__).parent / "agent.db"))
    rows = conn.execute("SELECT genres FROM song_features WHERE genres IS NOT NULL").fetchall()
    conn.close()
    genre_counter: Counter = Counter()
    for (genres_json,) in rows:
        for g in json.loads(genres_json or "[]"):
            genre_counter[g] += 1
    if genre_counter:
        print("\nGenre breakdown:")
        for genre, cnt in genre_counter.most_common(8):
            bar = "█" * cnt
            print(f"  {genre:20s} {bar} {cnt}")

    # Top similar artists mentioned across all analyses
    artist_counter: Counter = Counter()
    conn = sqlite3.connect(str(Path(__file__).parent / "agent.db"))
    rows = conn.execute("SELECT similar_artists FROM song_features WHERE similar_artists IS NOT NULL").fetchall()
    conn.close()
    for (artists_json,) in rows:
        for a in json.loads(artists_json or "[]"):
            artist_counter[a.strip()] += 1
    if artist_counter:
        print("\nMost mentioned similar artists:")
        for artist, cnt in artist_counter.most_common(10):
            print(f"  {cnt:3d}x  {artist}")


def main():
    from song_db import SongDB
    db = SongDB(Path(__file__).parent / "agent.db")

    parser = argparse.ArgumentParser(description="Query the music intelligence library")
    parser.add_argument("--bpm", help="BPM range, e.g. 130-150 or 140")
    parser.add_argument("--key", help="Key, e.g. 'C minor'")
    parser.add_argument("--genre", help="Genre keyword, e.g. 'trap'")
    parser.add_argument("--mood", help="Mood keyword, e.g. 'dark'")
    parser.add_argument("--energy", choices=["low", "medium", "high"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--similar", metavar="WAV_PATH", help="Find songs similar to this WAV")
    parser.add_argument("--n", type=int, default=5, help="Number of similar songs to return")
    parser.add_argument("--insights", action="store_true", help="Show library-wide stats")
    args = parser.parse_args()

    if args.insights:
        cmd_insights(db)
    elif args.similar:
        cmd_similar(db, args.similar, args.n)
    else:
        cmd_query(db, args)


if __name__ == "__main__":
    main()
```

**Step 2: Verify it loads without errors**

Run: `uv run python query.py --insights`
Expected: Prints "Library Insights" panel. Zero results is fine (no songs analyzed yet), but must not crash.

**Step 3: Verify structural query works**

Run: `uv run python -c "
from pathlib import Path
from song_db import SongDB
db = SongDB(Path('agent.db'))
# Insert a synthetic row
db.upsert_features(999998, {'bpm': 142.0, 'key': 'A minor', 'genres': ['trap'], 'energy': 'high', 'mood': 'dark'})
import sqlite3; conn = sqlite3.connect('agent.db'); conn.execute('UPDATE song_features SET song_id=999998 WHERE song_id=999998'); conn.commit(); conn.close()
rows = db.query_songs(bpm_min=130, key='A minor')
print('rows:', len(rows))
# cleanup
import sqlite3; conn = sqlite3.connect('agent.db'); conn.execute('DELETE FROM song_features WHERE song_id=999998'); conn.commit(); conn.close()
"`
Expected: `rows: 1`

---

### Task 4: Add MCP tools to `server.py`

**Files:**
- Modify: `server.py` (append before `if __name__ == "__main__":`)

**Step 1: Append query MCP tools to `server.py`**

Add the following block just before the final `if __name__ == "__main__":` line in `server.py`:

```python
# ─── Library Query Tools ─────────────────────────────────────────────────────

def _get_agent_db():
    """Return SongDB instance pointing at the agent database."""
    import sys as _sys
    # agent.db lives alongside server.py
    _db_path = Path(__file__).parent / "agent.db"
    if not _db_path.exists():
        return None
    from song_db import SongDB
    return SongDB(_db_path)


@mcp.tool()
def query_library(
    bpm_min: float = None,
    bpm_max: float = None,
    key: str = "",
    genre: str = "",
    mood: str = "",
    energy: str = "",
    limit: int = 20,
) -> str:
    """
    Search the analyzed music library by BPM range, key, genre, mood, or energy.
    Returns matching songs with their analysis doc paths.

    Args:
        bpm_min: Minimum BPM (optional).
        bpm_max: Maximum BPM (optional).
        key: Key filter, e.g. "C minor" (optional, partial match).
        genre: Genre keyword, e.g. "trap" (optional).
        mood: Mood keyword, e.g. "dark" (optional).
        energy: "low", "medium", or "high" (optional).
        limit: Max results (default 20).
    """
    try:
        db = _get_agent_db()
        if db is None:
            return _tool_output({"error": "agent.db not found"}, "query_library")
        rows = db.query_songs(
            bpm_min=bpm_min or None,
            bpm_max=bpm_max or None,
            key=key or None,
            genre=genre or None,
            mood=mood or None,
            energy=energy or None,
            limit=limit,
        )
        return _tool_output({"results": rows, "count": len(rows)}, "query_library")
    except Exception as e:
        return _tool_output({"error": str(e)}, "query_library")


@mcp.tool()
def similar_songs(wav_path: str, n: int = 5) -> str:
    """
    Find songs in the library that sound similar to the given WAV file.
    Uses CLAP audio embeddings + cosine similarity.

    Args:
        wav_path: Path to the query WAV file.
        n: Number of similar songs to return (default 5).
    """
    try:
        from extractor import compute_embedding, cosine_similarity
        db = _get_agent_db()
        if db is None:
            return _tool_output({"error": "agent.db not found"}, "similar_songs")

        query_embed = compute_embedding(str(wav_path), max_duration=30.0)
        if query_embed is None:
            return _tool_output({"error": "could not compute embedding for query file"}, "similar_songs")

        all_embeds = db.get_all_embeddings()
        if not all_embeds:
            return _tool_output(
                {"error": "no embeddings stored yet — run extractor.py first"},
                "similar_songs"
            )

        scores = []
        for row in all_embeds:
            if row["embedding"]:
                sim = cosine_similarity(query_embed, row["embedding"])
                scores.append({
                    "similarity": round(sim, 4),
                    "artist": row.get("artist"),
                    "title": row.get("title"),
                    "file_path": row.get("file_path"),
                })
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return _tool_output({"results": scores[:n], "count": len(scores[:n])}, "similar_songs")
    except Exception as e:
        return _tool_output({"error": str(e)}, "similar_songs")


@mcp.tool()
def library_insights() -> str:
    """
    Return aggregate stats across the entire analyzed music library:
    BPM distribution, top keys, genre breakdown, most mentioned similar artists.
    """
    try:
        import json as _json
        from collections import Counter
        db = _get_agent_db()
        if db is None:
            return _tool_output({"error": "agent.db not found"}, "library_insights")

        stats = db.features_stats()
        db_stats = db.stats()

        # Genre breakdown
        genre_counter: Counter = Counter()
        artist_counter: Counter = Counter()
        for row in db.get_all_embeddings():
            pass  # just need file_path, handled below

        with db._conn() as conn:
            genre_rows = conn.execute(
                "SELECT genres FROM song_features WHERE genres IS NOT NULL"
            ).fetchall()
            artist_rows = conn.execute(
                "SELECT similar_artists FROM song_features WHERE similar_artists IS NOT NULL"
            ).fetchall()

        for (g,) in genre_rows:
            for genre in _json.loads(g or "[]"):
                genre_counter[genre] += 1
        for (a,) in artist_rows:
            for artist in _json.loads(a or "[]"):
                artist_counter[artist.strip()] += 1

        return _tool_output({
            "library": {
                "total_songs": sum(db_stats.values()),
                "analyzed": db_stats.get("done", 0),
                "pending": db_stats.get("pending", 0),
                "errors": db_stats.get("error", 0),
                "with_features": stats["total_with_features"],
            },
            "bpm": {
                "min": stats["bpm_min"],
                "max": stats["bpm_max"],
                "avg": stats["bpm_avg"],
            },
            "top_keys": stats["top_keys"],
            "top_genres": [{"genre": g, "count": c} for g, c in genre_counter.most_common(10)],
            "top_similar_artists": [{"artist": a, "count": c} for a, c in artist_counter.most_common(15)],
        }, "library_insights")
    except Exception as e:
        return _tool_output({"error": str(e)}, "library_insights")


@mcp.tool()
def get_library_suggestions(bpm: float, key: str = "", n: int = 10) -> str:
    """
    Suggest library songs that fit a given BPM and key — useful when working in Ableton.
    Returns songs within ±10 BPM of the target and matching key (if provided).

    Args:
        bpm: Target BPM (e.g. 140.0 from Ableton session).
        key: Target key (e.g. "C minor"). Optional.
        n: Max results (default 10).
    """
    try:
        db = _get_agent_db()
        if db is None:
            return _tool_output({"error": "agent.db not found"}, "get_library_suggestions")
        rows = db.query_songs(
            bpm_min=bpm - 10,
            bpm_max=bpm + 10,
            key=key or None,
            limit=n,
        )
        return _tool_output({
            "target_bpm": bpm,
            "target_key": key,
            "results": rows,
            "count": len(rows),
        }, "get_library_suggestions")
    except Exception as e:
        return _tool_output({"error": str(e)}, "get_library_suggestions")
```

**Step 2: Verify all four tools load**

Run: `uv run python -c "
from server import query_library, similar_songs, library_insights, get_library_suggestions
print('query_library:', query_library(bpm_min=130, bpm_max=150))
print('library_insights:', library_insights()[:200])
print('get_library_suggestions:', get_library_suggestions(140.0, 'C minor'))
print('All tools OK')
"`
Expected: All four print JSON output, no import errors.

---

### Task 5: Wire extractor into `agent.py`

**Files:**
- Modify: `agent.py`

**Step 1: Add extractor call after `mark_done` in `main_loop`**

In `agent.py`, find this block:

```python
        if success:
            db.mark_done(str(wav_path), str(doc_path))
            log(f"  DONE: {doc_path.name}")
            songs_since_discovery += 1
```

Replace with:

```python
        if success:
            db.mark_done(str(wav_path), str(doc_path))
            log(f"  DONE: {doc_path.name}")
            songs_since_discovery += 1
            # Extract structured features + embedding (non-blocking, errors only log)
            try:
                from extractor import extract_song as _extract
                song_id = db.get_song_id(str(wav_path))
                if song_id:
                    _extract(db, song_id, doc_path, wav_path, log_fn=log)
            except Exception as _e:
                log(f"  extractor warning: {_e}")
```

**Step 2: Same change for `--one` test mode**

In the `args.one` block, find:

```python
        success = run_claude_analysis(wav_path, doc_path, artist, title, log, spotify_data=spotify_data)
        if success:
            db.mark_done(str(wav_path), str(doc_path))
            log(f"SUCCESS: {doc_path}")
```

Replace with:

```python
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
```

**Step 3: Verify agent.py still parses**

Run: `uv run python -c "import ast; ast.parse(open('agent.py').read()); print('agent.py syntax OK')"`
Expected: `agent.py syntax OK`

---

### Task 6: Create `dashboard.py`

**Files:**
- Create: `dashboard.py`

**Step 1: Create `dashboard.py`**

```python
#!/usr/bin/env python3
"""
Terminal dashboard for the Music Intelligence Agent.

Usage:
  uv run python dashboard.py           # single snapshot
  uv run python dashboard.py --watch   # live refresh every 5s
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.live import Live
from rich.text import Text
from rich import box

DB_PATH = Path(__file__).parent / "agent.db"
LOG_PATH = Path(__file__).parent / "agent.log"
REFRESH_SECONDS = 5


def _read_db():
    import sqlite3
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    status_rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM songs GROUP BY status"
    ).fetchall()
    status = {r["status"]: r["cnt"] for r in status_rows}

    recent = conn.execute(
        """SELECT s.artist, s.title, s.analyzed_at, sf.bpm, sf.key, sf.energy
           FROM songs s
           LEFT JOIN song_features sf ON s.id = sf.song_id
           WHERE s.status = 'done'
           ORDER BY s.analyzed_at DESC LIMIT 10"""
    ).fetchall()

    discovery = conn.execute(
        """SELECT source_artist, COUNT(*) as cnt
           FROM discovery_log GROUP BY source_artist ORDER BY cnt DESC LIMIT 5"""
    ).fetchall()

    key_rows = conn.execute(
        "SELECT key, COUNT(*) as cnt FROM song_features WHERE key IS NOT NULL GROUP BY key ORDER BY cnt DESC LIMIT 8"
    ).fetchall()

    bpm_row = conn.execute(
        "SELECT MIN(bpm), MAX(bpm), AVG(bpm) FROM song_features WHERE bpm IS NOT NULL"
    ).fetchone()

    genre_rows = conn.execute(
        "SELECT genres FROM song_features WHERE genres IS NOT NULL"
    ).fetchall()

    conn.close()

    genre_counter: Counter = Counter()
    for (g,) in genre_rows:
        for genre in json.loads(g or "[]"):
            genre_counter[genre] += 1

    return {
        "status": status,
        "recent": [dict(r) for r in recent],
        "discovery": [dict(r) for r in discovery],
        "top_keys": [dict(r) for r in key_rows],
        "bpm": (bpm_row[0], bpm_row[1], round(bpm_row[2], 1) if bpm_row[2] else None),
        "top_genres": genre_counter.most_common(6),
    }


def _read_log_tail(n: int = 6) -> list[str]:
    if not LOG_PATH.exists():
        return ["(agent.log not found)"]
    with open(LOG_PATH) as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


def _build_display(data: dict) -> list:
    renderables = []

    # ── Library Stats ──
    s = data["status"]
    total = sum(s.values())
    done = s.get("done", 0)
    pct = f"{100*done//total}%" if total else "0%"
    stats_text = (
        f"[green]Done: {done}[/green]  "
        f"[yellow]Pending: {s.get('pending', 0)}[/yellow]  "
        f"[blue]Analyzing: {s.get('analyzing', 0)}[/blue]  "
        f"[red]Error: {s.get('error', 0)}[/red]  "
        f"[dim]{pct} complete[/dim]"
    )
    renderables.append(Panel(stats_text, title="[bold]Library Stats[/bold]", box=box.ROUNDED))

    # ── Agent Log ──
    log_lines = _read_log_tail()
    log_text = "\n".join(log_lines) or "(no log entries)"
    renderables.append(Panel(log_text, title="[bold]Agent Log (last 6 lines)[/bold]", box=box.ROUNDED))

    # ── Recent Analyses ──
    if data["recent"]:
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        table.add_column("Artist", max_width=20)
        table.add_column("Title", max_width=25)
        table.add_column("BPM", justify="right", max_width=6)
        table.add_column("Key", max_width=12)
        table.add_column("Energy", max_width=8)
        table.add_column("Analyzed At", max_width=20)
        for r in data["recent"]:
            table.add_row(
                str(r.get("artist") or "?")[:20],
                str(r.get("title") or "?")[:25],
                str(r.get("bpm") or ""),
                str(r.get("key") or ""),
                str(r.get("energy") or ""),
                str(r.get("analyzed_at") or "")[:19],
            )
        renderables.append(Panel(table, title="[bold]Recent Analyses[/bold]", box=box.ROUNDED))

    # ── Keys + BPM side by side ──
    side_panels = []

    if data["top_keys"]:
        key_table = Table(box=box.SIMPLE, show_header=False)
        key_table.add_column("Key")
        key_table.add_column("Bar")
        key_table.add_column("N", justify="right")
        max_cnt = max(r["cnt"] for r in data["top_keys"]) or 1
        for r in data["top_keys"]:
            bar = "█" * int(10 * r["cnt"] / max_cnt)
            key_table.add_row(r["key"] or "?", f"[cyan]{bar}[/cyan]", str(r["cnt"]))
        side_panels.append(Panel(key_table, title="[bold]Top Keys[/bold]", box=box.ROUNDED))

    bpm_min, bpm_max, bpm_avg = data["bpm"]
    if bpm_avg is not None:
        bpm_text = f"Min: {bpm_min}\nMax: {bpm_max}\nAvg: {bpm_avg}"
        side_panels.append(Panel(bpm_text, title="[bold]BPM Range[/bold]", box=box.ROUNDED))

    if data["top_genres"]:
        genre_table = Table(box=box.SIMPLE, show_header=False)
        genre_table.add_column("Genre")
        genre_table.add_column("N", justify="right")
        for genre, cnt in data["top_genres"]:
            genre_table.add_row(genre, str(cnt))
        side_panels.append(Panel(genre_table, title="[bold]Genres[/bold]", box=box.ROUNDED))

    if side_panels:
        renderables.append(Columns(side_panels))

    return renderables


def render_once(console: Console):
    data = _read_db()
    if data is None:
        console.print("[red]agent.db not found — is the agent running?[/red]")
        return
    for r in _build_display(data):
        console.print(r)


def render_watch(console: Console):
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            from rich.console import Group
            data = _read_db()
            if data is None:
                live.update(Panel("[red]agent.db not found[/red]"))
            else:
                renderables = _build_display(data)
                live.update(Group(*renderables))
            time.sleep(REFRESH_SECONDS)


def main():
    parser = argparse.ArgumentParser(description="Music Intelligence Agent Dashboard")
    parser.add_argument("--watch", action="store_true", help="Live refresh every 5s")
    args = parser.parse_args()

    console = Console()
    if args.watch:
        render_watch(console)
    else:
        render_once(console)


if __name__ == "__main__":
    main()
```

**Step 2: Verify dashboard renders**

Run: `uv run python dashboard.py`
Expected: Renders panels without errors. Stats will show 229 pending / 0 done since nothing is analyzed yet. Key/BPM/genre panels will be absent (no features). No crash.

---

### Task 7: Integration smoke test

**Step 1: Verify all new files have clean syntax**

Run: `uv run python -c "
import ast
for f in ['extractor.py', 'query.py', 'dashboard.py', 'song_db.py', 'agent.py']:
    ast.parse(open(f).read())
    print(f'OK: {f}')
"`
Expected: `OK: <filename>` for each file.

**Step 2: Run extractor self-tests**

Run: `uv run python extractor.py --no-embed`
Expected: "All self-tests PASSED"

**Step 3: Run query CLI**

Run: `uv run python query.py --insights`
Expected: Prints insights without crash. All counts will be 0 (nothing analyzed yet).

**Step 4: Run dashboard**

Run: `uv run python dashboard.py`
Expected: Renders without crash, shows library stats.

**Step 5: Verify all four MCP tools load**

Run: `uv run python -c "
from server import query_library, similar_songs, library_insights, get_library_suggestions
import json
r = json.loads(library_insights())
print('library_insights ok:', r.get('ok'))
r2 = json.loads(query_library(bpm_min=130))
print('query_library ok:', r2.get('ok'))
r3 = json.loads(get_library_suggestions(140.0))
print('get_library_suggestions ok:', r3.get('ok'))
print('All MCP tools OK')
"`
Expected: all three print `ok: True`, then "All MCP tools OK".

---

## Summary of new files and changes

| File | Action |
|------|--------|
| `pyproject.toml` | Add `rich>=13.0.0` |
| `song_db.py` | Add `song_features` table + 5 new methods |
| `extractor.py` | **New** — MD parser + CLAP embedding + backfill |
| `query.py` | **New** — CLI structural + semantic search |
| `dashboard.py` | **New** — rich terminal dashboard |
| `server.py` | Add 4 MCP tools: `query_library`, `similar_songs`, `library_insights`, `get_library_suggestions` |
| `agent.py` | Call `extract_song` after each `mark_done` |
