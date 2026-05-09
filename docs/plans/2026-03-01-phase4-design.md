# Phase 4 — Query Layer, Dashboard, Cross-Song Intelligence & Ableton Wiring
**Date:** 2026-03-01
**Status:** Approved

## Overview

Phase 4 turns the growing library of analysis documents into a queryable, browsable intelligence layer. Four capabilities are added in one phase, sequenced by dependency:

1. **Data extraction** — parse analysis MDs → structured SQLite rows + CLAP embeddings
2. **Query layer** — CLI + MCP tools to search/filter the library structurally and semantically
3. **Dashboard** — terminal UI to monitor the agent and browse analyses
4. **Ableton wiring** — MCP tool to surface library matches for the current Ableton session

---

## Architecture

```
analysis .md files (on disk)
        │
   extractor.py  ←── called by agent.py after each mark_done()
        │
   song_features table (SQLite)
   ┌──────────────────────────────┐
   │ bpm, key, chords, genres,    │
   │ mood, energy, instruments,   │
   │ similar_artists, embedding   │
   └──────────────────────────────┘
        │                │
   query.py CLI     server.py MCP tools
   (structural +    (query_library,
    semantic)        similar_songs,
                     library_insights,
                     get_library_suggestions)
        │
   dashboard.py  ←── reads songs + discovery_log + agent.log
```

---

## 1. Data Layer

### New SQLite table: `song_features`

Added via `SongDB._init_db()` migration:

```sql
CREATE TABLE IF NOT EXISTS song_features (
    song_id          INTEGER PRIMARY KEY REFERENCES songs(id),
    bpm              REAL,
    key              TEXT,
    chords           TEXT,   -- JSON array: ["Am", "F", "C", "G"]
    genres           TEXT,   -- JSON array: ["trap", "drill"]
    mood             TEXT,   -- free text: "dark, aggressive"
    energy           TEXT,   -- "low" | "medium" | "high"
    instruments      TEXT,   -- JSON array: ["808", "hi-hat", "piano"]
    similar_artists  TEXT,   -- JSON array: ["Lil Baby", "Gunna"]
    embedding        BLOB,   -- CLAP 512-d float32, stored as bytes
    extracted_at     TEXT
);
```

### New file: `extractor.py`

- `extract_song(db, song_id, doc_path, wav_path, log=print)` — main entry point
  - Reads the MD file, uses regex against fixed section headers to extract fields
  - Computes CLAP embedding for the WAV by calling the model directly (reuses server.py's `_get_clap_model()`)
  - Upserts into `song_features`
  - Returns the features dict or None on failure

- `backfill(db, analyses_dir, music_dir, log=print)` — processes all `done` songs without a features row
  - Used as a one-time catchup when Phase 4 first runs

- `__main__` — runs backfill then prints summary stats

### MD parsing strategy

The agent writes MDs with consistent headers. Regex targets:
- `## Production Snapshot` block → BPM (`\b(\d+)\s*(?:BPM|bpm)`), key (`key[:\s]+([A-G][#b]?\s*(?:major|minor)?)`)
- `## Emotional Character` block → mood (first sentence, max 60 chars)
- `## Sonic Breakdown` block + `## Production Snapshot` → instruments (comma-separated list after "instruments:" or "stems:")
- `## Related Artists & Sounds` block → similar artists (extract bold/list names)
- Genre: inferred from similar artists + mood context (simple keyword matching against known genre terms)

### CLAP embedding

Reuses the CLAP model already loaded in `server.py`. `extractor.py` imports `_get_clap_model` and `_load_audio_clap` from `server.py` to compute the 512-d embedding. Stored as `numpy.float32.tobytes()`, retrieved with `numpy.frombuffer(..., dtype=numpy.float32)`.

---

## 2. Query Layer

### `query.py` — CLI

```
uv run python query.py --bpm 135-145
uv run python query.py --key "C minor" --genre trap --limit 20
uv run python query.py --mood dark --energy high
uv run python query.py --similar ~/Documents/music/music\ data/artist_title.wav --n 5
uv run python query.py --insights
```

All flags optional and combinable. Structural flags query `song_features` via SQL. `--similar` loads the target WAV's stored embedding (or computes it on the fly) and returns top-N by cosine similarity.

Output: table to stdout (artist, title, BPM, key, mood, doc path).

### New MCP tools (added to `server.py`)

**`query_library(bpm_min, bpm_max, key, genre, mood, energy, limit=20)`**
- SQL query over `song_features JOIN songs`
- Returns list of `{artist, title, bpm, key, mood, doc_path, wav_path}`

**`similar_songs(wav_path, n=5)`**
- Computes CLAP embedding for `wav_path` (or looks up stored embedding)
- Cosine similarity against all stored embeddings
- Returns top-N matches with similarity scores

**`library_insights()`**
- Aggregates across `song_features`:
  - BPM distribution (min/max/median/mode)
  - Top 10 keys
  - Top 10 genres
  - Top 20 similar artists mentioned (across all analyses)
  - Energy distribution
  - Total analyzed vs pending

---

## 3. Dashboard

### `dashboard.py`

Uses `rich` library (`rich` added to dependencies). Live-refreshing terminal UI.

```
uv run python dashboard.py          # single snapshot
uv run python dashboard.py --watch  # live refresh every 5s
```

Layout (stacked panels):
```
┌─ Library Stats ──────────────────────────────────┐
│ Done: 87  Analyzing: 1  Pending: 142  Error: 0   │
│ Discovered: 23  (YouTube: 12, SoundCloud: 5, Billboard: 6) │
└──────────────────────────────────────────────────┘
┌─ Agent Status ───────────────────────────────────┐
│ [last 5 lines of agent.log]                       │
└──────────────────────────────────────────────────┘
┌─ Recent Analyses (last 10) ──────────────────────┐
│ Artist          Title              BPM   Key      │
│ ...                                               │
└──────────────────────────────────────────────────┘
┌─ Top Keys   ┐ ┌─ BPM Distribution ──────────────┐
│ C minor: 14 │ │ 120-130: ████ 8                 │
│ A minor: 11 │ │ 130-140: ████████ 16            │
│ ...         │ │ 140-150: ██████████ 21           │
└─────────────┘ └─────────────────────────────────┘
```

---

## 4. Agent + Ableton Wiring

### Agent change (`agent.py`)

After `db.mark_done()` succeeds, call:
```python
from extractor import extract_song
extract_song(db, song_id, doc_path, wav_path, log)
```
Errors are caught and logged, never fail the main loop.

### New MCP tool: `get_library_suggestions(bpm, key, n=10)`

Queries `song_features` for songs matching `bpm ± 10` and same key (or relative major/minor). Returns ranked list with doc paths — designed to be called from Ableton MCP context: "suggest library songs that fit this session."

---

## New Dependencies

- `rich>=13.0.0` — dashboard terminal UI

---

## Files Changed

| File | Change |
|------|--------|
| `extractor.py` | **New** — MD parser + CLAP embedding pipeline |
| `query.py` | **New** — CLI query tool |
| `dashboard.py` | **New** — terminal dashboard |
| `song_db.py` | **Modify** — add `song_features` table migration + query helpers |
| `server.py` | **Modify** — add `query_library`, `similar_songs`, `library_insights`, `get_library_suggestions` tools |
| `agent.py` | **Modify** — call `extract_song` after `mark_done` |
| `pyproject.toml` | **Modify** — add `rich>=13.0.0` |

---

## Verification

1. `uv run python extractor.py` → backfills all done songs, prints features summary
2. `uv run python query.py --insights` → prints library-wide stats
3. `uv run python query.py --key "C minor" --bpm 130-150` → returns matching songs
4. `uv run python query.py --similar <wav_path> --n 5` → returns 5 similar songs
5. `uv run python dashboard.py` → renders panel layout without errors
6. MCP tool smoke test: `uv run python -c "from server import query_library, similar_songs, library_insights; print('OK')"`
