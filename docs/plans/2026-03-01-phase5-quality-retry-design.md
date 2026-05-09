# Phase 5 — Analysis Quality Scoring & Smart Retry
**Date:** 2026-03-01
**Status:** Approved

## Overview

Phase 5 makes the agent self-correcting. After each analysis, a quality scorer evaluates the output and flags low-quality docs for retry. Retries use a progressively different tool set and an explicit remediation prompt. The queue is also reordered so chart songs and recent discoveries are analyzed before older library tracks.

---

## Architecture

```
agent.py
  │ run_claude_analysis(retry_count=N)
  │   └─ build_analysis_prompt(retry_count) → picks tool set
  │
  ↓ on success
extractor.py → extract_song() → upsert song_features
  │
quality.py → score_doc(doc_path, features)
  │ score ≥ 75 → stay 'done'
  │ score < 75, retry_count < 3 → mark_needs_retry()
  │ score < 75, retry_count ≥ 3 → mark_abandoned()
  │
song_db.py → get_pending_songs() [smart-ordered]
  priority: needs_retry > billboard > discovered > FIFO
```

---

## 1. Quality Scoring (`quality.py`)

### `score_doc(doc_path, features) -> dict`

Returns `{score: int, issues: list[str], quality: str}`.

| Check | Points |
|---|---|
| Word count ≥ 500 | 20 |
| All 8 required sections present | 25 |
| Lyrics not empty / not all `[?]` | 20 |
| `features["bpm"]` not None | 15 |
| `features["key"]` not None | 10 |
| `features["similar_artists"]` has ≥ 2 entries | 10 |

Required sections (from agent.py prompt template):
- Production Snapshot, Full Lyrics, Lyrical Analysis, Sonic Breakdown,
  Emotional Character, Why This Works, Popularity & Real-World Reception,
  Related Artists & Sounds

Score thresholds:
- `≥ 75` → `"good"` — no retry needed
- `40–74` → `"needs_retry"` — retry on next cycle
- `< 40` → `"bad"` — same as needs_retry but logged more urgently

### `score_doc` is called by agent.py after every successful analysis.

---

## 2. Database Changes (`song_db.py`)

### New columns on `songs` table (migration in `_init_db`):

```sql
ALTER TABLE songs ADD COLUMN retry_count    INTEGER DEFAULT 0;
ALTER TABLE songs ADD COLUMN quality_score  INTEGER;
ALTER TABLE songs ADD COLUMN quality_issues TEXT;   -- JSON array
```

(Added via `CREATE TABLE IF NOT EXISTS` migration — safe on existing DBs using `ALTER TABLE IF NOT EXISTS` pattern or try/except.)

### New status value: `needs_retry`

Full status flow:
```
pending → analyzing → done           (quality good)
                    → needs_retry    (quality < 75, retry_count < 3)
                    → abandoned      (retry_count ≥ 3)
error   → needs_retry                (auto-promote errors for retry)
```

### New DB methods:

```python
def update_quality(self, file_path: str, score: int, issues: list[str]):
    """Store quality score and issues on a done song."""

def mark_needs_retry(self, file_path: str, score: int, issues: list[str]):
    """Set status='needs_retry', store quality info."""

def mark_abandoned(self, file_path: str, score: int, issues: list[str]):
    """Set status='abandoned' after max retries exceeded."""

def increment_retry_count(self, file_path: str):
    """Bump retry_count by 1 before re-analyzing."""
```

### Smart queue ordering in `get_pending_songs()`:

Returns file paths ordered by priority:
1. `needs_retry` songs (retry_count < 3)
2. `pending` songs present in `discovery_log` with `source_artist='billboard'`
3. `pending` songs present in `discovery_log` (any source)
4. All remaining `pending` songs (FIFO by id)

Implementation: single SQL UNION or ORDER BY CASE expression.

---

## 3. Agent Changes (`agent.py`)

### `build_analysis_prompt(wav_path, doc_path, artist, title, spotify_data, retry_count, quality_issues)`

Tool set selected by `retry_count`:

| retry_count | Tools passed to `--allowedTools` | Prompt preamble |
|---|---|---|
| 0 | `deep_listen, transcribe_lyrics, WebSearch, Write, Read` | None (current behavior) |
| 1 | `analyze_audio, transcribe_lyrics, detect_chords, get_song_structure, WebSearch, Write, Read` | Remediation note with prior issues |
| 2 | `analyze_stems, transcribe_lyrics, detect_chords, WebSearch, Write, Read` | Stronger remediation note |

Remediation preamble (injected when retry_count > 0):
```
RETRY ATTEMPT {retry_count} — Previous analysis scored {score}/100.
Issues found: {issues}
Use the listed tools individually this time. Ensure every section
contains real data — do not leave any section empty or as placeholder text.
```

### `run_claude_analysis()` signature change:
```python
def run_claude_analysis(wav_path, doc_path, artist, title, log,
                        spotify_data=None, retry_count=0, quality_issues=None)
```

### Main loop changes:

After `extract_song()` succeeds → call `quality.score_doc()`:
```python
from quality import score_doc
result = score_doc(doc_path, features)
if result["quality"] == "good":
    db.update_quality(str(wav_path), result["score"], result["issues"])
    log(f"  Quality: {result['score']}/100 ✓")
elif retry_count < 3:
    db.mark_needs_retry(str(wav_path), result["score"], result["issues"])
    log(f"  Quality: {result['score']}/100 — queued for retry")
else:
    db.mark_abandoned(str(wav_path), result["score"], result["issues"])
    log(f"  Quality: {result['score']}/100 — abandoned after {retry_count} retries")
```

`retry_count` and `quality_issues` are fetched from DB before analysis and passed through.

Also: promote `error` songs to `needs_retry` at startup (one-time migration call).

---

## Files Changed

| File | Change |
|---|---|
| `quality.py` | **New** — score_doc() |
| `song_db.py` | Add 3 columns, 4 methods, rewrite get_pending_songs() |
| `agent.py` | Pass retry_count through, call score_doc, smart queue |

## Unchanged
`server.py`, `extractor.py`, `query.py`, `dashboard.py`, `discovery.py`, `charts.py`, `power.py`

---

## Verification

1. `uv run python quality.py` → runs self-test against synthetic good/bad docs, prints scores
2. `uv run python -c "from song_db import SongDB; ..."` → verify new columns + methods
3. `uv run python -c "import ast; ast.parse(open('agent.py').read()); print('OK')"` → syntax clean
4. Manually insert a low-quality done song → confirm it gets marked `needs_retry`
5. `uv run python query.py --insights` → shows retry/abandoned counts
