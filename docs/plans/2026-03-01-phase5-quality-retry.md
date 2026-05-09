# Phase 5 — Analysis Quality Scoring & Smart Retry

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the agent self-correcting — score every analysis doc after it's written, auto-retry poor-quality docs with a different tool set, and prioritize chart/discovered songs at the front of the queue.

**Architecture:** `quality.py` scores docs 0–100 across 6 checks. `song_db.py` gets 3 new columns (`retry_count`, `quality_score`, `quality_issues`) and 5 new methods. `get_pending_songs()` is rewritten to return smart-ordered queue (retries → billboard → discovered → FIFO). `agent.py` calls quality scoring after each analysis and selects the tool set based on `retry_count`.

**Tech Stack:** SQLite (existing), Python stdlib re/json (no new deps)

**Critical:** No real analysis docs exist yet (229 pending, 0 done). All tests use synthetic docs — don't wait for real data to test.

---

### Task 1: Create `quality.py` — doc quality scorer

**Files:**
- Create: `quality.py`

**Step 1: Create `quality.py`**

```python
#!/usr/bin/env python3
"""
Scores analysis markdown documents for quality.
Called after every successful analysis to decide if re-analysis is needed.
"""
import re
from pathlib import Path

REQUIRED_SECTIONS = [
    "Production Snapshot",
    "Full Lyrics",
    "Lyrical Analysis",
    "Sonic Breakdown",
    "Emotional Character",
    "Why This Works",
    "Popularity & Real-World Reception",
    "Related Artists & Sounds",
]

_PLACEHOLDER_PATTERNS = [
    "[test", "[no lyrics", "[inaudible", "[instrumental",
    "test lyrics", "test analysis",
]


def _get_section(text: str, header: str) -> str:
    """Return text of a ## section, or empty string."""
    pattern = re.compile(
        r'^## ' + re.escape(header) + r'\s*\n(.*?)(?=^## |\Z)',
        re.MULTILINE | re.DOTALL
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def score_doc(doc_path: Path, features: dict | None = None) -> dict:
    """
    Score an analysis doc 0-100 across 6 checks.
    Returns {score, issues, quality} where quality is 'good'|'needs_retry'|'bad'.

    Args:
        doc_path: Path to the .md analysis document.
        features: Dict from extractor.parse_analysis_doc (bpm, key, similar_artists, etc.)
                  Pass None if extractor hasn't run yet.
    """
    if not doc_path or not doc_path.exists():
        return {"score": 0, "issues": ["document not found"], "quality": "bad"}

    text = doc_path.read_text(encoding="utf-8", errors="replace")
    issues = []
    score = 0

    # ── Check 1: Word count (20 pts) ──────────────────────────────────────
    word_count = len(text.split())
    if word_count >= 500:
        score += 20
    elif word_count >= 200:
        score += 10
        issues.append(f"short document ({word_count} words, expected ≥500)")
    else:
        issues.append(f"very short document ({word_count} words)")

    # ── Check 2: Required sections present (25 pts) ───────────────────────
    missing = [s for s in REQUIRED_SECTIONS if f"## {s}" not in text]
    section_pts = int(25 * (len(REQUIRED_SECTIONS) - len(missing)) / len(REQUIRED_SECTIONS))
    score += section_pts
    if missing:
        issues.append(f"missing sections: {', '.join(missing)}")

    # ── Check 3: Lyrics quality (20 pts) ──────────────────────────────────
    lyrics = _get_section(text, "Full Lyrics")
    if lyrics:
        clean = re.sub(r'\[.*?\]', '', lyrics).strip()
        is_placeholder = any(p in lyrics.lower() for p in _PLACEHOLDER_PATTERNS)
        if not is_placeholder and len(clean) >= 100:
            score += 20
        elif len(clean) >= 20:
            score += 10
            issues.append("lyrics may be incomplete or placeholder")
        else:
            issues.append("lyrics empty or all placeholder text")
    else:
        issues.append("Full Lyrics section is empty")

    # ── Check 4: BPM extracted (15 pts) ───────────────────────────────────
    if features and features.get("bpm"):
        score += 15
    else:
        issues.append("BPM not extracted from doc")

    # ── Check 5: Key extracted (10 pts) ───────────────────────────────────
    if features and features.get("key"):
        score += 10
    else:
        issues.append("key not extracted from doc")

    # ── Check 6: Similar artists (10 pts) ─────────────────────────────────
    if features and features.get("similar_artists") and len(features["similar_artists"]) >= 2:
        score += 10
    else:
        issues.append("fewer than 2 similar artists found")

    # ── Quality tier ──────────────────────────────────────────────────────
    if score >= 75:
        quality = "good"
    elif score >= 40:
        quality = "needs_retry"
    else:
        quality = "bad"

    return {"score": score, "issues": issues, "quality": quality}


# ─── Self-tests ──────────────────────────────────────────────────────────────

_GOOD_DOC = """# Test Song — Test Artist

## Production Snapshot
BPM: 140, Key: C minor, Chord progression: Am - F - C - G
Duration: 3:12, Instruments: 808 bass, hi-hat, snare, piano

## Full Lyrics
Verse 1:
Walking through the city late at night
Every shadow tells a different story bright
The neon lights reflect on broken glass
These moments always seem to fade too fast

Chorus:
We're living in the moment, can't look back
The future's bright but the past is turning black
Hold on tight, we're going for a ride
Nothing's gonna stop us side by side

## Lyrical Analysis
The lyrics explore themes of urban isolation and transient beauty. The opening
imagery of city lights creates a vivid nocturnal setting. The chorus pivots to
defiant optimism despite the darkness suggested by "past is turning black."

## Sonic Breakdown
Heavy 808 bass anchors the track at 140bpm with a half-time feel. Hi-hats roll
in triplet patterns creating forward momentum. Piano stabs on beats 2 and 4
add melodic color while the snare hits hard on the backbeat.

## Emotional Character
Dark and aggressive energy throughout, building tension from verse to chorus.
High energy track with an intense, brooding mood that shifts to triumphant in
the chorus section.

## Why This Works
The contrast between the dark verses and the uplifting chorus creates emotional
tension and release. The 808 bass provides the low-end foundation that makes
the drop hit hard.

## Popularity & Real-World Reception
This track has accumulated over 10 million streams on Spotify with a popularity
score of 78/100. The song peaked at #23 on the Billboard Hot 100.

## Related Artists & Sounds
- **Lil Baby** — similar melodic trap production and emotional delivery
- **Gunna** — comparable auto-tune usage and beat selection
- Young Thug, Roddy Ricch, NBA YoungBoy share the melodic sensibility
"""

_BAD_DOC = """# Bad Song — Unknown Artist

## Production Snapshot
[placeholder]

## Full Lyrics
[test lyrics]
"""


def _run_self_tests():
    import tempfile, json

    # Test good doc
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(_GOOD_DOC)
        good_path = Path(f.name)

    good_features = {
        "bpm": 140.0, "key": "C minor",
        "similar_artists": ["Lil Baby", "Gunna", "Young Thug"],
    }
    good = score_doc(good_path, good_features)
    good_path.unlink()
    assert good["score"] >= 75, f"good doc scored {good['score']}, expected ≥75"
    assert good["quality"] == "good", f"good doc quality={good['quality']}"
    print(f"Good doc: score={good['score']}, quality={good['quality']!r}, issues={good['issues']}")

    # Test bad doc
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(_BAD_DOC)
        bad_path = Path(f.name)

    bad = score_doc(bad_path, None)
    bad_path.unlink()
    assert bad["score"] < 40, f"bad doc scored {bad['score']}, expected <40"
    assert bad["quality"] == "bad", f"bad doc quality={bad['quality']}"
    assert len(bad["issues"]) >= 3, f"bad doc should have ≥3 issues, got: {bad['issues']}"
    print(f"Bad doc: score={bad['score']}, quality={bad['quality']!r}, issues={bad['issues']}")

    # Test missing doc
    missing = score_doc(Path("/tmp/does_not_exist_xyz.md"), None)
    assert missing["score"] == 0
    assert missing["quality"] == "bad"
    print(f"Missing doc: score={missing['score']}, quality={missing['quality']!r}")

    print("All quality.py self-tests PASSED")


if __name__ == "__main__":
    _run_self_tests()
```

**Step 2: Run self-tests**

Run: `uv run python quality.py`
Expected:
```
Good doc: score=<≥75>, quality='good', issues=[...]
Bad doc: score=<≤39>, quality='bad', issues=[...]
Missing doc: score=0, quality='bad'
All quality.py self-tests PASSED
```

**Step 3: Verify syntax**

Run: `uv run python -c "import ast; ast.parse(open('quality.py').read()); print('OK')"`
Expected: `OK`

---

### Task 2: Add DB columns + methods + smart queue to `song_db.py`

**Files:**
- Modify: `song_db.py`

**Step 1: Add SQLite column migration to `_init_db()`**

In `SongDB._init_db()`, after the `conn.executescript(...)` call (after line 61), add:

```python
            # Migrate: add quality/retry columns if they don't exist yet
            for col_def in [
                "ALTER TABLE songs ADD COLUMN retry_count INTEGER DEFAULT 0",
                "ALTER TABLE songs ADD COLUMN quality_score INTEGER",
                "ALTER TABLE songs ADD COLUMN quality_issues TEXT",
            ]:
                try:
                    conn.execute(col_def)
                except Exception:
                    pass  # column already exists
```

**Step 2: Replace `get_pending_songs()` with smart-ordered version**

Replace the entire `get_pending_songs` method (lines 71–77) with:

```python
def get_pending_songs(self) -> list[str]:
    """
    Return file paths to analyze, smart-ordered by priority:
    1. needs_retry songs (retry_count < 3)
    2. pending billboard chart songs
    3. pending discovered songs (any source)
    4. remaining pending songs (FIFO)
    """
    with self._conn() as conn:
        rows = conn.execute(
            """SELECT s.file_path
               FROM songs s
               WHERE s.status IN ('pending', 'needs_retry')
                 AND (s.retry_count IS NULL OR s.retry_count < 3)
               ORDER BY
                 CASE s.status WHEN 'needs_retry' THEN 0 ELSE 1 END,
                 CASE WHEN EXISTS(
                   SELECT 1 FROM discovery_log dl
                   WHERE dl.downloaded_file = s.file_path
                     AND dl.source_artist = 'billboard'
                 ) THEN 0 ELSE 1 END,
                 CASE WHEN EXISTS(
                   SELECT 1 FROM discovery_log dl
                   WHERE dl.downloaded_file = s.file_path
                 ) THEN 0 ELSE 1 END,
                 s.id"""
        ).fetchall()
    return [row["file_path"] for row in rows]
```

**Step 3: Add 5 new methods to `SongDB`**

Append after `features_stats()` (before the `if __name__ == "__main__":` block):

```python
def get_song_retry_info(self, file_path: str) -> dict:
    """Return {retry_count, quality_score, quality_issues} for a song."""
    with self._conn() as conn:
        row = conn.execute(
            "SELECT retry_count, quality_score, quality_issues FROM songs WHERE file_path = ?",
            (str(file_path),)
        ).fetchone()
    if not row:
        return {"retry_count": 0, "quality_score": None, "quality_issues": []}
    return {
        "retry_count": row["retry_count"] or 0,
        "quality_score": row["quality_score"],
        "quality_issues": json.loads(row["quality_issues"] or "[]"),
    }

def update_quality(self, file_path: str, score: int, issues: list):
    """Store quality score on a good (no retry needed) done song."""
    with self._conn() as conn:
        conn.execute(
            "UPDATE songs SET quality_score = ?, quality_issues = ? WHERE file_path = ?",
            (score, json.dumps(issues), str(file_path))
        )

def mark_needs_retry(self, file_path: str, score: int, issues: list):
    """Set status='needs_retry' and store quality info."""
    with self._conn() as conn:
        conn.execute(
            """UPDATE songs SET status = 'needs_retry', quality_score = ?,
               quality_issues = ? WHERE file_path = ?""",
            (score, json.dumps(issues), str(file_path))
        )

def mark_abandoned(self, file_path: str, score: int, issues: list):
    """Set status='abandoned' after max retries exhausted."""
    with self._conn() as conn:
        conn.execute(
            """UPDATE songs SET status = 'abandoned', quality_score = ?,
               quality_issues = ? WHERE file_path = ?""",
            (score, json.dumps(issues), str(file_path))
        )

def increment_retry_count(self, file_path: str):
    """Increment retry_count by 1 before re-analyzing a needs_retry song."""
    with self._conn() as conn:
        conn.execute(
            "UPDATE songs SET retry_count = COALESCE(retry_count, 0) + 1 WHERE file_path = ?",
            (str(file_path),)
        )

def promote_errors_to_retry(self):
    """Move all status='error' songs to 'needs_retry' so they get re-tried."""
    with self._conn() as conn:
        conn.execute(
            "UPDATE songs SET status = 'needs_retry' WHERE status = 'error'"
        )
```

**Step 4: Verify migration + new methods**

Run: `uv run python -c "
from pathlib import Path
from song_db import SongDB
db = SongDB(Path('agent.db'))

# Check columns exist
import sqlite3
conn = sqlite3.connect('agent.db')
cols = {r[1] for r in conn.execute('PRAGMA table_info(songs)').fetchall()}
conn.close()
assert 'retry_count' in cols, f'retry_count missing from songs, cols={cols}'
assert 'quality_score' in cols, 'quality_score missing'
assert 'quality_issues' in cols, 'quality_issues missing'
print('columns OK')

# Check smart queue ordering works on empty features
songs = db.get_pending_songs()
print(f'pending queue: {len(songs)} songs')

# Check new methods (using a placeholder song_id)
db.update_quality('/tmp/fake.wav', 85, ['test issue'])
info = db.get_song_retry_info('/tmp/fake.wav')
print('get_song_retry_info on unknown path:', info)
print('PASS')
"`
Expected: `columns OK`, `pending queue: 229 songs`, `PASS`

**Step 5: Verify syntax**

Run: `uv run python -c "import ast; ast.parse(open('song_db.py').read()); print('song_db.py OK')"`
Expected: `song_db.py OK`

---

### Task 3: Update `agent.py` — retry logic + quality scoring

**Files:**
- Modify: `agent.py`

**Step 1: Update `build_analysis_prompt()` to accept retry params**

Replace the function signature (line 83):
```python
def build_analysis_prompt(wav_path: Path, doc_path: Path, artist: str, title: str, spotify_data: dict = None) -> str:
```
With:
```python
def build_analysis_prompt(wav_path: Path, doc_path: Path, artist: str, title: str,
                          spotify_data: dict = None, retry_count: int = 0,
                          quality_issues: list = None) -> str:
```

Then add a retry preamble block right after the `spotify_block` assignment (before the `return f"""` line):

```python
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
```

Then update the return statement — the opening line currently starts with:
```python
    return f"""You are a music intelligence analyst...
```
Change it to include the retry_block right after the first line:
```python
    return f"""You are a music intelligence analyst. Complete ALL steps below autonomously without asking questions. Write the final document using the Write tool.
{retry_block}
SONG FILE: {wav_path}
```

**Step 2: Update `run_claude_analysis()` to use retry tool sets**

Replace the function signature (line 162):
```python
def run_claude_analysis(wav_path: Path, doc_path: Path, artist: str, title: str, log, spotify_data: dict = None) -> bool:
```
With:
```python
def run_claude_analysis(wav_path: Path, doc_path: Path, artist: str, title: str, log,
                        spotify_data: dict = None, retry_count: int = 0,
                        quality_issues: list = None) -> bool:
```

Replace the prompt + allowed_tools block (lines 168–172):
```python
    prompt = build_analysis_prompt(wav_path, doc_path, artist, title, spotify_data=spotify_data)

    allowed_tools = (
        "deep_listen,transcribe_lyrics,WebSearch,Write,Read"
    )
```
With:
```python
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
```

**Step 3: Update `main_loop()` — fetch retry info, call quality scoring**

In `main_loop()`, find the block that starts with:
```python
        db.mark_analyzing(str(wav_path))
```

Add these lines BEFORE `db.mark_analyzing(...)` to fetch retry info and increment if it's a retry:

```python
        # Fetch retry info before marking as analyzing
        retry_info = db.get_song_retry_info(str(wav_path))
        retry_count = retry_info["retry_count"]
        quality_issues = retry_info["quality_issues"]
        if retry_count > 0:
            db.increment_retry_count(str(wav_path))
            log(f"  Retry attempt {retry_count}/3 — prior issues: {quality_issues}")

        db.mark_analyzing(str(wav_path))
```

Then find the line:
```python
        success = run_claude_analysis(wav_path, doc_path, artist, title, log, spotify_data=spotify_data)
```
Replace with:
```python
        success = run_claude_analysis(
            wav_path, doc_path, artist, title, log,
            spotify_data=spotify_data,
            retry_count=retry_count,
            quality_issues=quality_issues,
        )
```

Then find the success block where `_extract` is called:
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

Replace with:
```python
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
```

**Step 4: Add `promote_errors_to_retry()` at startup in `main_loop()`**

Find the block just before the `while True:` loop:
```python
    songs_since_discovery = 0

    while True:
```

Add between them:
```python
    songs_since_discovery = 0

    # Promote old errors to needs_retry so they get a second chance
    db.promote_errors_to_retry()
    log("Promoted any previous error songs to needs_retry")

    while True:
```

**Step 5: Verify syntax**

Run: `uv run python -c "import ast; ast.parse(open('agent.py').read()); print('agent.py OK')"`
Expected: `agent.py OK`

**Step 6: End-to-end smoke test — synthetic retry scenario**

Run: `uv run python -c "
import json
from pathlib import Path
from song_db import SongDB
from quality import score_doc
import tempfile

db = SongDB(Path('agent.db'))

# Simulate a low-quality done song
test_path = '/tmp/test_quality_song.wav'
db.add_song(test_path, title='Test', artist='Test Artist')
db.mark_analyzing(test_path)

with tempfile.NamedTemporaryFile(suffix='.md', mode='w', delete=False) as f:
    f.write('# Bad\n## Production Snapshot\n[placeholder]\n## Full Lyrics\n[test lyrics]\n')
    doc_path = Path(f.name)

db.mark_done(test_path, str(doc_path))
result = score_doc(doc_path, None)
print(f'Score: {result[\"score\"]}, quality: {result[\"quality\"]}, issues: {result[\"issues\"]}')
assert result['quality'] in ('bad', 'needs_retry'), f'expected bad/needs_retry, got {result[\"quality\"]}'

db.mark_needs_retry(test_path, result['score'], result['issues'])
db.increment_retry_count(test_path)
info = db.get_song_retry_info(test_path)
print(f'After retry increment: retry_count={info[\"retry_count\"]}, score={info[\"quality_score\"]}')
assert info['retry_count'] == 1

# Check it appears in pending queue
queue = db.get_pending_songs()
assert test_path in queue, 'needs_retry song should appear in queue'
print(f'Queue includes needs_retry song: YES')

# Cleanup
import sqlite3; conn = sqlite3.connect('agent.db'); conn.execute('DELETE FROM songs WHERE file_path=?', (test_path,)); conn.commit(); conn.close()
doc_path.unlink()
print('Smoke test PASSED')
"`
Expected: `Smoke test PASSED`

---

## Verification checklist

1. `uv run python quality.py` → "All quality.py self-tests PASSED"
2. `uv run python -c "..."` (Task 2 Step 4) → "columns OK", "PASS"
3. `uv run python -c "import ast; ast.parse(open('agent.py').read()); print('OK')"` → "OK"
4. Task 3 Step 6 smoke test → "Smoke test PASSED"
5. `uv run python query.py --insights` → no crash (stats will include new statuses once agent runs)
