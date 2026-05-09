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
        elif not is_placeholder and len(clean) >= 20:
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
    # When features=None, max achievable score is 65 (checks 4-6 unavailable).
    # Scale thresholds proportionally so tier is consistent.
    max_score = 65 if features is None else 100
    if score >= int(75 * max_score / 100):
        quality = "good"
    elif score >= int(40 * max_score / 100):
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
    import tempfile

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
