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
    m = re.search(r'\b(\d{2,3}(?:\.\d)?)\b\s*(?:BPM|bpm)', text)
    if m:
        return float(m.group(1))
    m = re.search(r'(?:BPM|bpm|Tempo)[:\s]+\b(\d{2,3}(?:\.\d)?)\b', text)
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
        r'(?:instruments?|stems?|detected\s+instruments?)[:\s]+([^\n]{4,200})',
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
            raw_name = m.group(1).strip()
            # split comma-separated names on a single bullet line
            for name in re.split(r',\s*', raw_name):
                name = name.strip()
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
    genres = _infer_genres(full_text=text)

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
            # Some transformers versions return BaseModelOutputWithPooling instead of a tensor
            if not isinstance(embed, torch.Tensor):
                embed = embed.pooler_output if hasattr(embed, 'pooler_output') else embed.last_hidden_state.mean(dim=1)
            embed = torch.nn.functional.normalize(embed, dim=-1)
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
