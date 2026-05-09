#!/usr/bin/env python3
"""
Audio Understanding MCP Server — Hermes/Native Edition.

Gives your AI agent the ability to natively listen to and understand music/audio.
Works with any MCP-compatible client (Hermes Agent, Claude Code, Codex CLI, etc.).
Combines:
  - CLAP for semantic audio understanding (genre, mood, instruments)
  - Demucs for source separation (vocals, drums, bass, other)
  - Whisper large-v3 for lyrics transcription from isolated vocals
  - librosa for signal analysis (tempo, key, chords, structure, rhythm)

New in v4:
  - LRU memory + disk cache for all expensive operations (CLAP, Demucs, Whisper, librosa)
  - Deterministic chunk IDs for RAG referencing (stems, whisper chunks, sections, chords, phrases)
  - Schema versioning on all tool outputs (auto-invalidates cache on format changes)
  - Parallel CPU/GPU pipeline in deep_listen (~25% faster)

Supports MP3, WAV, FLAC, OGG, M4A, AAC, OPUS.
v4.1: GPU lock, safe disk cache (JSON/NPZ), disk eviction, Whisper arg fixes.
v4.2: Audio download tools (download_audio, search_and_download) via yt-dlp.
"""

import gc
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from mcp.server.fastmcp import FastMCP
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, ClapModel, ClapProcessor, pipeline

# Suppress known noisy warnings (set AUDIO_MCP_SHOW_WARNINGS=1 to see all)
if not os.environ.get("AUDIO_MCP_SHOW_WARNINGS"):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", message=".*PySoundFile.*")
    warnings.filterwarnings("ignore", message=".*n_fft.*")

# ─── Constants ────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "4.2"

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_SAMPLE_RATE = 48000
WHISPER_MODEL_ID = "openai/whisper-large-v3"
WHISPER_SR = 16000
DEMUCS_MODEL_NAME = "htdemucs"
DEMUCS_SR = 44100
LIBROSA_SR = 22050
MAX_DURATION = 900  # 15 min (up from 10)
WHISPER_CHUNK_SEC = 30  # chunk size for long-form transcription

SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

GENRE_LABELS = [
    "rock music", "pop music", "hip hop music", "jazz music",
    "classical music", "electronic music", "dance music", "R&B music",
    "soul music", "funk music", "heavy metal music", "punk rock",
    "country music", "folk music", "blues music", "reggae music",
    "latin music", "ambient music", "lo-fi music", "indie rock",
    "alternative rock", "world music", "film soundtrack",
    "new age music", "trap music", "house music", "techno music",
    "drum and bass", "dubstep", "gospel music",
]

INSTRUMENT_LABELS = [
    "acoustic guitar playing", "electric guitar playing",
    "bass guitar playing", "piano playing", "synthesizer",
    "drum kit playing", "violin playing", "cello playing",
    "trumpet playing", "saxophone playing", "flute playing",
    "harmonica playing", "organ playing", "banjo playing",
    "ukulele playing", "harp playing", "clarinet playing",
    "trombone playing", "percussion instruments", "turntable scratching",
    "singing voice", "choir singing", "string ensemble",
    "brass ensemble", "electric bass playing", "acoustic drums",
    "electronic drums", "beat boxing",
]

MOOD_LABELS = [
    "happy and joyful music", "sad and melancholic music",
    "energetic and upbeat music", "calm and relaxing music",
    "aggressive and intense music", "dark and ominous music",
    "uplifting and inspiring music", "romantic and tender music",
    "nostalgic and wistful music", "peaceful and serene music",
    "tense and suspenseful music", "dreamy and ethereal music",
    "powerful and epic music", "playful and fun music",
    "mysterious and eerie music", "angry and rebellious music",
    "groovy and danceable music", "contemplative and introspective music",
]

VOCAL_STYLE_LABELS = [
    "male singing voice", "female singing voice",
    "rapping voice", "whispering voice", "screaming voice",
    "falsetto singing", "deep voice singing", "auto-tuned voice",
    "spoken word", "humming", "vocal harmonies",
    "breathy singing", "powerful belting voice", "soft gentle singing",
]

# Krumhansl-Schmuckler key-finding profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Camelot wheel for harmonic mixing
CAMELOT_WHEEL = {
    "C major": "8B", "G major": "9B", "D major": "10B", "A major": "11B",
    "E major": "12B", "B major": "1B", "F# major": "2B", "C# major": "3B",
    "G# major": "4B", "D# major": "5B", "A# major": "6B", "F major": "7B",
    "A minor": "8A", "E minor": "9A", "B minor": "10A", "F# minor": "11A",
    "C# minor": "12A", "G# minor": "1A", "D# minor": "2A", "A# minor": "3A",
    "F minor": "4A", "C minor": "5A", "G minor": "6A", "D minor": "7A",
}

# ─── Expanded Chord Templates ────────────────────────────────────────────────

CHORD_TEMPLATES = {}
for _i, _name in enumerate(KEY_NAMES):
    # Major triad
    _t = np.zeros(12); _t[_i] = 1.0; _t[(_i+4)%12] = 1.0; _t[(_i+7)%12] = 1.0
    CHORD_TEMPLATES[_name] = _t.copy()
    # Minor triad
    _t = np.zeros(12); _t[_i] = 1.0; _t[(_i+3)%12] = 1.0; _t[(_i+7)%12] = 1.0
    CHORD_TEMPLATES[f"{_name}m"] = _t.copy()
    # Dominant 7th
    _t = np.zeros(12); _t[_i] = 1.0; _t[(_i+4)%12] = 1.0; _t[(_i+7)%12] = 1.0; _t[(_i+10)%12] = 0.8
    CHORD_TEMPLATES[f"{_name}7"] = _t.copy()
    # Minor 7th
    _t = np.zeros(12); _t[_i] = 1.0; _t[(_i+3)%12] = 1.0; _t[(_i+7)%12] = 1.0; _t[(_i+10)%12] = 0.8
    CHORD_TEMPLATES[f"{_name}m7"] = _t.copy()
    # Major 7th
    _t = np.zeros(12); _t[_i] = 1.0; _t[(_i+4)%12] = 1.0; _t[(_i+7)%12] = 1.0; _t[(_i+11)%12] = 0.8
    CHORD_TEMPLATES[f"{_name}maj7"] = _t.copy()
    # Diminished
    _t = np.zeros(12); _t[_i] = 1.0; _t[(_i+3)%12] = 1.0; _t[(_i+6)%12] = 1.0
    CHORD_TEMPLATES[f"{_name}dim"] = _t.copy()
    # Augmented
    _t = np.zeros(12); _t[_i] = 1.0; _t[(_i+4)%12] = 1.0; _t[(_i+8)%12] = 1.0
    CHORD_TEMPLATES[f"{_name}aug"] = _t.copy()
    # Sus2
    _t = np.zeros(12); _t[_i] = 1.0; _t[(_i+2)%12] = 1.0; _t[(_i+7)%12] = 1.0
    CHORD_TEMPLATES[f"{_name}sus2"] = _t.copy()
    # Sus4
    _t = np.zeros(12); _t[_i] = 1.0; _t[(_i+5)%12] = 1.0; _t[(_i+7)%12] = 1.0
    CHORD_TEMPLATES[f"{_name}sus4"] = _t.copy()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _log(msg: str):
    """Log to stderr (visible in MCP server logs)."""
    print(msg, file=sys.stderr, flush=True)


def validate_audio_path(file_path: str) -> Path:
    """Validate and resolve an audio file path."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}. Supported: {', '.join(SUPPORTED_FORMATS)}")
    return path


def estimate_key(chroma: np.ndarray) -> tuple[str, float]:
    """Estimate musical key using Krumhansl-Schmuckler algorithm."""
    chroma_avg = chroma.mean(axis=1)
    if np.all(chroma_avg == 0):
        return "unknown", 0.0
    best_corr = -2.0
    best_key = "C major"
    for i in range(12):
        rolled = np.roll(chroma_avg, -i)
        maj_corr = np.corrcoef(rolled, MAJOR_PROFILE)[0, 1]
        min_corr = np.corrcoef(rolled, MINOR_PROFILE)[0, 1]
        # Guard against NaN
        if np.isnan(maj_corr):
            maj_corr = -2.0
        if np.isnan(min_corr):
            min_corr = -2.0
        maj_corr = float(maj_corr)
        min_corr = float(min_corr)
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = f"{KEY_NAMES[i]} major"
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = f"{KEY_NAMES[i]} minor"
    return best_key, best_corr


def format_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


def hz_to_note(hz: float) -> str:
    """Convert frequency to nearest musical note name."""
    if hz <= 0:
        return "?"
    midi = 69 + 12 * np.log2(hz / 440.0)
    note_idx = int(round(midi)) % 12
    octave = int(round(midi)) // 12 - 1
    return f"{NOTE_NAMES[note_idx]}{octave}"


def resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a numpy audio array."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def key_distance_semitones(key_a: str, key_b: str) -> int:
    """Calculate semitone distance between two keys."""
    note_to_idx = {n: i for i, n in enumerate(KEY_NAMES)}
    try:
        root_a = key_a.split()[0]
        root_b = key_b.split()[0]
        diff = (note_to_idx[root_b] - note_to_idx[root_a]) % 12
        return diff if diff <= 6 else diff - 12
    except (KeyError, IndexError):
        return 0


# ─── Cache & Identity ─────────────────────────────────────────────────────────


class FileIdentifier:
    """Deterministic cache keys and chunk IDs for audio files."""

    @staticmethod
    def file_hash(file_path: str) -> str:
        """SHA-256 of resolved_path|mtime_ns|file_size, truncated to 16 hex chars."""
        path = Path(file_path).expanduser().resolve()
        stat = path.stat()
        identity = f"{path}|{stat.st_mtime_ns}|{stat.st_size}"
        return hashlib.sha256(identity.encode()).hexdigest()[:16]

    @staticmethod
    def cache_key(file_path: str, operation: str, **params) -> str:
        """Generate a cache key: {hash}:{op}:{sorted_params}:sv={SCHEMA_VERSION}"""
        h = FileIdentifier.file_hash(file_path)
        param_str = ":".join(f"{k}={v}" for k, v in sorted(params.items())) if params else ""
        parts = [h, operation]
        if param_str:
            parts.append(param_str)
        parts.append(f"sv={SCHEMA_VERSION}")
        return ":".join(parts)

    # ─── Chunk IDs ───

    @staticmethod
    def stem_id(file_path: str, stem_name: str) -> str:
        return f"{FileIdentifier.file_hash(file_path)}:stem:{stem_name}"

    @staticmethod
    def whisper_id(file_path: str, idx: int, start: float, end: float) -> str:
        return f"{FileIdentifier.file_hash(file_path)}:whisper:{idx}:{start:.2f}-{end:.2f}"

    @staticmethod
    def section_id(file_path: str, idx: int, start: float, end: float) -> str:
        return f"{FileIdentifier.file_hash(file_path)}:section:{idx}:{start:.2f}-{end:.2f}"

    @staticmethod
    def chord_id(file_path: str, idx: int, time_val: float) -> str:
        return f"{FileIdentifier.file_hash(file_path)}:chord:{idx}:{time_val:.2f}"

    @staticmethod
    def phrase_id(file_path: str, idx: int, start: float) -> str:
        return f"{FileIdentifier.file_hash(file_path)}:phrase:{idx}:{start:.3f}"


class AnalysisCache:
    """LRU memory cache + disk persistence using JSON/NPZ (no pickle).

    Disk cache is bounded by max_disk_bytes and ttl_days. Eviction runs
    on startup and periodically after writes.
    """

    _DISK_MAX_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB default
    _DISK_TTL_DAYS = 30

    def __init__(self, max_memory_items: int = 128, disk_dir: str = "~/.cache/audio-mcp"):
        self._memory = OrderedDict()
        self._max_items = max_memory_items
        self._disk_dir = Path(disk_dir).expanduser()
        self._disk_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        self._writes_since_evict = 0
        self._lock = threading.Lock()
        # Run eviction on startup in background
        threading.Thread(target=self._disk_evict, daemon=True).start()

    def _disk_path(self, key: str) -> Path:
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self._disk_dir / f"{safe_key}.cache"

    @staticmethod
    def _serialize(value) -> bytes:
        """Serialize a value to bytes using JSON for plain data, NPZ for numpy/torch."""
        if isinstance(value, torch.Tensor):
            import io
            buf = io.BytesIO()
            np.savez_compressed(buf, data=value.cpu().numpy())
            return b"NPZ_TENSOR\n" + buf.getvalue()
        if isinstance(value, np.ndarray):
            import io
            buf = io.BytesIO()
            np.savez_compressed(buf, data=value)
            return b"NPZ_ARRAY\n" + buf.getvalue()
        if isinstance(value, dict):
            # Check for nested numpy/tensor values (e.g. stems dict)
            has_arrays = any(isinstance(v, (np.ndarray, torch.Tensor)) for v in value.values())
            if has_arrays:
                import io
                buf = io.BytesIO()
                save_dict = {}
                meta = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        save_dict[k] = v
                        meta[k] = "ndarray"
                    elif isinstance(v, torch.Tensor):
                        save_dict[k] = v.cpu().numpy()
                        meta[k] = "tensor"
                    else:
                        meta[k] = json.dumps(v)
                np.savez_compressed(buf, **save_dict)
                header = json.dumps(meta).encode()
                return b"NPZ_DICT\n" + len(header).to_bytes(4, "little") + header + buf.getvalue()
        # Default: JSON
        return b"JSON\n" + json.dumps(value).encode()

    @staticmethod
    def _deserialize(data: bytes):
        """Deserialize bytes back to a Python object."""
        newline = data.index(b"\n")
        tag = data[:newline]
        payload = data[newline + 1:]
        if tag == b"JSON":
            return json.loads(payload)
        if tag == b"NPZ_ARRAY":
            import io
            return np.load(io.BytesIO(payload))["data"]
        if tag == b"NPZ_TENSOR":
            import io
            arr = np.load(io.BytesIO(payload))["data"]
            return torch.from_numpy(arr)
        if tag == b"NPZ_DICT":
            import io
            header_len = int.from_bytes(payload[:4], "little")
            meta = json.loads(payload[4:4 + header_len])
            npz_data = np.load(io.BytesIO(payload[4 + header_len:]))
            result = {}
            for k, v in meta.items():
                if v == "ndarray":
                    result[k] = npz_data[k]
                elif v == "tensor":
                    result[k] = torch.from_numpy(npz_data[k])
                else:
                    result[k] = json.loads(v)
            return result
        raise ValueError(f"Unknown cache format tag: {tag}")

    def get(self, key: str):
        with self._lock:
            # Memory first
            if key in self._memory:
                self._memory.move_to_end(key)
                self._hits += 1
                _log(f"[cache] HIT (memory): {key[:60]}")
                return self._memory[key]

        # Disk fallback (outside lock for I/O)
        disk_path = self._disk_path(key)
        if disk_path.exists():
            try:
                data = disk_path.read_bytes()
                value = self._deserialize(data)
                # Touch atime for LRU eviction
                disk_path.touch()
                # Promote to memory
                with self._lock:
                    self._memory[key] = value
                    self._evict()
                    self._hits += 1
                _log(f"[cache] HIT (disk): {key[:60]}")
                return value
            except Exception:
                disk_path.unlink(missing_ok=True)

        with self._lock:
            self._misses += 1
        return None

    def put(self, key: str, value, persist: bool = True):
        with self._lock:
            self._memory[key] = value
            self._memory.move_to_end(key)
            self._evict()

        if persist:
            try:
                disk_path = self._disk_path(key)
                data = self._serialize(value)
                # Atomic write: temp file then rename
                fd, tmp_path = tempfile.mkstemp(dir=self._disk_dir, suffix=".tmp")
                closed = False
                try:
                    os.write(fd, data)
                    os.close(fd)
                    closed = True
                    os.replace(tmp_path, disk_path)
                except Exception:
                    if not closed:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                    Path(tmp_path).unlink(missing_ok=True)
                    raise
                self._writes_since_evict += 1
                if self._writes_since_evict >= 50:
                    self._writes_since_evict = 0
                    threading.Thread(target=self._disk_evict, daemon=True).start()
            except Exception as e:
                _log(f"[cache] disk write failed: {e}")

    def invalidate(self, file_path: str):
        """Wipe all memory entries for a file hash."""
        h = FileIdentifier.file_hash(file_path)
        with self._lock:
            to_remove = [k for k in self._memory if k.startswith(h + ":")]
            for k in to_remove:
                del self._memory[k]
        _log(f"[cache] invalidated {len(to_remove)} memory entries for {h}")

    def _evict(self):
        """Evict oldest entries beyond max. Caller must hold self._lock."""
        while len(self._memory) > self._max_items:
            self._memory.popitem(last=False)

    def _disk_evict(self):
        """Remove expired and excess disk cache entries."""
        try:
            cache_files = list(self._disk_dir.glob("*.cache"))
            if not cache_files:
                return

            now = time.time()
            ttl_seconds = self._DISK_TTL_DAYS * 86400

            # Remove expired files first
            remaining = []
            for f in cache_files:
                try:
                    age = now - f.stat().st_mtime
                    if age > ttl_seconds:
                        f.unlink(missing_ok=True)
                    else:
                        remaining.append(f)
                except OSError:
                    pass

            # Check total size, evict oldest if over limit
            entries = []
            total_size = 0
            for f in remaining:
                try:
                    stat = f.stat()
                    entries.append((f, stat.st_mtime, stat.st_size))
                    total_size += stat.st_size
                except OSError:
                    pass

            if total_size > self._DISK_MAX_BYTES:
                # Sort by mtime ascending (oldest first)
                entries.sort(key=lambda x: x[1])
                removed = 0
                for f, _, size in entries:
                    if total_size <= self._DISK_MAX_BYTES:
                        break
                    f.unlink(missing_ok=True)
                    total_size -= size
                    removed += 1
                if removed:
                    _log(f"[cache] disk eviction: removed {removed} files, {total_size // (1024*1024)}MB remaining")

            # Clean up legacy pickle files from v4.0
            for f in self._disk_dir.glob("*.pkl"):
                f.unlink(missing_ok=True)

            # Clean up stale temp files
            for f in self._disk_dir.glob("*.tmp"):
                try:
                    if now - f.stat().st_mtime > 300:  # older than 5 min
                        f.unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception as e:
            _log(f"[cache] disk eviction error: {e}")

    def stats(self) -> dict:
        total = self._hits + self._misses
        disk_files = list(self._disk_dir.glob("*.cache"))
        disk_size = sum(f.stat().st_size for f in disk_files if f.exists())
        return {
            "memory_items": len(self._memory),
            "disk_files": len(disk_files),
            "disk_size_mb": round(disk_size / (1024 * 1024), 1),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0,
        }


# ─── VRAM Manager ─────────────────────────────────────────────────────────────

_device = "cuda" if torch.cuda.is_available() else "cpu"


class VRAMManager:
    """Manages GPU memory by loading/unloading models on demand."""

    def __init__(self):
        self.clap_model = None
        self.clap_processor = None
        self.demucs_model = None
        self.whisper_pipe = None
        self._loaded = set()  # track what's loaded

    def _free_vram(self):
        """Force garbage collection and CUDA cache clear."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_free_vram_mb(self) -> float:
        if not torch.cuda.is_available():
            return float("inf")
        free, total = torch.cuda.mem_get_info()
        return free / (1024 * 1024)

    def unload(self, model_name: str):
        """Unload a model from GPU to free VRAM. Caller must hold gpu_lock."""
        if model_name == "clap" and "clap" in self._loaded:
            del self.clap_model
            del self.clap_processor
            self.clap_model = None
            self.clap_processor = None
            self._loaded.discard("clap")
            self._free_vram()
            _log(f"Unloaded CLAP ({self._get_free_vram_mb():.0f}MB free)")

        elif model_name == "demucs" and "demucs" in self._loaded:
            del self.demucs_model
            self.demucs_model = None
            self._loaded.discard("demucs")
            self._free_vram()
            _log(f"Unloaded Demucs ({self._get_free_vram_mb():.0f}MB free)")

        elif model_name == "whisper" and "whisper" in self._loaded:
            del self.whisper_pipe
            self.whisper_pipe = None
            self._loaded.discard("whisper")
            self._free_vram()
            _log(f"Unloaded Whisper ({self._get_free_vram_mb():.0f}MB free)")

    def unload_all_except(self, keep: str = ""):
        """Unload all models except the specified one."""
        for name in list(self._loaded):
            if name != keep:
                self.unload(name)

    def get_clap(self):
        if self.clap_model is None:
            _log(f"Loading CLAP... ({self._get_free_vram_mb():.0f}MB free)")
            self.clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
            self.clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(_device).eval()
            self._loaded.add("clap")
            _log(f"CLAP loaded ({self._get_free_vram_mb():.0f}MB free)")
        return self.clap_model, self.clap_processor

    def get_demucs(self):
        if self.demucs_model is None:
            _log(f"Loading Demucs... ({self._get_free_vram_mb():.0f}MB free)")
            from demucs.pretrained import get_model
            self.demucs_model = get_model(DEMUCS_MODEL_NAME)
            self.demucs_model.to(_device)
            self.demucs_model.eval()
            self._loaded.add("demucs")
            _log(f"Demucs loaded ({self._get_free_vram_mb():.0f}MB free)")
        return self.demucs_model

    def get_whisper(self):
        if self.whisper_pipe is None:
            _log(f"Loading Whisper large-v3... ({self._get_free_vram_mb():.0f}MB free)")
            _dtype = torch.float16 if _device == "cuda" else torch.float32
            _dev = torch.device(_device)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID, torch_dtype=_dtype, low_cpu_mem_usage=True,
            ).to(_dev)
            processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
            self.whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=_dtype,
                device=_dev,
            )
            self._loaded.add("whisper")
            _log(f"Whisper loaded ({self._get_free_vram_mb():.0f}MB free)")
        return self.whisper_pipe


models = VRAMManager()
gpu_lock = threading.Lock()  # serialize all GPU model access across concurrent tool calls


# ─── Globals ──────────────────────────────────────────────────────────────────

cache = AnalysisCache()
_cpu_pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))


def _tool_output(data: dict, tool_name: str = "") -> str:
    """Wrap tool output with schema version, tool name, and ok/error status."""
    data["schema_version"] = SCHEMA_VERSION
    if tool_name:
        data["tool"] = tool_name
    data["ok"] = "error" not in data
    return json.dumps(data, indent=2)


# ─── Audio Loading (cached) ──────────────────────────────────────────────────


def load_audio(file_path: str, sr: int, duration: Optional[float] = None) -> np.ndarray:
    """Load audio file as mono numpy array at target sample rate."""
    key = FileIdentifier.cache_key(file_path, "load_audio", sr=sr, duration=duration or "full")
    cached = cache.get(key)
    if cached is not None:
        return cached
    path = validate_audio_path(file_path)
    y, _ = librosa.load(str(path), sr=sr, duration=duration, mono=True)
    cache.put(key, y, persist=False)  # waveforms too large for disk cache
    return y


# ─── CLAP Functions (cached) ─────────────────────────────────────────────────


def _unwrap_clap_output(output) -> torch.Tensor:
    """Extract tensor from CLAP model output (handles both old tensor and new BaseModelOutputWithPooling)."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unexpected CLAP output type: {type(output)}")


def _clap_audio_embedding(audio: np.ndarray, _audio_cache_key: str = "") -> torch.Tensor:
    """Get normalized CLAP audio embedding, using cache if key provided."""
    if _audio_cache_key:
        embed_key = f"{_audio_cache_key}:clap_embed"
        cached = cache.get(embed_key)
        if cached is not None:
            return cached.to(_device)

    with gpu_lock:
        model, processor = models.get_clap()
        device = next(model.parameters()).device
        audio_inputs = processor(audio=audio, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt", padding=True)
        audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
        with torch.no_grad():
            embed = _unwrap_clap_output(model.get_audio_features(**audio_inputs))
            embed = embed / embed.norm(dim=-1, keepdim=True)

    if _audio_cache_key:
        cache.put(embed_key, embed.cpu(), persist=True)
    return embed


def clap_classify(audio: np.ndarray, labels: list[str], top_k: int = 5, _audio_cache_key: str = "") -> list[dict]:
    """Zero-shot classification using CLAP."""
    # Check full classification cache
    if _audio_cache_key:
        labels_hash = hashlib.sha256("|".join(sorted(labels)).encode()).hexdigest()[:8]
        cls_key = f"{_audio_cache_key}:clap_classify:labels={labels_hash}:top_k={top_k}"
        cached = cache.get(cls_key)
        if cached is not None:
            return cached

    audio_embed = _clap_audio_embedding(audio, _audio_cache_key)

    with gpu_lock:
        model, processor = models.get_clap()
        device = next(model.parameters()).device
        text_inputs = processor(text=labels, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_embed = _unwrap_clap_output(model.get_text_features(**text_inputs))
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            similarity = (audio_embed @ text_embed.T).squeeze(0)
            probs = similarity.softmax(dim=-1)
            scores = probs.cpu().numpy().tolist()

    if isinstance(scores, float):
        scores = [scores]
    results = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    result = [{"label": l, "confidence": round(s, 4)} for l, s in results[:top_k]]

    if _audio_cache_key:
        cache.put(cls_key, result, persist=True)
    return result


def clap_similarity(audio: np.ndarray, query: str, _audio_cache_key: str = "") -> float:
    """Compute similarity between audio and a text query using CLAP."""
    audio_embed = _clap_audio_embedding(audio, _audio_cache_key)

    with gpu_lock:
        model, processor = models.get_clap()
        device = next(model.parameters()).device
        text_inputs = processor(text=[query], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_embed = _unwrap_clap_output(model.get_text_features(**text_inputs))
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            sim = (audio_embed @ text_embed.T).squeeze().item()
    return round(sim, 4)


# ─── Signal Analysis (cached) ────────────────────────────────────────────────


def get_signal_features(file_path: str, max_duration: float = 120.0) -> dict:
    """Extract signal-level features with librosa."""
    key = FileIdentifier.cache_key(file_path, "signal_features", max_duration=max_duration)
    cached = cache.get(key)
    if cached is not None:
        return cached
    y = load_audio(file_path, sr=LIBROSA_SR, duration=max_duration)
    result = signal_features_from_array(y, LIBROSA_SR)
    cache.put(key, result, persist=True)
    return result


def signal_features_from_array(y: np.ndarray, sr: int) -> dict:
    """Compute signal features from a numpy array."""
    duration = librosa.get_duration(y=y, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, key_confidence = estimate_key(chroma)

    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))

    rms_db = librosa.amplitude_to_db(rms)
    valid_db = rms_db[rms_db > -80]
    dynamic_range = float(np.max(rms_db) - np.min(valid_db)) if len(valid_db) > 0 else 0.0

    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    notes_per_second = len(onset_times) / duration if duration > 0 else 0

    return {
        "duration_seconds": round(duration, 2),
        "duration_formatted": format_seconds(duration),
        "tempo_bpm": round(tempo_val, 1),
        "estimated_key": key,
        "key_confidence": round(key_confidence, 3),
        "beat_count": len(beat_times),
        "spectral": {
            "centroid_hz": round(spectral_centroid, 1),
            "rolloff_hz": round(spectral_rolloff, 1),
            "bandwidth_hz": round(spectral_bandwidth, 1),
            "brightness": "bright" if spectral_centroid > 3000 else "warm" if spectral_centroid > 1500 else "dark",
        },
        "zero_crossing_rate": round(zero_crossing_rate, 4),
        "loudness": {
            "rms_mean": round(rms_mean, 4),
            "rms_max": round(rms_max, 4),
            "dynamic_range_db": round(dynamic_range, 1),
        },
        "notes_per_second": round(notes_per_second, 2),
        "onset_count": len(onset_times),
    }


# ─── Demucs Stem Separation (cached) ─────────────────────────────────────────


def separate_stems(file_path: str, max_duration: Optional[float] = None) -> dict[str, np.ndarray]:
    """Separate audio into stems using Demucs. Returns dict: stem name -> mono numpy array at DEMUCS_SR."""
    key = FileIdentifier.cache_key(file_path, "demucs", max_duration=max_duration or "full")
    cached = cache.get(key)
    if cached is not None:
        return cached

    from demucs.apply import apply_model

    path = validate_audio_path(file_path)

    wav, sr = torchaudio.load(str(path))

    if max_duration is not None:
        max_samples = int(max_duration * sr)
        wav = wav[:, :max_samples]

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    with gpu_lock:
        model = models.get_demucs()

        if sr != model.samplerate:
            wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)

        ref = wav.mean(0)
        wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)

        with torch.no_grad():
            sources = apply_model(
                model, wav_norm[None].to(_device),
                device=_device, split=True, overlap=0.25, progress=False,
            )

        source_names = model.sources

    sources = sources[0].cpu().numpy()

    stems = {}
    for i, name in enumerate(source_names):
        stems[name] = sources[i].mean(axis=0)

    cache.put(key, stems, persist=False)
    return stems


# ─── Whisper Transcription (cached + IDs) ─────────────────────────────────────


def _is_hallucination(text: str) -> bool:
    """Detect Whisper hallucination patterns."""
    t = text.strip().lower()
    if not t:
        return True
    # Repeated single character (EEEEEE, aaaaaaa, etc.)
    if len(t) > 5 and len(set(t.replace(" ", ""))) <= 2:
        return True
    # Known hallucination phrases on silence/hums
    hallucination_phrases = [
        "thank you for watching", "thanks for watching", "subscribe",
        "like and subscribe", "please subscribe", "see you next time",
        "thank you for listening", "thanks for listening",
        "i'll see you in the next", "bye bye",
    ]
    for phrase in hallucination_phrases:
        if phrase in t:
            return True
    # Excessive repetition of a word/short phrase
    words = t.split()
    if len(words) > 4:
        unique = set(words)
        if len(unique) <= 2:
            return True
    return False


def _clean_transcription(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Filter hallucinated chunks and reconstruct clean text."""
    clean = [c for c in chunks if not _is_hallucination(c["text"])]
    full_text = " ".join(c["text"] for c in clean if c["text"])
    return full_text, clean


def transcribe_audio_array(
    audio: np.ndarray, sr: int, language: Optional[str] = None,
    _file_path: str = "", _max_duration: float = 0,
) -> dict:
    """Transcribe audio using Whisper large-v3. Processes in chunks for long audio."""
    # Cache check
    if _file_path:
        key = FileIdentifier.cache_key(_file_path, "whisper_vocals", lang=language or "auto", max_duration=_max_duration)
        cached = cache.get(key)
        if cached is not None:
            return cached

    if sr != WHISPER_SR:
        audio = resample_np(audio, sr, WHISPER_SR)
    audio = audio.astype(np.float32)

    gen_kwargs = {"task": "transcribe"}
    if language:
        gen_kwargs["language"] = language

    total_samples = len(audio)
    chunk_samples = WHISPER_CHUNK_SEC * WHISPER_SR
    overlap_samples = 5 * WHISPER_SR  # 5 sec overlap

    # Short audio: single pass
    if total_samples <= chunk_samples * 1.5:
        with gpu_lock:
            pipe = models.get_whisper()
            result = pipe(
                {"raw": audio, "sampling_rate": WHISPER_SR},
                return_timestamps=True,
                generate_kwargs=gen_kwargs,
            )
        chunks = []
        for c in result.get("chunks", []):
            ts = c.get("timestamp", (None, None))
            chunks.append({
                "text": c["text"].strip(),
                "start": round(ts[0], 2) if ts[0] is not None else None,
                "end": round(ts[1], 2) if ts[1] is not None else None,
            })
        full_text, clean_chunks = _clean_transcription(chunks)

        # Add IDs
        if _file_path:
            for idx, chunk in enumerate(clean_chunks):
                start = chunk.get("start") or 0
                end = chunk.get("end") or 0
                chunk["id"] = FileIdentifier.whisper_id(_file_path, idx, start, end)

        output = {"full_text": full_text, "chunks": clean_chunks}
        if _file_path:
            cache.put(key, output, persist=True)
        return output

    # Long audio: chunked processing
    all_chunks = []
    offset = 0
    with gpu_lock:
        pipe = models.get_whisper()
        while offset < total_samples:
            end = min(offset + chunk_samples, total_samples)
            chunk_audio = audio[offset:end]
            time_offset = offset / WHISPER_SR

            result = pipe(
                {"raw": chunk_audio, "sampling_rate": WHISPER_SR},
                return_timestamps=True,
                generate_kwargs=gen_kwargs,
            )

            for c in result.get("chunks", []):
                ts = c.get("timestamp", (None, None))
                all_chunks.append({
                    "text": c["text"].strip(),
                    "start": round(ts[0] + time_offset, 2) if ts[0] is not None else None,
                    "end": round(ts[1] + time_offset, 2) if ts[1] is not None else None,
                })

            offset += chunk_samples - overlap_samples

    # Deduplicate overlapping chunks
    deduped = []
    for c in all_chunks:
        if not deduped or c["text"] != deduped[-1]["text"]:
            deduped.append(c)

    full_text, clean_chunks = _clean_transcription(deduped)

    # Add IDs
    if _file_path:
        for idx, chunk in enumerate(clean_chunks):
            start = chunk.get("start") or 0
            end = chunk.get("end") or 0
            chunk["id"] = FileIdentifier.whisper_id(_file_path, idx, start, end)

    output = {"full_text": full_text, "chunks": clean_chunks}
    if _file_path:
        cache.put(key, output, persist=True)
    return output


# ─── Chord Detection (with IDs) ──────────────────────────────────────────────


def detect_chords_from_array(
    y: np.ndarray, sr: int, hop_length: int = 4096,
    beat_times: Optional[np.ndarray] = None, _file_path: str = "",
) -> list[dict]:
    """Detect chord progression with beat-quantized smoothing and expanded templates."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(range(chroma.shape[1]), sr=sr, hop_length=hop_length)

    # Per-frame chord detection
    raw_chords = []
    for i in range(chroma.shape[1]):
        frame = chroma[:, i]
        if frame.max() < 0.1:
            raw_chords.append(("N", 0.0))
            continue
        frame_norm = frame / (frame.max() + 1e-8)
        best_chord = "N"
        best_score = -1
        for chord_name, template in CHORD_TEMPLATES.items():
            score = float(np.dot(frame_norm, template))
            if score > best_score:
                best_score = score
                best_chord = chord_name
        raw_chords.append((best_chord, best_score))

    # Beat quantization: snap to beat grid if available
    if beat_times is not None and len(beat_times) > 1:
        beat_interval = float(np.median(np.diff(beat_times)))
    else:
        # Estimate beat interval from tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        beat_interval = 60.0 / max(tempo_val, 30)

    # Minimum hold: half a beat
    min_hold_frames = max(1, int((beat_interval * 0.5) / (hop_length / sr)))

    # Smooth: majority vote over min_hold windows
    smoothed = []
    i = 0
    while i < len(raw_chords):
        window_end = min(i + min_hold_frames, len(raw_chords))
        window = raw_chords[i:window_end]
        # Count chord occurrences in window (exclude "N")
        counts = {}
        for chord, score in window:
            if chord != "N":
                counts[chord] = counts.get(chord, 0) + score
        if counts:
            best = max(counts, key=counts.get)
            avg_score = counts[best] / sum(1 for c, _ in window if c == best)
        else:
            best = "N"
            avg_score = 0
        smoothed.append((best, avg_score, float(times[i])))
        i = window_end

    # Collapse consecutive same chords
    chord_seq = []
    prev = None
    for chord, score, t in smoothed:
        if chord != "N" and chord != prev:
            chord_seq.append({
                "chord": chord,
                "time": round(t, 2),
                "time_formatted": format_seconds(t),
                "confidence": round(score, 3),
            })
            prev = chord

    # Add IDs
    if _file_path:
        for idx, c in enumerate(chord_seq):
            c["id"] = FileIdentifier.chord_id(_file_path, idx, c["time"])

    return chord_seq


# ─── Song Structure Detection (with IDs) ─────────────────────────────────────


def detect_sections_from_array(y: np.ndarray, sr: int, _file_path: str = "") -> list[dict]:
    """Detect song sections with repetition-based labeling."""
    duration = librosa.get_duration(y=y, sr=sr)
    hop = 512

    # Compute features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]

    features = np.vstack([mfcc, chroma])

    # Detect boundaries using energy changes
    rms_smooth = np.convolve(rms, np.ones(40) / 40, mode="same")
    diff = np.abs(np.diff(rms_smooth))
    threshold = np.percentile(diff, 92)
    boundaries = np.where(diff > threshold)[0]

    # Ensure min 4 seconds between boundaries
    min_gap = int(4 * sr / hop)
    filtered = [0]
    for b in boundaries:
        if b - filtered[-1] > min_gap:
            filtered.append(b)
    bound_frames = np.array(filtered)
    bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=hop)

    # Build sections with features
    sections = []
    for i in range(len(bound_times)):
        start_t = float(bound_times[i])
        end_t = float(bound_times[i + 1]) if i + 1 < len(bound_times) else duration

        if end_t - start_t < 1.0:
            continue

        start_f = bound_frames[i]
        end_f = bound_frames[i + 1] if i + 1 < len(bound_frames) else features.shape[1]
        end_f = min(end_f, features.shape[1])

        if end_f <= start_f:
            continue

        section_features = features[:, start_f:end_f].mean(axis=1)
        start_sample = int(start_t * sr)
        end_sample = min(int(end_t * sr), len(y))
        section_audio = y[start_sample:end_sample]
        energy = float(np.sqrt(np.mean(section_audio ** 2))) if len(section_audio) > 0 else 0

        sections.append({
            "start": round(start_t, 2),
            "end": round(end_t, 2),
            "start_formatted": format_seconds(start_t),
            "end_formatted": format_seconds(end_t),
            "duration": round(end_t - start_t, 2),
            "energy": round(energy, 4),
            "_features": section_features,
        })

    if not sections:
        return []

    # Compute self-similarity between sections
    feat_matrix = np.array([s["_features"] for s in sections])
    norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    feat_norm = feat_matrix / norms
    sim_matrix = feat_norm @ feat_norm.T

    # Cluster similar sections (assign letter labels)
    labels = [None] * len(sections)
    current_label = 0
    label_map = "ABCDEFGHIJKLMNOP"
    for i in range(len(sections)):
        if labels[i] is not None:
            continue
        letter = label_map[current_label % len(label_map)]
        labels[i] = letter
        for j in range(i + 1, len(sections)):
            if labels[j] is None and sim_matrix[i][j] > 0.75:
                labels[j] = letter
        current_label += 1

    # Map letters to song part names based on energy, position, repetition
    energies = [s["energy"] for s in sections]
    max_e = max(energies) if energies else 1
    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1

    # Find which label is the "chorus" (most repeated loud section)
    chorus_label = None
    best_chorus_score = -1
    for l in set(labels):
        indices = [i for i, x in enumerate(labels) if x == l]
        avg_energy = np.mean([sections[i]["energy"] for i in indices])
        count = len(indices)
        score = avg_energy * count  # louder + more repeated = more likely chorus
        if score > best_chorus_score and count >= 2:
            best_chorus_score = score
            chorus_label = l

    for i, s in enumerate(sections):
        ratio = s["energy"] / max_e if max_e > 0 else 0
        is_first = (i == 0)
        is_last = (i == len(sections) - 1)
        label = labels[i]

        if is_first and ratio < 0.5:
            part_name = "intro"
        elif is_last and ratio < 0.4:
            part_name = "outro"
        elif label == chorus_label:
            part_name = "chorus"
        elif ratio < 0.25:
            part_name = "breakdown"
        elif ratio < 0.55:
            part_name = "verse"
        elif ratio < 0.75:
            part_name = "pre-chorus"
        else:
            # Loud but not chorus — could be bridge or drop
            if label_counts.get(label, 0) == 1:
                part_name = "bridge"
            else:
                part_name = "chorus"

        s["section_label"] = f"{label}"
        s["part"] = part_name
        del s["_features"]

    # Add IDs
    if _file_path:
        for idx, s in enumerate(sections):
            s["id"] = FileIdentifier.section_id(_file_path, idx, s["start"], s["end"])

    return sections


# ─── Vocal Analysis ──────────────────────────────────────────────────────────


def analyze_vocals(vocal_audio: np.ndarray, sr: int, _audio_cache_key: str = "") -> dict:
    """Analyze isolated vocal stem for pitch, range, and style."""
    if sr != LIBROSA_SR:
        y = resample_np(vocal_audio, sr, LIBROSA_SR)
        sr_use = LIBROSA_SR
    else:
        y = vocal_audio
        sr_use = sr

    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    if rms_mean < 0.005:
        return {"has_vocals": False, "note": "No significant vocal content detected"}

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr_use
    )

    voiced_f0 = f0[~np.isnan(f0)]

    if len(voiced_f0) < 10:
        return {"has_vocals": True, "pitch_tracking": "insufficient voiced frames"}

    low_hz = float(np.percentile(voiced_f0, 5))
    high_hz = float(np.percentile(voiced_f0, 95))
    median_hz = float(np.median(voiced_f0))

    f0_diff = np.diff(voiced_f0)
    vibrato_rate = float(np.mean(np.abs(f0_diff)))

    rms_db = librosa.amplitude_to_db(rms)
    vocal_dynamic_range = float(np.max(rms_db) - np.percentile(rms_db[rms_db > -60], 10)) if np.any(rms_db > -60) else 0

    vocal_clap = resample_np(vocal_audio, sr, CLAP_SAMPLE_RATE)
    style = clap_classify(vocal_clap, VOCAL_STYLE_LABELS, top_k=5, _audio_cache_key=_audio_cache_key)

    return {
        "has_vocals": True,
        "pitch_range": {
            "low": hz_to_note(low_hz), "low_hz": round(low_hz, 1),
            "high": hz_to_note(high_hz), "high_hz": round(high_hz, 1),
            "median": hz_to_note(median_hz), "median_hz": round(median_hz, 1),
        },
        "vibrato_intensity": round(vibrato_rate, 2),
        "dynamic_range_db": round(vocal_dynamic_range, 1),
        "vocal_style": style,
        "voiced_percentage": round(100 * len(voiced_f0) / len(f0), 1),
    }


# ─── Rhythm Analysis ─────────────────────────────────────────────────────────


def analyze_rhythm(drum_audio: np.ndarray, sr: int, _audio_cache_key: str = "") -> dict:
    """Analyze isolated drum stem for rhythm patterns."""
    if sr != LIBROSA_SR:
        y = resample_np(drum_audio, sr, LIBROSA_SR)
        sr_use = LIBROSA_SR
    else:
        y = drum_audio
        sr_use = sr

    rms_mean = float(np.mean(librosa.feature.rms(y=y)[0]))
    if rms_mean < 0.003:
        return {"has_drums": False}

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr_use)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    onsets = librosa.onset.onset_detect(y=y, sr=sr_use)
    onset_times = librosa.frames_to_time(onsets, sr=sr_use)
    beat_times = librosa.frames_to_time(beats, sr=sr_use)

    if len(beat_times) > 2:
        intervals = np.diff(beat_times)
        regularity = 1.0 - float(np.std(intervals) / (np.mean(intervals) + 1e-8))
    else:
        regularity = 0.0

    hits_per_beat = len(onset_times) / max(len(beat_times), 1)

    drum_clap = resample_np(drum_audio, sr, CLAP_SAMPLE_RATE)
    rhythm_labels = [
        "four on the floor kick drum pattern", "trap style hi-hat pattern",
        "boom bap drum pattern", "breakbeat drum pattern",
        "acoustic live drum performance", "electronic programmed drums",
        "slow tempo drums", "fast tempo drums", "syncopated rhythm", "straight rhythm",
    ]
    rhythm_style = clap_classify(drum_clap, rhythm_labels, top_k=4, _audio_cache_key=_audio_cache_key)

    return {
        "has_drums": True, "tempo_bpm": round(tempo_val, 1),
        "beat_count": len(beat_times), "onset_count": len(onset_times),
        "hits_per_beat": round(hits_per_beat, 1),
        "beat_regularity": round(regularity, 3),
        "groove": "tight/quantized" if regularity > 0.9 else "loose/human" if regularity > 0.7 else "very loose/free",
        "rhythm_style": rhythm_style,
    }


# ─── Melody Extraction (with IDs) ────────────────────────────────────────────


def extract_melody_from_array(y: np.ndarray, sr: int, _file_path: str = "") -> dict:
    """Extract melodic content: pitch contour, phrases, and hooks."""
    if sr != LIBROSA_SR:
        y = resample_np(y, sr, LIBROSA_SR)
        sr = LIBROSA_SR

    rms_mean = float(np.mean(librosa.feature.rms(y=y)[0]))
    if rms_mean < 0.003:
        return {"has_melody": False}

    # Pitch tracking
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )

    if f0 is None or np.all(np.isnan(f0)):
        return {"has_melody": False}

    times = librosa.times_like(f0, sr=sr)

    # Extract pitched segments (phrases)
    phrases = []
    current_phrase = []
    for i in range(len(f0)):
        if not np.isnan(f0[i]):
            midi = int(round(69 + 12 * np.log2(f0[i] / 440.0)))
            note_name = hz_to_note(f0[i])
            current_phrase.append({
                "time": round(float(times[i]), 3),
                "midi": midi,
                "note": note_name,
                "hz": round(float(f0[i]), 1),
            })
        else:
            if len(current_phrase) >= 4:  # minimum phrase length
                phrases.append(current_phrase)
            current_phrase = []
    if len(current_phrase) >= 4:
        phrases.append(current_phrase)

    if not phrases:
        return {"has_melody": True, "phrases": [], "note": "Pitch detected but no clear phrases"}

    # Detect repeated melodic patterns (hooks)
    # Convert phrases to MIDI interval sequences for comparison
    def phrase_to_intervals(phrase):
        return tuple(phrase[i+1]["midi"] - phrase[i]["midi"] for i in range(min(len(phrase)-1, 8)))

    interval_seqs = [phrase_to_intervals(p) for p in phrases]
    hooks = []
    seen = set()
    for i, seq_i in enumerate(interval_seqs):
        if len(seq_i) < 3:
            continue
        for j in range(i + 1, len(interval_seqs)):
            seq_j = interval_seqs[j]
            # Compare first N intervals
            compare_len = min(len(seq_i), len(seq_j), 6)
            if compare_len < 3:
                continue
            matches = sum(1 for a, b in zip(seq_i[:compare_len], seq_j[:compare_len]) if abs(a - b) <= 1)
            if matches >= compare_len * 0.7:
                key = (i, j)
                if key not in seen:
                    seen.add(key)
                    hooks.append({
                        "phrase_indices": [i, j],
                        "times": [
                            format_seconds(phrases[i][0]["time"]),
                            format_seconds(phrases[j][0]["time"]),
                        ],
                        "similarity": round(matches / compare_len, 2),
                    })

    # Summarize
    voiced_f0 = f0[~np.isnan(f0)]

    phrase_summaries = []
    for p_idx, p in enumerate(phrases[:12]):  # max 12 phrases
        summary = {
            "start": format_seconds(p[0]["time"]),
            "end": format_seconds(p[-1]["time"]),
            "duration": round(p[-1]["time"] - p[0]["time"], 2),
            "notes": [n["note"] for n in p[:16]],  # first 16 notes
            "starting_note": p[0]["note"],
        }
        if _file_path:
            summary["id"] = FileIdentifier.phrase_id(_file_path, p_idx, p[0]["time"])
        phrase_summaries.append(summary)

    return {
        "has_melody": True,
        "phrase_count": len(phrases),
        "pitch_range": {
            "low": hz_to_note(float(np.percentile(voiced_f0, 5))),
            "high": hz_to_note(float(np.percentile(voiced_f0, 95))),
            "median": hz_to_note(float(np.median(voiced_f0))),
        },
        "phrases": phrase_summaries,
        "repeated_hooks": hooks[:5],
        "hook_detected": len(hooks) > 0,
    }


# ─── MCP Server ──────────────────────────────────────────────────────────────

mcp = FastMCP(
    "audio-understanding",
    instructions=(
        "Audio analysis tools for understanding music and sound. "
        "Use analyze_audio for a complete breakdown. Use audio_query "
        "for specific yes/no or descriptive questions about audio content."
    ),
)


# ─── Original Tools (enhanced) ───────────────────────────────────────────────


@mcp.tool()
def analyze_audio(file_path: str, max_duration: float = 60.0) -> str:
    """
    Complete analysis of an audio file. Returns tempo, key, spectral features,
    detected genre, instruments, mood, and audio quality assessment.

    Args:
        file_path: Absolute or ~ path to MP3, WAV, FLAC, or OGG file.
        max_duration: Max seconds to analyze (default 60). Longer = slower but more accurate.
    """
    max_duration = min(max_duration, MAX_DURATION)
    clap_key = FileIdentifier.cache_key(file_path, "clap_audio", sr=CLAP_SAMPLE_RATE, max_duration=max_duration)

    # Parallel: signal features (CPU) while loading CLAP audio
    future_signal = _cpu_pool.submit(get_signal_features, file_path, max_duration)
    audio_clap = load_audio(file_path, sr=CLAP_SAMPLE_RATE, duration=max_duration)

    genres = clap_classify(audio_clap, GENRE_LABELS, top_k=5, _audio_cache_key=clap_key)
    instruments = clap_classify(audio_clap, INSTRUMENT_LABELS, top_k=8, _audio_cache_key=clap_key)
    moods = clap_classify(audio_clap, MOOD_LABELS, top_k=5, _audio_cache_key=clap_key)

    signal = future_signal.result()
    return _tool_output({
        "file": file_path, "signal_analysis": signal,
        "genre": genres, "instruments": instruments, "mood": moods,
    }, "analyze_audio")


@mcp.tool()
def classify_genre(file_path: str, max_duration: float = 30.0) -> str:
    """
    Classify the genre of an audio file. Returns ranked genre predictions.

    Args:
        file_path: Path to the audio file.
        max_duration: Seconds to analyze (default 30).
    """
    clap_key = FileIdentifier.cache_key(file_path, "clap_audio", sr=CLAP_SAMPLE_RATE, max_duration=max_duration)
    audio = load_audio(file_path, sr=CLAP_SAMPLE_RATE, duration=max_duration)
    return _tool_output({
        "file": file_path,
        "genres": clap_classify(audio, GENRE_LABELS, top_k=10, _audio_cache_key=clap_key),
    }, "classify_genre")


@mcp.tool()
def classify_mood(file_path: str, max_duration: float = 30.0) -> str:
    """
    Detect the mood and emotional character of an audio file.

    Args:
        file_path: Path to the audio file.
        max_duration: Seconds to analyze (default 30).
    """
    clap_key = FileIdentifier.cache_key(file_path, "clap_audio", sr=CLAP_SAMPLE_RATE, max_duration=max_duration)
    audio = load_audio(file_path, sr=CLAP_SAMPLE_RATE, duration=max_duration)
    return _tool_output({
        "file": file_path,
        "moods": clap_classify(audio, MOOD_LABELS, top_k=10, _audio_cache_key=clap_key),
    }, "classify_mood")


@mcp.tool()
def identify_instruments(file_path: str, max_duration: float = 30.0) -> str:
    """
    Identify instruments and sound sources in an audio file.

    Args:
        file_path: Path to the audio file.
        max_duration: Seconds to analyze (default 30).
    """
    clap_key = FileIdentifier.cache_key(file_path, "clap_audio", sr=CLAP_SAMPLE_RATE, max_duration=max_duration)
    audio = load_audio(file_path, sr=CLAP_SAMPLE_RATE, duration=max_duration)
    return _tool_output({
        "file": file_path,
        "instruments": clap_classify(audio, INSTRUMENT_LABELS, top_k=10, _audio_cache_key=clap_key),
    }, "identify_instruments")


@mcp.tool()
def audio_query(file_path: str, query: str, max_duration: float = 30.0) -> str:
    """
    Ask a free-form question about an audio file. Uses CLAP to compare the
    audio against your query text. Good for questions like:
    - "Is there a guitar solo?"
    - "Does this sound like jazz?"
    - "Is there singing?"
    - "Is this a live recording?"

    Also compares against the negation for a yes/no style answer.

    Args:
        file_path: Path to the audio file.
        query: Your question or description to match against.
        max_duration: Seconds to analyze.
    """
    clap_key = FileIdentifier.cache_key(file_path, "clap_audio", sr=CLAP_SAMPLE_RATE, max_duration=max_duration)
    audio = load_audio(file_path, sr=CLAP_SAMPLE_RATE, duration=max_duration)
    positive_score = clap_similarity(audio, query, _audio_cache_key=clap_key)
    negation = f"no {query}" if not query.startswith("no ") else query[3:]
    negative_score = clap_similarity(audio, negation, _audio_cache_key=clap_key)
    confidence = round(abs(positive_score - negative_score), 4)
    return _tool_output({
        "file": file_path, "query": query,
        "similarity_score": positive_score, "negation_score": negative_score,
        "likely_answer": "yes" if positive_score > negative_score else "no",
        "confidence": confidence,
        "reliability": "high" if confidence > 0.05 else "low (scores too close)",
    }, "audio_query")


@mcp.tool()
def compare_audio(file_path_a: str, file_path_b: str, max_duration: float = 30.0) -> str:
    """
    Compare two audio files. Returns how similar they sound, plus
    individual analysis of tempo, key, and spectral characteristics.

    Args:
        file_path_a: Path to the first audio file.
        file_path_b: Path to the second audio file.
        max_duration: Seconds to analyze from each file.
    """
    clap_key_a = FileIdentifier.cache_key(file_path_a, "clap_audio", sr=CLAP_SAMPLE_RATE, max_duration=max_duration)
    clap_key_b = FileIdentifier.cache_key(file_path_b, "clap_audio", sr=CLAP_SAMPLE_RATE, max_duration=max_duration)

    # Parallel: load both files + signal features
    future_audio_a = _cpu_pool.submit(load_audio, file_path_a, CLAP_SAMPLE_RATE, max_duration)
    future_audio_b = _cpu_pool.submit(load_audio, file_path_b, CLAP_SAMPLE_RATE, max_duration)
    future_sig_a = _cpu_pool.submit(get_signal_features, file_path_a, max_duration)
    future_sig_b = _cpu_pool.submit(get_signal_features, file_path_b, max_duration)

    audio_a = future_audio_a.result()
    audio_b = future_audio_b.result()

    # CLAP embeddings (GPU, sequential)
    embed_a = _clap_audio_embedding(audio_a, clap_key_a)
    embed_b = _clap_audio_embedding(audio_b, clap_key_b)
    with torch.no_grad():
        similarity = (embed_a @ embed_b.T).squeeze().item()

    signal_a = future_sig_a.result()
    signal_b = future_sig_b.result()

    return _tool_output({
        "file_a": file_path_a, "file_b": file_path_b,
        "semantic_similarity": round(similarity, 4),
        "similarity_description": (
            "very similar" if similarity > 0.85 else "similar" if similarity > 0.7
            else "somewhat similar" if similarity > 0.5 else "different" if similarity > 0.3
            else "very different"
        ),
        "tempo_difference_bpm": round(abs(signal_a["tempo_bpm"] - signal_b["tempo_bpm"]), 1),
        "same_key": signal_a["estimated_key"] == signal_b["estimated_key"],
        "file_a_summary": {"tempo": signal_a["tempo_bpm"], "key": signal_a["estimated_key"], "brightness": signal_a["spectral"]["brightness"], "duration": signal_a["duration_formatted"]},
        "file_b_summary": {"tempo": signal_b["tempo_bpm"], "key": signal_b["estimated_key"], "brightness": signal_b["spectral"]["brightness"], "duration": signal_b["duration_formatted"]},
    }, "compare_audio")


@mcp.tool()
def get_audio_info(file_path: str) -> str:
    """
    Get basic metadata about an audio file (duration, format, sample rate).
    Fast — does not run any ML analysis.

    Args:
        file_path: Path to the audio file.
    """
    path = validate_audio_path(file_path)
    info = sf.info(str(path))
    file_size = path.stat().st_size
    return _tool_output({
        "file": str(path), "format": info.format, "subtype": info.subtype,
        "sample_rate": info.samplerate, "channels": info.channels,
        "frames": info.frames, "duration_seconds": round(info.duration, 2),
        "duration_formatted": format_seconds(info.duration),
        "file_size_mb": round(file_size / (1024 * 1024), 2),
    }, "get_audio_info")


@mcp.tool()
def analyze_section(file_path: str, start_seconds: float, end_seconds: float) -> str:
    """
    Analyze a specific section of an audio file. Useful for examining
    intros, solos, verses, choruses, or any specific time range.

    Args:
        file_path: Path to the audio file.
        start_seconds: Start time in seconds.
        end_seconds: End time in seconds.
    """
    validate_audio_path(file_path)
    duration = end_seconds - start_seconds
    if duration <= 0:
        raise ValueError("end_seconds must be greater than start_seconds")
    if duration > 120:
        raise ValueError("Section too long. Max 120 seconds per analysis.")

    section_clap_key = FileIdentifier.cache_key(
        file_path, "clap_section", sr=CLAP_SAMPLE_RATE, start=start_seconds, end=end_seconds,
    )

    y_librosa, _ = librosa.load(str(Path(file_path).expanduser()), sr=LIBROSA_SR, offset=start_seconds, duration=duration)
    y_clap, _ = librosa.load(str(Path(file_path).expanduser()), sr=CLAP_SAMPLE_RATE, offset=start_seconds, duration=duration)
    actual_dur = librosa.get_duration(y=y_librosa, sr=LIBROSA_SR)
    tempo, beats = librosa.beat.beat_track(y=y_librosa, sr=LIBROSA_SR)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    chroma = librosa.feature.chroma_cqt(y=y_librosa, sr=LIBROSA_SR)
    key, key_conf = estimate_key(chroma)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_librosa, sr=LIBROSA_SR)))
    rms = float(np.mean(librosa.feature.rms(y=y_librosa)))
    return _tool_output({
        "file": file_path,
        "section": f"{format_seconds(start_seconds)} - {format_seconds(end_seconds)}",
        "actual_duration": round(actual_dur, 2), "tempo_bpm": round(tempo_val, 1),
        "key": key, "spectral_centroid_hz": round(centroid, 1), "rms_loudness": round(rms, 4),
        "genres": clap_classify(y_clap, GENRE_LABELS, top_k=3, _audio_cache_key=section_clap_key),
        "instruments": clap_classify(y_clap, INSTRUMENT_LABELS, top_k=5, _audio_cache_key=section_clap_key),
        "moods": clap_classify(y_clap, MOOD_LABELS, top_k=3, _audio_cache_key=section_clap_key),
    }, "analyze_section")


# ─── Deep Listening Tools ────────────────────────────────────────────────────


@mcp.tool()
def transcribe_lyrics(file_path: str, max_duration: float = 300.0, language: str = "") -> str:
    """
    Transcribe lyrics/vocals from a music track. Uses Demucs to isolate
    the vocal stem first, then Whisper to transcribe. Returns full text
    plus word-level timestamps.

    Args:
        file_path: Path to the audio file.
        max_duration: Max seconds to process (default 300 = 5 min).
        language: Language code (e.g. "english", "spanish"). Empty = auto-detect.
    """
    max_duration = min(max_duration, MAX_DURATION)
    stems = separate_stems(file_path, max_duration=max_duration)
    with gpu_lock:
        models.unload("demucs")  # free VRAM for Whisper

    vocals = stems.get("vocals")
    if vocals is None:
        return _tool_output({"error": "Could not separate vocals"}, "transcribe_lyrics")

    transcription = transcribe_audio_array(
        vocals, DEMUCS_SR, language=language or None,
        _file_path=file_path, _max_duration=max_duration,
    )
    return _tool_output({
        "file": file_path,
        "lyrics": transcription["full_text"],
        "timestamped_lyrics": transcription["chunks"],
    }, "transcribe_lyrics")


@mcp.tool()
def analyze_stems(file_path: str, max_duration: float = 120.0) -> str:
    """
    Separate a track into stems (vocals, drums, bass, other) using Demucs
    and analyze each stem individually. Gives much more accurate instrument
    and content detection than analyzing the full mix.

    Args:
        file_path: Path to the audio file.
        max_duration: Max seconds to process (default 120).
    """
    max_duration = min(max_duration, MAX_DURATION)
    stems = separate_stems(file_path, max_duration=max_duration)
    with gpu_lock:
        models.unload("demucs")  # free VRAM for CLAP

    result = {"file": file_path, "stems": {}}
    for stem_name, stem_audio in stems.items():
        stem_librosa = resample_np(stem_audio, DEMUCS_SR, LIBROSA_SR)
        stem_signal = signal_features_from_array(stem_librosa, LIBROSA_SR)
        stem_clap = resample_np(stem_audio, DEMUCS_SR, CLAP_SAMPLE_RATE)
        stem_cache_key = FileIdentifier.cache_key(file_path, "clap_stem", stem=stem_name, max_duration=max_duration)

        if stem_name == "vocals":
            clap_labels = VOCAL_STYLE_LABELS
        elif stem_name == "drums":
            clap_labels = ["acoustic drums", "electronic drums", "trap hi-hats", "kick drum", "snare drum", "cymbals", "percussion", "four on the floor beat", "breakbeat", "boom bap drums"]
        elif stem_name == "bass":
            clap_labels = ["electric bass guitar", "acoustic bass", "synthesizer bass", "808 bass", "sub bass", "slap bass", "fingerstyle bass", "distorted bass", "clean bass tone", "deep bass"]
        else:
            clap_labels = INSTRUMENT_LABELS

        result["stems"][stem_name] = {
            "id": FileIdentifier.stem_id(file_path, stem_name),
            "signal": stem_signal,
            "classification": clap_classify(stem_clap, clap_labels, top_k=5, _audio_cache_key=stem_cache_key),
        }

    return _tool_output(result, "analyze_stems")


@mcp.tool()
def detect_chords(file_path: str, max_duration: float = 120.0) -> str:
    """
    Detect the chord progression of a track. Uses Demucs to isolate the
    harmonic content (removes drums) for cleaner chord detection.

    Args:
        file_path: Path to the audio file.
        max_duration: Max seconds to analyze.
    """
    max_duration = min(max_duration, MAX_DURATION)
    stems = separate_stems(file_path, max_duration=max_duration)
    with gpu_lock:
        models.unload("demucs")

    harmonic = stems.get("other", np.zeros(1)) + stems.get("bass", np.zeros(1))
    harmonic_lr = resample_np(harmonic, DEMUCS_SR, LIBROSA_SR)

    chords = detect_chords_from_array(harmonic_lr, LIBROSA_SR, _file_path=file_path)

    chord_counts = {}
    for c in chords:
        chord_counts[c["chord"]] = chord_counts.get(c["chord"], 0) + 1
    top_chords = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)

    return _tool_output({
        "file": file_path,
        "chord_progression": chords,
        "most_used_chords": [{"chord": c, "occurrences": n} for c, n in top_chords[:8]],
        "total_chord_changes": len(chords),
    }, "detect_chords")


@mcp.tool()
def get_song_structure(file_path: str, max_duration: float = 300.0) -> str:
    """
    Detect the structural sections of a song (intro, verse, chorus, etc.)
    based on energy and spectral changes.

    Args:
        file_path: Path to the audio file.
        max_duration: Max seconds to analyze.
    """
    max_duration = min(max_duration, MAX_DURATION)
    y = load_audio(file_path, sr=LIBROSA_SR, duration=max_duration)
    sections = detect_sections_from_array(y, LIBROSA_SR, _file_path=file_path)
    return _tool_output({
        "file": file_path,
        "total_duration": format_seconds(librosa.get_duration(y=y, sr=LIBROSA_SR)),
        "sections": sections,
        "section_count": len(sections),
    }, "get_song_structure")


@mcp.tool()
def deep_listen(file_path: str, max_duration: float = 120.0) -> str:
    """
    The most comprehensive analysis tool. Separates the track into stems,
    transcribes lyrics, detects chords, analyzes song structure, examines
    vocal characteristics, and classifies rhythm patterns. This gives your
    agent the closest experience to actually hearing the music.

    Slower than other tools (30-90 seconds) but returns everything.

    Args:
        file_path: Absolute path to the audio file.
        max_duration: Max seconds to process (default 120). Full songs up to 5 min supported.
    """
    max_duration = min(max_duration, MAX_DURATION)
    result = {"file": file_path}
    clap_key = FileIdentifier.cache_key(file_path, "clap_audio", sr=CLAP_SAMPLE_RATE, max_duration=max_duration)

    # Phase 1: Load once at CLAP SR (highest), resample for librosa
    audio_clap = load_audio(file_path, sr=CLAP_SAMPLE_RATE, duration=max_duration)
    y_librosa = resample_np(audio_clap, CLAP_SAMPLE_RATE, LIBROSA_SR)

    # Phase 2: CLAP full mix (GPU) || signal_features (CPU thread)
    future_signal = _cpu_pool.submit(signal_features_from_array, y_librosa, LIBROSA_SR)
    result["genre"] = clap_classify(audio_clap, GENRE_LABELS, top_k=5, _audio_cache_key=clap_key)
    result["mood"] = clap_classify(audio_clap, MOOD_LABELS, top_k=5, _audio_cache_key=clap_key)
    signal = future_signal.result()
    result["signal_analysis"] = signal

    # Phase 3: Demucs (GPU) — unload CLAP first
    with gpu_lock:
        models.unload("clap")
    stems = separate_stems(file_path, max_duration=max_duration)

    # Phase 4: CLAP per-stem classification (GPU) — unload Demucs first
    with gpu_lock:
        models.unload("demucs")
    stem_analysis = {}
    for stem_name, stem_audio in stems.items():
        stem_rms = float(np.sqrt(np.mean(stem_audio ** 2)))
        is_present = stem_rms > 0.005

        if not is_present:
            # Skip GPU classification for silent/near-silent stems
            stem_analysis[stem_name] = {
                "id": FileIdentifier.stem_id(file_path, stem_name),
                "energy": round(stem_rms, 4),
                "present": False,
                "classification": [],
            }
            continue

        stem_clap = resample_np(stem_audio, DEMUCS_SR, CLAP_SAMPLE_RATE)
        stem_cache_key = FileIdentifier.cache_key(file_path, "clap_stem", stem=stem_name, max_duration=max_duration)

        if stem_name == "vocals":
            labels = VOCAL_STYLE_LABELS
        elif stem_name == "drums":
            labels = ["acoustic drums", "electronic drums", "trap hi-hats", "kick drum", "snare drum", "cymbals", "percussion", "boom bap drums", "breakbeat"]
        elif stem_name == "bass":
            labels = ["electric bass guitar", "acoustic bass", "synthesizer bass", "808 bass", "sub bass", "slap bass", "deep bass"]
        else:
            labels = INSTRUMENT_LABELS

        stem_analysis[stem_name] = {
            "id": FileIdentifier.stem_id(file_path, stem_name),
            "energy": round(stem_rms, 4),
            "present": True,
            "classification": clap_classify(stem_clap, labels, top_k=4, _audio_cache_key=stem_cache_key),
        }
    result["stems"] = stem_analysis

    # Phase 5: Whisper (GPU) || chords + structure + melody (CPU threads)
    # Launch CPU tasks first
    harmonic = stems.get("other", np.zeros(1)) + stems.get("bass", np.zeros(1))
    harmonic_lr = resample_np(harmonic, DEMUCS_SR, LIBROSA_SR)
    other = stems.get("other")

    future_chords = _cpu_pool.submit(detect_chords_from_array, harmonic_lr, LIBROSA_SR, 4096, None, file_path)
    future_sections = _cpu_pool.submit(detect_sections_from_array, y_librosa, LIBROSA_SR, file_path)
    future_melody = _cpu_pool.submit(extract_melody_from_array, other, DEMUCS_SR, file_path) if other is not None else None

    # GPU: Whisper transcription
    vocals = stems.get("vocals")
    if vocals is not None and stem_analysis.get("vocals", {}).get("present", False):
        try:
            with gpu_lock:
                models.unload("clap")
            transcription = transcribe_audio_array(
                vocals, DEMUCS_SR,
                _file_path=file_path, _max_duration=max_duration,
            )
            result["lyrics"] = {"full_text": transcription["full_text"], "timestamped": transcription["chunks"]}
        except Exception as e:
            result["lyrics"] = {"error": str(e)}
    else:
        result["lyrics"] = {"full_text": "", "note": "No vocals detected"}

    # Collect CPU results
    try:
        chords = future_chords.result()
        chord_counts = {}
        for c in chords:
            chord_counts[c["chord"]] = chord_counts.get(c["chord"], 0) + 1
        top_chords = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)
        result["chords"] = {"progression_sample": chords[:20], "most_used": [{"chord": c, "count": n} for c, n in top_chords[:6]]}
    except Exception as e:
        result["chords"] = {"error": str(e)}

    try:
        result["song_structure"] = future_sections.result()
    except Exception as e:
        result["song_structure"] = {"error": str(e)}

    if future_melody is not None:
        try:
            result["melody"] = future_melody.result()
        except Exception as e:
            result["melody"] = {"error": str(e)}

    # Phase 6: Vocal + rhythm analysis (GPU — uses CLAP internally)
    with gpu_lock:
        models.unload("whisper")

    if vocals is not None and stem_analysis.get("vocals", {}).get("present", False):
        try:
            vocal_cache_key = FileIdentifier.cache_key(file_path, "clap_stem", stem="vocals_analysis", max_duration=max_duration)
            result["vocal_analysis"] = analyze_vocals(vocals, DEMUCS_SR, _audio_cache_key=vocal_cache_key)
        except Exception as e:
            result["vocal_analysis"] = {"error": str(e)}

    drums = stems.get("drums")
    if drums is not None and stem_analysis.get("drums", {}).get("present", False):
        try:
            drum_cache_key = FileIdentifier.cache_key(file_path, "clap_stem", stem="drums_analysis", max_duration=max_duration)
            result["rhythm_analysis"] = analyze_rhythm(drums, DEMUCS_SR, _audio_cache_key=drum_cache_key)
        except Exception as e:
            result["rhythm_analysis"] = {"error": str(e)}

    return _tool_output(result, "deep_listen")


# ─── NEW: Creation & Utility Tools ───────────────────────────────────────────


@mcp.tool()
def extract_melody(file_path: str, max_duration: float = 120.0) -> str:
    """
    Extract melodic content from a track. Uses Demucs to isolate the
    melodic instruments, then tracks pitch to find phrases and hooks.

    Args:
        file_path: Path to the audio file.
        max_duration: Max seconds to analyze.
    """
    max_duration = min(max_duration, MAX_DURATION)
    stems = separate_stems(file_path, max_duration=max_duration)
    with gpu_lock:
        models.unload("demucs")

    other = stems.get("other")
    if other is None:
        return _tool_output({"error": "Could not separate melodic content"}, "extract_melody")

    melody = extract_melody_from_array(other, DEMUCS_SR, _file_path=file_path)
    return _tool_output({"file": file_path, **melody}, "extract_melody")


@mcp.tool()
def sync_tracks(file_path_a: str, file_path_b: str) -> str:
    """
    Compare two tracks and calculate what adjustments are needed to sync them
    for mixing, sampling, or recording over. Returns tempo ratio, key transposition,
    and Camelot wheel compatibility.

    Args:
        file_path_a: Path to the primary track (e.g., your beat).
        file_path_b: Path to the reference track (e.g., track you want to match).
    """
    # Parallel: analyze both tracks
    future_a = _cpu_pool.submit(get_signal_features, file_path_a, 60)
    future_b = _cpu_pool.submit(get_signal_features, file_path_b, 60)
    signal_a = future_a.result()
    signal_b = future_b.result()

    tempo_a = signal_a["tempo_bpm"]
    tempo_b = signal_b["tempo_bpm"]
    key_a = signal_a["estimated_key"]
    key_b = signal_b["estimated_key"]

    tempo_ratio = tempo_b / tempo_a if tempo_a > 0 else 1
    speed_change_pct = round((tempo_ratio - 1) * 100, 1)

    semitones = key_distance_semitones(key_a, key_b)

    camelot_a = CAMELOT_WHEEL.get(key_a, "?")
    camelot_b = CAMELOT_WHEEL.get(key_b, "?")

    # Determine harmonic compatibility
    harmonic_compatible = False
    if camelot_a != "?" and camelot_b != "?":
        num_a = int(camelot_a[:-1]) if camelot_a[:-1].isdigit() else 0
        num_b = int(camelot_b[:-1]) if camelot_b[:-1].isdigit() else 0
        letter_a = camelot_a[-1]
        letter_b = camelot_b[-1]
        # Compatible: same number, adjacent numbers, or same number different letter
        harmonic_compatible = (
            camelot_a == camelot_b
            or (num_a == num_b and letter_a != letter_b)
            or (abs(num_a - num_b) == 1 and letter_a == letter_b)
            or (abs(num_a - num_b) == 11 and letter_a == letter_b)  # wrap around
        )

    return _tool_output({
        "track_a": {
            "file": file_path_a, "tempo": tempo_a, "key": key_a, "camelot": camelot_a,
        },
        "track_b": {
            "file": file_path_b, "tempo": tempo_b, "key": key_b, "camelot": camelot_b,
        },
        "sync_instructions": {
            "tempo_ratio": round(tempo_ratio, 4),
            "speed_change_percent": speed_change_pct,
            "speed_direction": f"speed up track A by {abs(speed_change_pct)}%" if speed_change_pct > 0 else f"slow down track A by {abs(speed_change_pct)}%" if speed_change_pct < 0 else "same tempo",
            "key_transposition_semitones": semitones,
            "transpose_direction": f"transpose track A up {semitones} semitones" if semitones > 0 else f"transpose track A down {abs(semitones)} semitones" if semitones < 0 else "same key",
            "harmonically_compatible": harmonic_compatible,
        },
    }, "sync_tracks")


@mcp.tool()
def analyze_for_songwriting(
    beat_path: str,
    reference_paths: str = "",
    vocal_range_low: str = "",
    vocal_range_high: str = "",
    max_duration: float = 120.0,
) -> str:
    """
    Analyze a beat for songwriting. Returns key, tempo, chord map, suggested
    song structure, vocal entry points, and scale recommendations.
    Optionally compare against reference tracks for style guidance.

    Args:
        beat_path: Path to the beat/instrumental to write over.
        reference_paths: Comma-separated paths to reference tracks (optional).
        vocal_range_low: Your lowest comfortable note, e.g. "C3" (optional).
        vocal_range_high: Your highest comfortable note, e.g. "G4" (optional).
        max_duration: Max seconds to analyze.
    """
    max_duration = min(max_duration, MAX_DURATION)
    result = {"beat": beat_path}

    # Analyze the beat
    signal = get_signal_features(beat_path, max_duration)
    result["beat_analysis"] = {
        "tempo_bpm": signal["tempo_bpm"],
        "key": signal["estimated_key"],
        "key_confidence": signal["key_confidence"],
        "brightness": signal["spectral"]["brightness"],
        "duration": signal["duration_formatted"],
    }

    # Get beat structure
    y_beat = load_audio(beat_path, sr=LIBROSA_SR, duration=max_duration)
    sections = detect_sections_from_array(y_beat, LIBROSA_SR, _file_path=beat_path)
    result["beat_structure"] = sections

    # Detect chords from the beat (no stem separation needed for instrumentals)
    chords = detect_chords_from_array(y_beat, LIBROSA_SR, _file_path=beat_path)
    chord_counts = {}
    for c in chords:
        chord_counts[c["chord"]] = chord_counts.get(c["chord"], 0) + 1
    top_chords = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)
    result["chords"] = {
        "progression": chords[:30],
        "most_used": [{"chord": c, "count": n} for c, n in top_chords[:6]],
    }

    # Scale recommendation based on key
    key = signal["estimated_key"]
    root = key.split()[0] if key != "unknown" else "C"
    mode = key.split()[1] if len(key.split()) > 1 else "major"
    root_idx = KEY_NAMES.index(root) if root in KEY_NAMES else 0

    if mode == "minor":
        # Natural minor + pentatonic
        minor_scale = [(root_idx + s) % 12 for s in [0, 2, 3, 5, 7, 8, 10]]
        penta = [(root_idx + s) % 12 for s in [0, 3, 5, 7, 10]]
        result["scale_recommendation"] = {
            "scale": f"{root} natural minor",
            "notes": [KEY_NAMES[n] for n in minor_scale],
            "pentatonic": [KEY_NAMES[n] for n in penta],
            "avoid_notes": [KEY_NAMES[n] for n in range(12) if n not in minor_scale],
        }
    else:
        major_scale = [(root_idx + s) % 12 for s in [0, 2, 4, 5, 7, 9, 11]]
        penta = [(root_idx + s) % 12 for s in [0, 2, 4, 7, 9]]
        result["scale_recommendation"] = {
            "scale": f"{root} major",
            "notes": [KEY_NAMES[n] for n in major_scale],
            "pentatonic": [KEY_NAMES[n] for n in penta],
            "avoid_notes": [KEY_NAMES[n] for n in range(12) if n not in major_scale],
        }

    # Vocal range fit
    if vocal_range_low and vocal_range_high:
        result["vocal_fit"] = {
            "your_range": f"{vocal_range_low} - {vocal_range_high}",
            "recommendation": f"Sing melodies primarily within {vocal_range_low} to {vocal_range_high}, centering around the {root} in your range",
        }

    # Suggest song structure based on beat structure and tempo
    beat_dur = signal["duration_seconds"]
    bpm = signal["tempo_bpm"]
    beats_per_bar = 4
    bar_duration = (60 / bpm) * beats_per_bar

    suggested_bars = {
        "intro": 4, "verse": 16, "pre_chorus": 4,
        "chorus": 8, "bridge": 8, "outro": 4,
    }
    result["suggested_structure"] = {
        "bar_duration_seconds": round(bar_duration, 2),
        "template": [
            {"part": name, "bars": bars, "duration_seconds": round(bars * bar_duration, 1)}
            for name, bars in suggested_bars.items()
        ],
        "total_bars": sum(suggested_bars.values()),
        "total_duration": format_seconds(sum(suggested_bars.values()) * bar_duration),
    }

    # Find vocal entry points (where energy dips = space for vocals)
    rms = librosa.feature.rms(y=y_beat)[0]
    rms_smooth = np.convolve(rms, np.ones(10) / 10, mode="same")
    entry_points = []
    rms_times = librosa.frames_to_time(range(len(rms_smooth)), sr=LIBROSA_SR)
    below_median = rms_smooth < np.median(rms_smooth)
    in_gap = False
    gap_start = 0
    for i in range(len(below_median)):
        if below_median[i] and not in_gap:
            gap_start = float(rms_times[i])
            in_gap = True
        elif not below_median[i] and in_gap:
            gap_end = float(rms_times[i])
            if gap_end - gap_start > 1.0:  # gaps longer than 1 second
                entry_points.append({
                    "start": format_seconds(gap_start),
                    "end": format_seconds(gap_end),
                    "duration": round(gap_end - gap_start, 1),
                })
            in_gap = False

    result["vocal_entry_points"] = entry_points[:10]

    # Reference track analysis (parallel)
    if reference_paths:
        refs = [p.strip() for p in reference_paths.split(",") if p.strip()]

        def _analyze_ref(ref_path):
            ref_signal = get_signal_features(ref_path, 60)
            ref_sections = detect_sections_from_array(
                load_audio(ref_path, sr=LIBROSA_SR, duration=60), LIBROSA_SR, _file_path=ref_path,
            )
            return {
                "file": ref_path,
                "tempo": ref_signal["tempo_bpm"],
                "key": ref_signal["estimated_key"],
                "structure_summary": [
                    {"part": s.get("part", "unknown"), "duration": s["duration"]}
                    for s in ref_sections
                ],
            }

        futures = [(ref_path, _cpu_pool.submit(_analyze_ref, ref_path)) for ref_path in refs[:3]]
        ref_analyses = []
        for ref_path, future in futures:
            try:
                ref_analyses.append(future.result())
            except Exception as e:
                ref_analyses.append({"file": ref_path, "error": str(e)})
        result["reference_tracks"] = ref_analyses

    return _tool_output(result, "analyze_for_songwriting")


@mcp.tool()
def export_stems(file_path: str, output_dir: str = "", max_duration: float = 600.0) -> str:
    """
    Separate a track into stems and save each as a WAV file.
    Creates: vocals.wav, drums.wav, bass.wav, other.wav in the output directory.

    Args:
        file_path: Path to the audio file.
        output_dir: Directory to save stems. Defaults to same directory as input file.
        max_duration: Max seconds to process.
    """
    max_duration = min(max_duration, MAX_DURATION)
    path = validate_audio_path(file_path)

    if not output_dir:
        output_dir = str(path.parent / f"{path.stem}_stems")

    out_path = Path(output_dir).expanduser().resolve()
    home = Path.home().resolve()
    if not str(out_path).startswith(str(home)):
        return _tool_output({"error": f"Output path must be under home directory ({home})"}, "export_stems")
    out_path.mkdir(parents=True, exist_ok=True)

    stems = separate_stems(file_path, max_duration=max_duration)
    with gpu_lock:
        models.unload("demucs")

    saved = {}
    for name, audio in stems.items():
        stem_path = out_path / f"{name}.wav"
        sf.write(str(stem_path), audio, DEMUCS_SR)
        saved[name] = str(stem_path)

    return _tool_output({
        "file": file_path,
        "output_dir": str(out_path),
        "stems": saved,
        "sample_rate": DEMUCS_SR,
    }, "export_stems")


@mcp.tool()
def export_click_track(file_path: str, output_path: str = "") -> str:
    """
    Generate a click track WAV at the detected tempo of an audio file.
    Useful for recording vocals or instruments in time with a beat.

    Args:
        file_path: Path to the audio file to match tempo of.
        output_path: Where to save the click track. Defaults to <filename>_click.wav.
    """
    path = validate_audio_path(file_path)
    signal = get_signal_features(file_path, max_duration=60)
    tempo = signal["tempo_bpm"]
    duration = signal["duration_seconds"]

    if tempo <= 0 or np.isnan(tempo):
        return _tool_output({"error": "Could not detect a valid tempo from this audio"}, "export_click_track")
    tempo = max(tempo, 30.0)  # clamp to reasonable minimum

    if not output_path:
        output_path = str(path.parent / f"{path.stem}_click.wav")

    out_resolved = Path(output_path).expanduser().resolve()
    home = Path.home().resolve()
    if not str(out_resolved).startswith(str(home)):
        return _tool_output({"error": f"Output path must be under home directory ({home})"}, "export_click_track")

    sr = 44100
    total_samples = int(duration * sr)
    click = np.zeros(total_samples)

    beat_interval = 60.0 / tempo
    click_duration = 0.02  # 20ms click
    click_samples = int(click_duration * sr)

    t = 0
    beat_num = 0
    while t < duration:
        sample_idx = int(t * sr)
        end_idx = min(sample_idx + click_samples, total_samples)
        # Higher pitch on downbeat (beat 1)
        freq = 1500 if beat_num % 4 == 0 else 1000
        click_signal = 0.5 * np.sin(2 * np.pi * freq * np.arange(end_idx - sample_idx) / sr)
        # Apply envelope
        envelope = np.exp(-np.arange(len(click_signal)) / (click_samples * 0.3))
        click[sample_idx:end_idx] = click_signal * envelope
        t += beat_interval
        beat_num += 1

    sf.write(output_path, click, sr)
    return _tool_output({
        "file": file_path,
        "click_track": output_path,
        "tempo_bpm": tempo,
        "duration": format_seconds(duration),
        "total_beats": beat_num,
    }, "export_click_track")


# ─── Audio Download ──────────────────────────────────────────────────────────

DOWNLOAD_DIR = Path.home() / "Documents" / "music" / "music data"


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    import re
    # Remove problematic characters, keep alphanumeric, spaces, hyphens, underscores
    name = re.sub(r'[^\w\s\-]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:200]  # cap length


def _download_with_ytdlp(url_or_query: str, output_dir: Path, is_search: bool = False) -> dict:
    """Download audio using yt-dlp. Returns metadata dict with file path."""
    import subprocess
    import json as _json

    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: extract info without downloading to get the title
    info_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--dump-json", "--no-download",
    ]
    if is_search:
        info_cmd.append(f"ytsearch1:{url_or_query}")
    else:
        info_cmd.append(url_or_query)

    info_result = subprocess.run(
        info_cmd, capture_output=True, text=True, timeout=60,
        cwd=str(Path(__file__).parent),
    )
    if info_result.returncode != 0:
        raise RuntimeError(f"yt-dlp info failed: {info_result.stderr[:500]}")

    info = _json.loads(info_result.stdout)
    title = info.get("title", "unknown")
    uploader = info.get("uploader", info.get("channel", ""))
    duration = info.get("duration", 0)
    webpage_url = info.get("webpage_url", url_or_query)

    # Build filename
    if uploader:
        filename = _sanitize_filename(f"{uploader}_{title}")
    else:
        filename = _sanitize_filename(title)

    output_path = output_dir / f"{filename}.wav"

    # Skip if already downloaded
    if output_path.exists():
        return {
            "file_path": str(output_path),
            "title": title,
            "uploader": uploader,
            "duration_seconds": duration,
            "source_url": webpage_url,
            "already_existed": True,
        }

    # Second pass: download and convert to WAV
    dl_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--output", str(output_dir / f"{filename}.%(ext)s"),
        "--no-playlist",
        "--no-overwrites",
    ]
    if is_search:
        dl_cmd.append(f"ytsearch1:{url_or_query}")
    else:
        dl_cmd.append(webpage_url)

    dl_result = subprocess.run(
        dl_cmd, capture_output=True, text=True, timeout=300,
        cwd=str(Path(__file__).parent),
    )
    if dl_result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {dl_result.stderr[:500]}")

    # yt-dlp may save with slightly different name, find the actual file
    if not output_path.exists():
        wav_files = sorted(output_dir.glob(f"{filename}*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if wav_files:
            output_path = wav_files[0]
        else:
            raise FileNotFoundError(f"Download completed but WAV file not found in {output_dir}")

    return {
        "file_path": str(output_path),
        "title": title,
        "uploader": uploader,
        "duration_seconds": duration,
        "source_url": webpage_url,
        "already_existed": False,
    }


@mcp.tool()
def download_audio(url: str, output_dir: str = "") -> str:
    """
    Download audio from a YouTube or SoundCloud URL as a WAV file.
    Returns the local file path so it can be passed to any analysis tool.

    Args:
        url: YouTube or SoundCloud URL to download.
        output_dir: Directory to save to. Defaults to ~/Documents/music/music data/.
    """
    try:
        out_dir = Path(output_dir).expanduser().resolve() if output_dir else DOWNLOAD_DIR
        result = _download_with_ytdlp(url, out_dir, is_search=False)
        return _tool_output({
            "file": result["file_path"],
            "title": result["title"],
            "artist": result["uploader"],
            "duration": format_seconds(result["duration_seconds"]) if result["duration_seconds"] else "unknown",
            "source_url": result["source_url"],
            "already_existed": result["already_existed"],
        }, "download_audio")
    except Exception as e:
        return _tool_output({"error": str(e)}, "download_audio")


@mcp.tool()
def search_and_download(query: str, output_dir: str = "") -> str:
    """
    Search YouTube for a song by name and download it as a WAV file.
    Returns the local file path so it can be passed to any analysis tool.

    Args:
        query: Search query (e.g. "gunnr - nasty", "sewerperson what if").
        output_dir: Directory to save to. Defaults to ~/Documents/music/music data/.
    """
    try:
        out_dir = Path(output_dir).expanduser().resolve() if output_dir else DOWNLOAD_DIR
        result = _download_with_ytdlp(query, out_dir, is_search=True)
        return _tool_output({
            "file": result["file_path"],
            "title": result["title"],
            "artist": result["uploader"],
            "duration": format_seconds(result["duration_seconds"]) if result["duration_seconds"] else "unknown",
            "source_url": result["source_url"],
            "already_existed": result["already_existed"],
        }, "search_and_download")
    except Exception as e:
        return _tool_output({"error": str(e)}, "search_and_download")


# ─── Library Query Tools ─────────────────────────────────────────────────────

def _get_agent_db():
    """Return SongDB instance pointing at the agent database."""
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

        genre_counter: Counter = Counter()
        artist_counter: Counter = Counter()

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


if __name__ == "__main__":
    mcp.run(transport="stdio")
