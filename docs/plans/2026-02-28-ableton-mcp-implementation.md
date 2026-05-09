# Ableton MCP Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a separate FastMCP server that gives Claude full control over Ableton Live via AbletonOSC + real-time audio monitoring via BlackHole virtual audio device.

**Architecture:** A FastMCP server communicates with Ableton Live through the AbletonOSC Max for Live device via OSC messages (port 11000/11001). Audio monitoring is done by capturing from BlackHole virtual audio device using sounddevice + librosa analysis. MIDI generation is done algorithmically using music theory then dropped into Ableton clips via OSC.

**Tech Stack:** Python 3.11+, FastMCP, python-osc (pythonosc), mido, sounddevice, librosa, numpy, soundfile, uv

---

## Pre-requisites (user must do before running)

1. Install AbletonOSC M4L device: https://github.com/ideoforms/AbletonOSC — drag `AbletonOSC.amxd` into any Ableton session
2. Install BlackHole 2ch: https://existential.audio/blackhole/ — free virtual audio device
3. In macOS Audio MIDI Setup: create a Multi-Output Device combining Scarlett + BlackHole 2ch
4. In Ableton Preferences > Audio: set Output to the Multi-Output Device

---

## Task 1: Project Setup

**Files:**
- Create: `/Users/josii/Desktop/ableton-mcp/pyproject.toml`
- Create: `/Users/josii/Desktop/ableton-mcp/server.py`
- Create: `/Users/josii/Desktop/ableton-mcp/osc_client.py`
- Create: `/Users/josii/Desktop/ableton-mcp/midi_gen.py`
- Create: `/Users/josii/Desktop/ableton-mcp/audio_capture.py`
- Create: `/Users/josii/Desktop/ableton-mcp/tests/__init__.py`

**Step 1: Create project directory**

```bash
mkdir -p /Users/josii/Desktop/ableton-mcp/tests
cd /Users/josii/Desktop/ableton-mcp
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "ableton-mcp"
version = "1.0.0"
description = "MCP server for Ableton Live control via AbletonOSC"
requires-python = ">=3.11"
dependencies = [
    "fastmcp>=2.0.0",
    "python-osc>=1.8.0",
    "mido>=1.3.0",
    "sounddevice>=0.4.6",
    "librosa>=0.10.0",
    "numpy>=1.24.0",
    "soundfile>=0.12.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-mock>=3.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = ["pytest>=7.0", "pytest-mock>=3.0"]
```

**Step 3: Install dependencies**

```bash
cd /Users/josii/Desktop/ableton-mcp
uv sync
```

Expected: all packages install without errors.

**Step 4: Create empty source files**

```bash
touch server.py osc_client.py midi_gen.py audio_capture.py tests/__init__.py
```

**Step 5: Commit**

```bash
git init
git add pyproject.toml tests/__init__.py
git commit -m "feat: init ableton-mcp project"
```

---

## Task 2: OSC Client Layer

**Files:**
- Create: `/Users/josii/Desktop/ableton-mcp/osc_client.py`
- Create: `/Users/josii/Desktop/ableton-mcp/tests/test_osc_client.py`

**Step 1: Write the failing test**

```python
# tests/test_osc_client.py
from unittest.mock import patch, MagicMock
from osc_client import AbletonOSC

def test_send_calls_udp_client(mocker):
    mock_client = mocker.patch("osc_client.udp_client.SimpleUDPClient")
    osc = AbletonOSC()
    osc.send("/live/song/start_playing")
    mock_client.return_value.send_message.assert_called_once()

def test_connection_error_when_ableton_not_running():
    osc = AbletonOSC()
    # query with very short timeout should raise TimeoutError
    try:
        osc.query("/live/song/get/tempo", "/live/song/get/tempo", timeout=0.1)
        assert False, "Should have raised TimeoutError"
    except TimeoutError:
        pass

def test_is_connected_false_when_no_response():
    osc = AbletonOSC()
    assert osc.is_connected(timeout=0.1) == False
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/josii/Desktop/ableton-mcp
uv run pytest tests/test_osc_client.py -v
```

Expected: ImportError (osc_client not implemented yet)

**Step 3: Implement osc_client.py**

```python
# osc_client.py
"""OSC communication layer with AbletonOSC Max for Live device."""
import threading
from pythonosc import udp_client, dispatcher, osc_server


class AbletonOSC:
    """Synchronous wrapper around pythonosc for AbletonOSC communication."""

    def __init__(self, host="127.0.0.1", send_port=11000, receive_port=11001):
        self.host = host
        self.send_port = send_port
        self.receive_port = receive_port
        self.client = udp_client.SimpleUDPClient(host, send_port)
        self._responses: dict = {}
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

        self._dispatcher = dispatcher.Dispatcher()
        self._dispatcher.set_default_handler(self._handle_response)

        self._server = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", receive_port), self._dispatcher
        )
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()

    def _handle_response(self, address: str, *args):
        with self._lock:
            self._responses[address] = args
            if address in self._events:
                self._events[address].set()

    def send(self, address: str, *args):
        """Fire-and-forget OSC message."""
        self.client.send_message(address, list(args) if args else None)

    def query(self, send_address: str, response_address: str = None, *args, timeout: float = 2.0):
        """Send OSC message and wait for response. Returns tuple of response args."""
        if response_address is None:
            response_address = send_address
        event = threading.Event()
        with self._lock:
            self._events[response_address] = event
            self._responses.pop(response_address, None)
        self.client.send_message(send_address, list(args) if args else None)
        if event.wait(timeout):
            with self._lock:
                return self._responses.get(response_address, ())
        raise TimeoutError(
            f"No response from Ableton for '{send_address}'. "
            "Make sure Ableton is open and AbletonOSC is loaded in your session."
        )

    def is_connected(self, timeout: float = 1.0) -> bool:
        """Check if AbletonOSC is reachable."""
        try:
            self.query("/live/song/get/tempo", timeout=timeout)
            return True
        except TimeoutError:
            return False

    def shutdown(self):
        self._server.shutdown()
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_osc_client.py -v
```

Expected: all 3 tests PASS

**Step 5: Commit**

```bash
git add osc_client.py tests/test_osc_client.py
git commit -m "feat: add OSC client layer for AbletonOSC communication"
```

---

## Task 3: MIDI Generation

**Files:**
- Create: `/Users/josii/Desktop/ableton-mcp/midi_gen.py`
- Create: `/Users/josii/Desktop/ableton-mcp/tests/test_midi_gen.py`

**Step 1: Write the failing tests**

```python
# tests/test_midi_gen.py
from midi_gen import generate_melody, generate_chord_progression, generate_drum_pattern, Note

def test_generate_melody_returns_correct_number_of_bars():
    notes = generate_melody(key="C", scale="minor", bars=4)
    assert len(notes) > 0
    # total duration should be 4 bars (4 * 4 beats = 16 beats)
    total_duration = sum(n.duration for n in notes)
    assert abs(total_duration - 16.0) < 0.1

def test_generate_melody_stays_in_key():
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # semitones
    notes = generate_melody(key="C", scale="minor", bars=2)
    root = 60  # C4
    for note in notes:
        semitone = (note.pitch - root) % 12
        assert semitone in MINOR_SCALE, f"Note {note.pitch} not in C minor"

def test_generate_chord_progression_returns_4_chords_per_4_bars():
    chords = generate_chord_progression(key="G", bars=4)
    assert len(chords) == 4

def test_generate_drum_pattern_returns_notes():
    notes = generate_drum_pattern(style="trap", bars=2)
    assert len(notes) > 0
    # All drum notes should be in GM drum range (35-81)
    for note in notes:
        assert 35 <= note.pitch <= 81

def test_note_has_required_fields():
    notes = generate_melody(key="A", scale="major", bars=1)
    for note in notes:
        assert hasattr(note, 'pitch')
        assert hasattr(note, 'time')
        assert hasattr(note, 'duration')
        assert hasattr(note, 'velocity')
        assert 0 <= note.velocity <= 127
        assert note.pitch >= 0
```

**Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_midi_gen.py -v
```

Expected: ImportError

**Step 3: Implement midi_gen.py**

```python
# midi_gen.py
"""Algorithmic MIDI pattern generation using music theory."""
import random
from dataclasses import dataclass
from typing import Literal


@dataclass
class Note:
    pitch: int       # MIDI pitch 0-127
    time: float      # beat position (0.0 = start)
    duration: float  # in beats
    velocity: int    # 0-127
    mute: int = 0    # 0 or 1


# Scales as semitone intervals from root
SCALES = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues":      [0, 3, 5, 6, 7, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
}

NOTE_NAMES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
              "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
              "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}

# Common chord progressions as scale degree indices (0-based)
PROGRESSIONS = {
    "major": [[0, 4, 5, 3], [0, 5, 3, 4], [0, 3, 4, 4]],
    "minor": [[0, 5, 3, 4], [0, 2, 3, 4], [0, 3, 6, 4]],
}

# GM drum map pitches
KICK = 36
SNARE = 38
HIHAT_CLOSED = 42
HIHAT_OPEN = 46
CLAP = 39
RIMSHOT = 37


def _root_midi(key: str, octave: int = 4) -> int:
    return 12 * (octave + 1) + NOTE_NAMES[key]


def generate_melody(key: str = "C", scale: str = "minor", bars: int = 4,
                    style: str = "lofi") -> list[Note]:
    """Generate a melodic phrase in the given key/scale."""
    scale_degrees = SCALES.get(scale, SCALES["minor"])
    root = _root_midi(key, octave=4)
    # Build pool of available pitches (2 octaves)
    pitch_pool = []
    for octave_offset in [-12, 0, 12]:
        for degree in scale_degrees:
            p = root + degree + octave_offset
            if 48 <= p <= 84:
                pitch_pool.append(p)

    # Rhythm patterns in beats (lofi = lots of syncopation)
    rhythms = [0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.5]
    notes = []
    total_beats = bars * 4.0
    time = 0.0
    current_pitch = random.choice([p for p in pitch_pool if root <= p <= root + 12])

    while time < total_beats:
        dur = random.choice(rhythms)
        if time + dur > total_beats:
            dur = total_beats - time
        # Random walk — stay close to current pitch
        candidates = [p for p in pitch_pool if abs(p - current_pitch) <= 5]
        if not candidates:
            candidates = pitch_pool
        current_pitch = random.choice(candidates)
        vel = random.randint(60, 100)
        notes.append(Note(pitch=current_pitch, time=time, duration=dur * 0.9, velocity=vel))
        time += dur

    return notes


def generate_chord_progression(key: str = "C", bars: int = 4,
                                style: str = "minor") -> list[list[Note]]:
    """Generate a chord progression. Returns list of chords (each chord = list of Notes)."""
    scale_type = "minor" if style in ("minor", "emo", "lofi", "sad") else "major"
    scale_degrees = SCALES.get(scale_type, SCALES["minor"])
    root = _root_midi(key, octave=3)
    progression = random.choice(PROGRESSIONS[scale_type])

    chords = []
    beats_per_chord = (bars * 4) / len(progression)

    for i, degree_idx in enumerate(progression):
        chord_root = root + scale_degrees[degree_idx % len(scale_degrees)]
        # Build triad: root, third, fifth
        third = chord_root + scale_degrees[2 % len(scale_degrees)]
        fifth = chord_root + scale_degrees[4 % len(scale_degrees)]
        time = i * beats_per_chord
        chord_notes = [
            Note(pitch=chord_root, time=time, duration=beats_per_chord * 0.95, velocity=75),
            Note(pitch=third, time=time, duration=beats_per_chord * 0.95, velocity=70),
            Note(pitch=fifth, time=time, duration=beats_per_chord * 0.95, velocity=70),
        ]
        chords.append(chord_notes)

    return chords


def generate_drum_pattern(style: str = "trap", bars: int = 2) -> list[Note]:
    """Generate a drum pattern. Returns flat list of Notes using GM drum pitches."""
    notes = []
    total_beats = bars * 4.0
    step = 0.25  # 16th note grid

    for beat_16th in range(int(total_beats / step)):
        t = beat_16th * step
        beat = beat_16th % 16  # position within 2-bar pattern

        if style == "trap":
            # Kick: beats 0, 6, 10 (syncopated)
            if beat in (0, 6, 10):
                notes.append(Note(KICK, t, 0.2, random.randint(90, 110)))
            # Snare: beats 4, 12
            if beat in (4, 12):
                notes.append(Note(SNARE, t, 0.2, random.randint(85, 105)))
            # Hihat: every 16th, with velocity variation
            vel = random.randint(40, 80)
            hat = HIHAT_OPEN if beat % 4 == 2 else HIHAT_CLOSED
            notes.append(Note(hat, t, 0.2, vel))

        elif style == "lofi":
            # Kick: 0, 9
            if beat in (0, 9):
                notes.append(Note(KICK, t, 0.2, random.randint(80, 100)))
            # Snare: 4, 12, sometimes 14
            if beat in (4, 12) or (beat == 14 and random.random() > 0.6):
                notes.append(Note(SNARE, t, 0.2, random.randint(70, 90)))
            # Hihat: every 8th with swung feel
            if beat % 2 == 0:
                notes.append(Note(HIHAT_CLOSED, t, 0.2, random.randint(50, 75)))

    return notes


def flatten_chord_progression(chords: list[list[Note]]) -> list[Note]:
    """Flatten list of chords into a single list of notes."""
    return [note for chord in chords for note in chord]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_midi_gen.py -v
```

Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add midi_gen.py tests/test_midi_gen.py
git commit -m "feat: add algorithmic MIDI generation (melody, chords, drums)"
```

---

## Task 4: Audio Capture Layer

**Files:**
- Create: `/Users/josii/Desktop/ableton-mcp/audio_capture.py`
- Create: `/Users/josii/Desktop/ableton-mcp/tests/test_audio_capture.py`

**Step 1: Write the failing test**

```python
# tests/test_audio_capture.py
from audio_capture import list_audio_devices, find_blackhole_device, AudioCapture
import numpy as np

def test_list_audio_devices_returns_list():
    devices = list_audio_devices()
    assert isinstance(devices, list)
    assert len(devices) > 0
    assert all("name" in d for d in devices)

def test_find_blackhole_device_returns_none_or_int():
    idx = find_blackhole_device()
    assert idx is None or isinstance(idx, int)

def test_analyze_audio_array_returns_dict():
    capture = AudioCapture()
    # Generate 5 seconds of 440Hz sine wave (fake audio)
    sr = 44100
    t = np.linspace(0, 5, 5 * sr)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    result = capture.analyze_audio_array(audio, sr)
    assert "tempo" in result
    assert "key" in result
    assert "loudness_lufs" in result
    assert "frequency_balance" in result
```

**Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_audio_capture.py -v
```

Expected: ImportError

**Step 3: Implement audio_capture.py**

```python
# audio_capture.py
"""BlackHole audio capture and real-time analysis."""
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import tempfile
import os
from typing import Optional


def list_audio_devices() -> list[dict]:
    """List all available audio input devices."""
    devices = sd.query_devices()
    result = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            result.append({"index": i, "name": d["name"], "channels": d["max_input_channels"]})
    return result


def find_blackhole_device() -> Optional[int]:
    """Find BlackHole virtual audio device index. Returns None if not installed."""
    for d in list_audio_devices():
        if "blackhole" in d["name"].lower():
            return d["index"]
    return None


class AudioCapture:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.blackhole_idx = find_blackhole_device()

    def capture(self, seconds: float = 10.0) -> Optional[np.ndarray]:
        """Capture audio from BlackHole. Returns numpy array or None if unavailable."""
        if self.blackhole_idx is None:
            return None
        try:
            audio = sd.rec(
                int(seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=2,
                dtype="float32",
                device=self.blackhole_idx,
            )
            sd.wait()
            # Mix to mono
            return audio.mean(axis=1)
        except Exception:
            return None

    def analyze_audio_array(self, audio: np.ndarray, sr: int) -> dict:
        """Run librosa analysis on a numpy audio array."""
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        tempo = float(tempo) if hasattr(tempo, '__float__') else float(tempo[0])

        # Key
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_idx = int(chroma_mean.argmax())
        key = note_names[key_idx]

        # Loudness (approximate LUFS using RMS)
        rms = float(librosa.feature.rms(y=audio).mean())
        loudness_lufs = round(20 * np.log10(rms + 1e-9), 1)

        # Frequency balance (low/mid/high energy)
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        low_mask = freqs < 200
        mid_mask = (freqs >= 200) & (freqs < 4000)
        high_mask = freqs >= 4000
        low_energy = float(stft[low_mask].mean())
        mid_energy = float(stft[mid_mask].mean())
        high_energy = float(stft[high_mask].mean())
        total = low_energy + mid_energy + high_energy + 1e-9
        freq_balance = {
            "low_pct": round(100 * low_energy / total, 1),
            "mid_pct": round(100 * mid_energy / total, 1),
            "high_pct": round(100 * high_energy / total, 1),
        }

        # Spectral centroid (brightness)
        centroid = float(librosa.feature.spectral_centroid(y=audio, sr=sr).mean())

        return {
            "tempo": round(tempo, 1),
            "key": key,
            "loudness_lufs": loudness_lufs,
            "frequency_balance": freq_balance,
            "spectral_centroid_hz": round(centroid, 1),
        }

    def capture_and_analyze(self, seconds: float = 10.0) -> dict:
        """Capture from BlackHole and run full analysis."""
        audio = self.capture(seconds)
        if audio is None:
            return {"error": "BlackHole not found or not capturing. Check audio routing setup."}
        return self.analyze_audio_array(audio, self.sample_rate)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_audio_capture.py -v
```

Expected: all 3 tests PASS (BlackHole test returns None since BlackHole not installed yet — that's fine)

**Step 5: Commit**

```bash
git add audio_capture.py tests/test_audio_capture.py
git commit -m "feat: add BlackHole audio capture and librosa analysis"
```

---

## Task 5: MCP Server — Transport & Session Tools

**Files:**
- Create: `/Users/josii/Desktop/ableton-mcp/server.py`
- Create: `/Users/josii/Desktop/ableton-mcp/tests/test_transport.py`

**Step 1: Write the failing tests**

```python
# tests/test_transport.py
from unittest.mock import MagicMock, patch

def test_play_sends_correct_osc(mocker):
    mock_osc = mocker.MagicMock()
    with patch("server.osc", mock_osc):
        from server import play
        play()
        mock_osc.send.assert_called_with("/live/song/start_playing")

def test_stop_sends_correct_osc(mocker):
    mock_osc = mocker.MagicMock()
    with patch("server.osc", mock_osc):
        from server import stop
        stop()
        mock_osc.send.assert_called_with("/live/song/stop_playing")

def test_set_tempo_sends_correct_osc(mocker):
    mock_osc = mocker.MagicMock()
    with patch("server.osc", mock_osc):
        from server import set_tempo
        set_tempo(120.0)
        mock_osc.send.assert_called_with("/live/song/set/tempo", 120.0)
```

**Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_transport.py -v
```

Expected: ImportError

**Step 3: Implement server.py with transport + session tools**

```python
# server.py
"""Ableton MCP Server v1.0 — Control Ableton Live via AbletonOSC + BlackHole audio capture."""
import json
from typing import Optional
from mcp.server.fastmcp import FastMCP
from osc_client import AbletonOSC
from audio_capture import AudioCapture
from midi_gen import (
    generate_melody, generate_chord_progression, generate_drum_pattern,
    flatten_chord_progression
)

mcp = FastMCP("ableton-mcp")
osc = AbletonOSC()
capture = AudioCapture()

# ─── Connection Check ─────────────────────────────────────────────────────────

def _check_connection():
    if not osc.is_connected(timeout=1.0):
        raise RuntimeError(
            "Cannot reach Ableton. Make sure:\n"
            "1. Ableton Live is open\n"
            "2. AbletonOSC device is loaded in your session"
        )

# ─── Transport ────────────────────────────────────────────────────────────────

@mcp.tool()
def play() -> str:
    """Start playback in Ableton."""
    _check_connection()
    osc.send("/live/song/start_playing")
    return "Playback started."

@mcp.tool()
def stop() -> str:
    """Stop playback in Ableton."""
    _check_connection()
    osc.send("/live/song/stop_playing")
    return "Playback stopped."

@mcp.tool()
def record() -> str:
    """Arm record and start recording in Ableton."""
    _check_connection()
    osc.send("/live/song/set/record_mode", 1)
    osc.send("/live/song/start_playing")
    return "Recording started."

@mcp.tool()
def set_tempo(bpm: float) -> str:
    """Set the session tempo in BPM."""
    _check_connection()
    osc.send("/live/song/set/tempo", bpm)
    return f"Tempo set to {bpm} BPM."

@mcp.tool()
def get_session_info() -> str:
    """Get current session state: tempo, time signature, track count, playback position."""
    _check_connection()
    tempo = osc.query("/live/song/get/tempo")
    num_tracks_resp = osc.query("/live/song/get/num_tracks")
    time_sig = osc.query("/live/song/get/time_signature_numerator")

    info = {
        "tempo": round(float(tempo[0]), 2) if tempo else None,
        "num_tracks": int(num_tracks_resp[0]) if num_tracks_resp else 0,
        "time_sig_numerator": int(time_sig[0]) if time_sig else 4,
    }
    return json.dumps(info, indent=2)

# ─── Tracks ───────────────────────────────────────────────────────────────────

@mcp.tool()
def get_tracks() -> str:
    """List all tracks in the session with their names, types, and volumes."""
    _check_connection()
    num_resp = osc.query("/live/song/get/num_tracks")
    num_tracks = int(num_resp[0]) if num_resp else 0
    tracks = []
    for i in range(num_tracks):
        name_resp = osc.query("/live/track/get/name", "/live/track/get/name", i)
        vol_resp = osc.query("/live/track/get/volume", "/live/track/get/volume", i)
        mute_resp = osc.query("/live/track/get/mute", "/live/track/get/mute", i)
        tracks.append({
            "index": i,
            "name": name_resp[0] if name_resp else f"Track {i}",
            "volume": round(float(vol_resp[0]), 3) if vol_resp else 0.85,
            "muted": bool(mute_resp[0]) if mute_resp else False,
        })
    return json.dumps(tracks, indent=2)

@mcp.tool()
def create_midi_track(name: str = "MIDI Track") -> str:
    """Create a new MIDI track in the session."""
    _check_connection()
    osc.send("/live/song/create_midi_track", -1)
    return f"Created MIDI track '{name}'."

@mcp.tool()
def create_audio_track(name: str = "Audio Track") -> str:
    """Create a new audio track in the session."""
    _check_connection()
    osc.send("/live/song/create_audio_track", -1)
    return f"Created audio track '{name}'."

@mcp.tool()
def set_track_volume(track_index: int, volume: float) -> str:
    """Set track volume. volume is 0.0 (silent) to 1.0 (0dB). Use 0.85 for unity gain."""
    _check_connection()
    osc.send("/live/track/set/volume", track_index, volume)
    return f"Track {track_index} volume set to {volume}."

@mcp.tool()
def set_track_pan(track_index: int, pan: float) -> str:
    """Set track panning. -1.0 = full left, 0.0 = center, 1.0 = full right."""
    _check_connection()
    osc.send("/live/track/set/panning", track_index, pan)
    return f"Track {track_index} pan set to {pan}."

@mcp.tool()
def mute_track(track_index: int) -> str:
    """Mute a track by index."""
    _check_connection()
    osc.send("/live/track/set/mute", track_index, 1)
    return f"Track {track_index} muted."

@mcp.tool()
def unmute_track(track_index: int) -> str:
    """Unmute a track by index."""
    _check_connection()
    osc.send("/live/track/set/mute", track_index, 0)
    return f"Track {track_index} unmuted."

@mcp.tool()
def solo_track(track_index: int) -> str:
    """Solo a track by index."""
    _check_connection()
    osc.send("/live/track/set/solo", track_index, 1)
    return f"Track {track_index} soloed."

@mcp.tool()
def arm_track(track_index: int) -> str:
    """Arm a track for recording."""
    _check_connection()
    osc.send("/live/track/set/arm", track_index, 1)
    return f"Track {track_index} armed for recording."

# ─── Devices / Plugins ────────────────────────────────────────────────────────

@mcp.tool()
def get_devices(track_index: int) -> str:
    """List all devices/plugins on a track."""
    _check_connection()
    num_resp = osc.query("/live/track/get/num_devices", "/live/track/get/num_devices", track_index)
    num_devices = int(num_resp[0]) if num_resp else 0
    devices = []
    for i in range(num_devices):
        name_resp = osc.query("/live/device/get/name", "/live/device/get/name", track_index, i)
        devices.append({"index": i, "name": name_resp[0] if name_resp else f"Device {i}"})
    return json.dumps(devices, indent=2)

@mcp.tool()
def get_device_params(track_index: int, device_index: int) -> str:
    """List all parameters for a device/plugin on a track."""
    _check_connection()
    num_resp = osc.query(
        "/live/device/get/num_parameters",
        "/live/device/get/num_parameters",
        track_index, device_index
    )
    num_params = int(num_resp[0]) if num_resp else 0
    params = []
    for i in range(num_params):
        name_resp = osc.query(
            "/live/device/get/parameter/name",
            "/live/device/get/parameter/name",
            track_index, device_index, i
        )
        val_resp = osc.query(
            "/live/device/get/parameter/value",
            "/live/device/get/parameter/value",
            track_index, device_index, i
        )
        params.append({
            "index": i,
            "name": name_resp[0] if name_resp else f"Param {i}",
            "value": round(float(val_resp[0]), 4) if val_resp else 0.0,
        })
    return json.dumps(params, indent=2)

@mcp.tool()
def set_device_param(track_index: int, device_index: int, param_index: int, value: float) -> str:
    """Set a device parameter by index. Get param indices from get_device_params()."""
    _check_connection()
    osc.send("/live/device/set/parameter/value", track_index, device_index, param_index, value)
    return f"Track {track_index}, device {device_index}, param {param_index} set to {value}."

# ─── MIDI Clips ───────────────────────────────────────────────────────────────

@mcp.tool()
def generate_and_drop_melody(track_index: int, clip_slot: int, key: str = "C",
                              scale: str = "minor", bars: int = 4) -> str:
    """Generate a melody and drop it into an Ableton MIDI clip slot."""
    _check_connection()
    notes = generate_melody(key=key, scale=scale, bars=bars)
    length_beats = bars * 4.0
    osc.send("/live/clip_slot/create_clip", track_index, clip_slot, length_beats)
    # Pack notes as flat args: pitch, time, duration, velocity, mute
    args = []
    for n in notes:
        args.extend([n.pitch, n.time, n.duration, n.velocity, n.mute])
    osc.send("/live/clip/add/notes", track_index, clip_slot, *args)
    return f"Generated {len(notes)}-note melody in {key} {scale} ({bars} bars) → track {track_index}, slot {clip_slot}."

@mcp.tool()
def generate_and_drop_chords(track_index: int, clip_slot: int, key: str = "C",
                              bars: int = 4, style: str = "minor") -> str:
    """Generate a chord progression and drop it into an Ableton MIDI clip slot."""
    _check_connection()
    chords = generate_chord_progression(key=key, bars=bars, style=style)
    notes = flatten_chord_progression(chords)
    length_beats = bars * 4.0
    osc.send("/live/clip_slot/create_clip", track_index, clip_slot, length_beats)
    args = []
    for n in notes:
        args.extend([n.pitch, n.time, n.duration, n.velocity, n.mute])
    osc.send("/live/clip/add/notes", track_index, clip_slot, *args)
    return f"Generated {len(chords)}-chord progression in {key} ({bars} bars) → track {track_index}, slot {clip_slot}."

@mcp.tool()
def generate_and_drop_drums(track_index: int, clip_slot: int,
                             style: str = "trap", bars: int = 2) -> str:
    """Generate a drum pattern and drop it into an Ableton MIDI clip slot."""
    _check_connection()
    notes = generate_drum_pattern(style=style, bars=bars)
    length_beats = bars * 4.0
    osc.send("/live/clip_slot/create_clip", track_index, clip_slot, length_beats)
    args = []
    for n in notes:
        args.extend([n.pitch, n.time, n.duration, n.velocity, n.mute])
    osc.send("/live/clip/add/notes", track_index, clip_slot, *args)
    return f"Generated {style} drum pattern ({bars} bars) → track {track_index}, slot {clip_slot}."

@mcp.tool()
def fire_clip(track_index: int, clip_slot: int) -> str:
    """Launch/fire a clip in session view."""
    _check_connection()
    osc.send("/live/clip/fire", track_index, clip_slot)
    return f"Fired clip at track {track_index}, slot {clip_slot}."

# ─── Audio Monitoring ─────────────────────────────────────────────────────────

@mcp.tool()
def analyze_current_mix(seconds: float = 10.0) -> str:
    """Capture Ableton's output via BlackHole and analyze the current mix.
    Returns tempo, key, loudness, frequency balance, and mixing suggestions."""
    result = capture.capture_and_analyze(seconds)
    if "error" in result:
        return result["error"]

    freq = result.get("frequency_balance", {})
    suggestions = []
    if freq.get("low_pct", 0) > 45:
        suggestions.append("Low end is heavy (>45% energy). Try cutting bass/kick below 60Hz or reducing low-shelf.")
    if freq.get("high_pct", 0) < 10:
        suggestions.append("Mix sounds dull — high end is thin. Add air EQ (10kHz+) or check if high-pass is too aggressive.")
    if result.get("loudness_lufs", 0) < -18:
        suggestions.append("Mix is quiet. Bring up the master volume or add limiting.")
    if result.get("loudness_lufs", 0) > -6:
        suggestions.append("Mix might be clipping. Pull master volume down before adding limiting.")

    result["suggestions"] = suggestions
    return json.dumps(result, indent=2)

@mcp.tool()
def check_ableton_connection() -> str:
    """Check if Ableton is open and AbletonOSC is responding."""
    connected = osc.is_connected(timeout=2.0)
    bh = capture.blackhole_idx
    return json.dumps({
        "ableton_connected": connected,
        "blackhole_available": bh is not None,
        "blackhole_device_index": bh,
        "status": "Ready" if connected else "Ableton not connected — open Ableton and load AbletonOSC"
    }, indent=2)

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_transport.py -v
```

Expected: PASS (tests mock the osc object)

**Step 5: Commit**

```bash
git add server.py tests/test_transport.py
git commit -m "feat: add full MCP server with transport, track, device, MIDI, and audio tools"
```

---

## Task 6: Register with Claude Code

**Step 1: Find Claude Code settings**

```bash
cat ~/.claude/claude_desktop_config.json 2>/dev/null || cat ~/Library/Application\ Support/Claude/claude_desktop_config.json 2>/dev/null
```

**Step 2: Add ableton-mcp server entry**

Add to the `mcpServers` section in your Claude Code settings:

```json
"ableton-mcp": {
    "command": "/Users/josii/Desktop/ableton-mcp/.venv/bin/python",
    "args": ["/Users/josii/Desktop/ableton-mcp/server.py"],
    "env": {}
}
```

**Step 3: Restart Claude Code**

Quit and reopen Claude Code. Run:

> "check ableton connection"

Expected: Response showing Ableton status.

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete ableton-mcp v1.0"
```

---

## Quick Test Sequence (Once Ableton is Open)

1. Open Ableton → drag in AbletonOSC device → press play on device
2. In Claude Code: "check ableton connection" → should show connected
3. "get session info" → returns tempo, track count
4. "get tracks" → lists all tracks
5. "set tempo to 103 bpm"
6. "generate a trap drum pattern on track 0, slot 0"
7. "fire clip on track 0, slot 0"
8. "analyze current mix" → captures audio, returns analysis
