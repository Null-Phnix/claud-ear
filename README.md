# 🎧 Claud-Ear

> Give Claude the ability to **listen to and understand music/audio files** through MCP.

Claud-Ear is an MCP (Model Context Protocol) server that connects Claude to a full audio intelligence pipeline. Drop in an MP3, WAV, FLAC, OGG, M4A, or OPUS file and Claude can analyze, separate, transcribe, and understand it.

## What It Does

| Capability | Model/Tool |
|------------|-----------|
| 🔍 **Semantic understanding** — genre, mood, instruments, era | CLAP (LAION/CLAP Music & Speech) |
| 🎛️ **Source separation** — isolate vocals, drums, bass, other | Demucs HT |
| 📝 **Lyrics transcription** — transcribe lyrics from isolated vocals | Whisper large-v3 |
| 📊 **Signal analysis** — tempo, key, chords, structure, rhythm | librosa |
| ⬇️ **Audio downloading** — download from YouTube, Spotify, etc. | yt-dlp |
| 🏥 **Audio surgery** — EQ, stem manipulation, dynamics processing | sonic_surgery |
| 🎹 **Beat production** — generate beats, chord progressions, melodies | beat_studio + MIDI |

## Quick Start

### Prerequisites
- Python 3.11–3.13
- CUDA-capable GPU recommended (CPU-only works but is slower)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install & Run

```bash
# Clone
git clone https://github.com/Null-Phnix/claud-ear.git
cd claud-ear

# Install with uv (recommended)
uv sync

# Run the MCP server
uv run claud-ear
```

### Connect to Claude

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "claud-ear": {
      "command": "uv",
      "args": ["run", "claud-ear"]
    }
  }
}
```

Or add it to Claude Desktop's MCP config.

## Tools

### `deep_listen(file_path)`
Full analysis pipeline — semantic understanding, source separation, transcription, and signal analysis all in one call. This is the main tool.

### `analyze_audio(file_path)`
Quick analysis — genre, mood, instruments, tempo, key. Lighter than deep_listen.

### `separate_stems(file_path)`
Isolate vocals, drums, bass, and other stems from a track.

### `transcribe_lyrics(file_path)`
Extract and transcribe lyrics from vocals.

### `search_and_download(query)`
Search for and download audio from YouTube, Spotify, etc.

### `sonic_surgery(file_path, operation, **params)`
EQ adjustments, stem manipulation, dynamics processing.

### `generate_beat(genre, bpm, bars)`
Generate a beat with chord progressions, melodies, and drum patterns.

## Architecture

```
claud-ear/
├── server.py              # MCP server (FastMCP)
├── agent.py               # Autonomous batch analysis agent
├── beat_studio.py          # Beat production engine
├── quality.py              # Audio quality assessment
├── discovery.py            # Music discovery tools
├── song_db.py              # Track metadata & lyrics database
├── sonic_surgery.py        # Audio repair & enhancement
├── extractor.py            # Feature extraction pipeline
├── download_playlists.py   # Bulk downloader
├── analyze_bass.py         # Bass frequency analysis
├── analyze_bitter.py       # Mood/valence classifier
├── charts.py               # Billboard chart integration
├── power.py                # Energy/sleep scheduling
├── dashboard.py            # Web dashboard
├── query.py                # Natural language music search
└── docs/                   # Design docs & plans
```

## License

MIT — use it, fork it, vibe with it.
