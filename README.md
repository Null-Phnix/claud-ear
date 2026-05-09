# 🔈 Claud-Ear

> Give your AI agent the ability to **listen to and understand music/audio files** — works with ANY MCP client.

Claud-Ear connects your AI agent (Hermes Agent, Claude Code, Codex CLI, etc.) to a full audio intelligence pipeline. Drop in an MP3, WAV, FLAC, OGG, M4A, or OPUS file and your agent can analyze, separate, transcribe, and understand it.

**Default LLM backend: Ollama** (configurable to any OpenAI-compatible API).

---

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

---

## Quick Start

### Prerequisites
- Python 3.11–3.13
- CUDA-capable GPU recommended (CPU-only works but is slower)
- [Ollama](https://ollama.com) running locally (default) or any OpenAI-compatible API
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install & Run

```bash
# Clone
git clone https://github.com/Null-Phnix/claud-ear.git
cd claud-ear

# Install with uv
uv sync

# Test the LLM backend
uv run python llm_backend.py

# Run the MCP server
uv run claud-ear
```

### Configuration

By default, Claud-Ear connects to Ollama at `http://localhost:11434` using `llama3.1:8b`. To customize:

```bash
export AUDIO_LLM_MODEL=llama3.1:8b     # model name
export AUDIO_LLM_HOST=http://localhost:11434  # API endpoint
export AUDIO_LLM_PROVIDER=ollama       # or "openai" for OpenAI-compatible APIs
```

For OpenAI-compatible providers (vLLM, TGI, LiteLLM, etc.):
```bash
export AUDIO_LLM_PROVIDER=openai
export AUDIO_LLM_HOST=http://localhost:8000
export AUDIO_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

### Connect to Your Agent

**Hermes Agent** (or any MCP client) — add to your MCP config:

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

Or for Claude Code:
```bash
claude mcp add claud-ear -- uv run claud-ear
```

---

## Tools

### `deep_listen(file_path)`
Full analysis pipeline — semantic understanding, source separation, transcription, and signal analysis all in one call. This is the main tool.

### `analyze_audio(file_path)`
Quick analysis — genre, mood, instruments, tempo, key. Lighter than deep_listen.

### `separate_stems(file_path)`
Isolate vocals, drums, bass, and other stems from a track as separate audio files.

### `transcribe_lyrics(file_path)`
Extract and transcribe lyrics from vocals.

### `search_and_download(query)`
Search for and download audio from YouTube and other platforms via yt-dlp.

### `sonic_surgery(file_path, operation, **params)`
EQ adjustments, stem manipulation, dynamics processing.

### `generate_beat(genre, bpm, bars)`
Generate a beat with chord progressions, melodies, and drum patterns as MIDI.

---

## Architecture

```
claud-ear/
├── server.py              # MCP server (FastMCP) — main entry point
├── llm_backend.py         # Configurable LLM API client (Ollama/OpenAI)
├── agent.py               # Autonomous batch analysis agent
├── beat_studio.py         # Beat production engine
├── quality.py             # Audio quality assessment
├── discovery.py           # Music discovery tools
├── song_db.py             # Track metadata & lyrics database
├── sonic_surgery.py       # Audio repair & enhancement
├── extractor.py           # Feature extraction pipeline
├── download_playlists.py  # Bulk downloader
├── analyze_bass.py        # Bass frequency analysis
├── analyze_bitter.py      # Mood/valence classifier
├── charts.py              # Billboard chart integration
├── power.py               # Energy/sleep scheduling
├── dashboard.py           # Web dashboard
├── query.py               # Natural language music search
├── start_agent.sh         # Start autonomous agent
├── stop_agent.sh          # Stop autonomous agent
├── pause_at_130.sh        # Pause agent during peak hours
└── docs/                  # Design docs & implementation plans
```

---

## Autonomous Agent

Run the autonomous music intelligence agent to batch-analyze your library:

```bash
# Analyze one song (test mode)
uv run python agent.py --one

# Run in continuous loop
./start_agent.sh

# Stop
./stop_agent.sh
```

The agent scans `~/Documents/music/music data/`, finds pending tracks, analyzes them using the configured LLM backend, and writes full analysis documents to `~/Documents/music/analyses/`.

---

## License

MIT — use it, fork it, vibe with it.
