# 🔈 Claud-Ear

> Give your AI agent the ability to **listen to and understand music/audio files** — works with ANY MCP client.

[![Python](https://img.shields.io/badge/python-3.11%2B-4ade80?style=flat-square&labelColor=07061a)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22d3ee?style=flat-square&labelColor=07061a)](LICENSE)

**Audio intelligence MCP server. Semantic understanding, stem separation, lyrics transcription, signal analysis — all as tool calls.**

```bash
# Your agent asks:
"Analyze this track — what's the genre, tempo, key, and mood?"
"Separate the vocals from the instrumental"
"Transcribe the lyrics from this song"
"Generate a trap beat at 140 BPM"
```

Claud-Ear connects your AI agent (Hermes Agent, Claude Code, Codex CLI, etc.) to a full audio intelligence pipeline. Drop in an MP3, WAV, FLAC, OGG, M4A, or OPUS file and your agent can analyze, separate, transcribe, and understand it.

---

## Table of Contents

- [Why I Built This](#why-i-built-this)
- [What It Does](#what-it-does)
- [Current Pain Points](#current-pain-points)
- [End Goals](#end-goals--where-this-is-headed)
- [Quick Start](#quick-start)
- [Tools](#tools)
- [Architecture](#architecture)
- [Autonomous Agent](#autonomous-agent)
- [License](#license)

---

## Why I Built This

I have a music library of ~5,000 tracks. I wanted my agent to understand them like I do — not just "this is a 3-minute MP3" but "this is a melancholic D minor indie rock track with a prominent bass line and lyrics about loss."

Existing tools were either:
- **Shallow** — basic metadata (artist, title, duration) with no semantic understanding
- **Cloud-only** — upload your audio to someone's server, pay per analysis, hope they don't train on it
- **GUI-only** — great for humans, useless for agents that need structured tool calls
- **Single-purpose** — one tool for stems, another for transcription, another for analysis, no integration

Claud-Ear is an MCP server because my agent should be able to say "analyze this track" and get back structured data — genre, tempo, key, stems, lyrics, mood — as a tool call result. Not me manually running 4 different CLI tools and copy-pasting the output.

The autonomous agent mode exists because I don't want to manually trigger analysis on 5,000 tracks. It should run overnight, unsupervised, and finish the job.

---

## What It Does

| Capability | Model/Tool | What It Gives You |
|------------|-----------|-------------------|
| 🔍 **Semantic understanding** | CLAP (LAION/CLAP Music & Speech) | Genre, mood, instruments, era classification |
| 🎛️ **Source separation** | Demucs HT | Isolate vocals, drums, bass, other as separate files |
| 📝 **Lyrics transcription** | Whisper large-v3 | Transcribe lyrics from isolated vocals |
| 📊 **Signal analysis** | librosa | Tempo, key, chords, structure, rhythm |
| ⬇️ **Audio downloading** | yt-dlp | Download from YouTube, Spotify, etc. |
| 🏥 **Audio surgery** | sonic_surgery | EQ, stem manipulation, dynamics processing |
| 🎹 **Beat production** | beat_studio + MIDI | Generate beats, chord progressions, melodies |

**Default LLM backend: Ollama** (configurable to any OpenAI-compatible API).

---

## Current Pain Points

These are the battles I'm actively fighting:

1. **`server.py` is 104K lines** — This started as a clean MCP server and became a monolith. CLAP loading, Demucs inference, Whisper transcription, librosa analysis, caching, disk eviction, schema versioning — all in one file. It needs to be split into modules but I keep adding features instead of refactoring.

2. **8GB VRAM means one model at a time** — CLAP, Demucs, and Whisper all want GPU. I can't run them simultaneously. The "deep_listen" tool has to load/unload models in sequence, which turns a 2-minute analysis into a 10-minute analysis. I have a GPU lock system but it's a hack.

3. **Cache invalidation is hard** — I built LRU memory + disk cache with schema versioning. When I change the output format, old cache entries auto-invalidate. But the cache key logic is fragile — same file, same analysis, different day = cache miss because the schema version bumped. I'm over-engineering caching.

4. **yt-dlp breaks monthly** — YouTube changes their frontend, yt-dlp needs an update, and the `search_and_download` tool stops working until I manually update. This is not the tool's fault but it's a maintenance burden I didn't anticipate.

5. **15-minute max duration is arbitrary** — Set to 900 seconds because longer tracks OOM on 8GB VRAM. A 20-minute ambient piece or live set gets truncated. The limit should be dynamic based on available memory, not hardcoded.

6. **Autonomous agent gets stuck** — The batch analysis agent runs overnight but sometimes hangs on one track (corrupted file, unsupported codec, Demucs crash). There's no timeout per-track, so one bad file blocks the whole queue. I need per-track error isolation.

7. **Billboard/Spotify integrations are brittle** — `charts.py` and `discovery.py` depend on third-party APIs with rate limits and breaking changes. The Billboard scraper broke twice in 3 months. These are nice-to-have features that cost more maintenance than value.

---

## End Goals — Where This Is Headed

### Short Term (now → 3 months)
- **Split `server.py` into modules** — one file per capability (clap.py, demucs.py, whisper.py, librosa.py, cache.py)
- **Per-track timeouts in autonomous agent** — one bad file shouldn't block 5,000
- **Dynamic duration limits** — detect available VRAM and set max duration accordingly
- **Better error isolation** — each tool runs in its own subprocess with timeouts and cleanup

### Medium Term (3–6 months)
- **Unified audio knowledge base** — all analyzed tracks feed into a ChromaDB graph (genre connections, similar tracks, playlist generation)
- **Cross-project integration** — Deep Video Watcher's beat detection informs Claud-Ear's analysis; Huginn-scraped lyrics feed into track metadata
- **Local model consolidation** — one vision-audio model instead of CLAP + Demucs + Whisper + librosa juggling

### Long Term (6–12 months)
- **Fully autonomous music curation** — "Here are 10,000 tracks. Generate me 20 playlists that flow well, with transitions, mood arcs, and no jarring genre jumps"
- **Real-time audio analysis** — analyze a track as it's playing, not as a batch job
- **Integration with Bifrost** — mythology-themed music (Wagnerian opera, Japanese taiko, Nordic folk) gets linked to cultural context in the knowledge graph

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
