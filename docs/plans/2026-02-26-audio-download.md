# Audio Download Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `download_audio` and `search_and_download` MCP tools so Claude can fetch music from YouTube/SoundCloud without the user manually downloading files.

**Architecture:** Uses yt-dlp (Python library) to search and download audio, converting to WAV via ffmpeg. Two new `@mcp.tool()` functions appended to server.py, using subprocess to avoid blocking. Files saved to the user's music data folder.

**Tech Stack:** yt-dlp, ffmpeg (already installed), subprocess

---

### Task 1: Add yt-dlp dependency

**Files:**
- Modify: `pyproject.toml:6-15` (dependencies list)

**Step 1: Add yt-dlp to dependencies**

In `pyproject.toml`, add `"yt-dlp>=2024.0.0"` to the dependencies list after `"demucs>=4.0.0"`.

**Step 2: Install the new dependency**

Run: `cd /Users/josii/Desktop/audio-mcp-review && uv sync`
Expected: yt-dlp installs successfully

**Step 3: Verify yt-dlp works**

Run: `cd /Users/josii/Desktop/audio-mcp-review && uv run python -c "import yt_dlp; print(yt_dlp.version.__version__)"`
Expected: Prints a version number

---

### Task 2: Add the `download_audio` tool

**Files:**
- Modify: `server.py:2299` (append before `if __name__`)

**Step 1: Add the download_audio constants and helper**

Insert the following at line 2299 (before `if __name__`):

```python
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
    actual_url = info.get("webpage_url", url_or_query)

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
        "--output", str(output_path.with_suffix(".%(ext)s")),
        "--no-playlist",
        "--no-overwrites",
    ]
    if is_search:
        dl_cmd.append(f"ytsearch1:{url_or_query}")
    else:
        dl_cmd.append(actual_url)

    dl_result = subprocess.run(
        dl_cmd, capture_output=True, text=True, timeout=300,
        cwd=str(Path(__file__).parent),
    )
    if dl_result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {dl_result.stderr[:500]}")

    # yt-dlp may save with slightly different name, find the actual file
    if not output_path.exists():
        # Look for any new .wav file in the output dir
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
```

**Step 2: Add the download_audio MCP tool**

Append right after the helper:

```python
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
```

**Step 3: Verify it loads without errors**

Run: `cd /Users/josii/Desktop/audio-mcp-review && uv run python -c "from server import download_audio; print('OK')"`
Expected: Prints "OK"

---

### Task 3: Add the `search_and_download` tool

**Files:**
- Modify: `server.py` (append after download_audio, before `if __name__`)

**Step 1: Add the search_and_download MCP tool**

```python
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
```

**Step 2: Verify it loads without errors**

Run: `cd /Users/josii/Desktop/audio-mcp-review && uv run python -c "from server import search_and_download; print('OK')"`
Expected: Prints "OK"

---

### Task 4: Test download_audio with a real URL

**Step 1: Test with a short YouTube video**

Run: `cd /Users/josii/Desktop/audio-mcp-review && uv run python -c "from server import download_audio; print(download_audio('https://www.youtube.com/watch?v=dQw4w9WgXcQ'))"`
Expected: JSON output with file path, title, duration. WAV file created in music data folder.

**Step 2: Verify the downloaded file works with existing tools**

Run: `cd /Users/josii/Desktop/audio-mcp-review && uv run python -c "from server import get_audio_info; print(get_audio_info('<path from step 1>'))"`
Expected: Audio info returned (duration, sample rate, etc.)

**Step 3: Clean up test file**

Delete the test WAV file downloaded in step 1.

---

### Task 5: Test search_and_download with a real query

**Step 1: Test search**

Run: `cd /Users/josii/Desktop/audio-mcp-review && uv run python -c "from server import search_and_download; print(search_and_download('gunnr nasty official'))"`
Expected: JSON with file path, title matching "nasty" by gunnr, WAV file created.

**Step 2: Test duplicate detection**

Run the same command again.
Expected: `"already_existed": true` in the output, no re-download.

**Step 3: Clean up test file**

Delete the test WAV file.

---

### Task 6: Bump version and update description

**Files:**
- Modify: `server.py:1-20` (docstring)
- Modify: `pyproject.toml:3` (version)

**Step 1: Update server.py docstring**

Add to the docstring after "v4.1:" line:
```
v4.2: Audio download tools (download_audio, search_and_download) via yt-dlp.
```

**Step 2: Update SCHEMA_VERSION**

Change `SCHEMA_VERSION = "4.1"` to `SCHEMA_VERSION = "4.2"` in server.py.

**Step 3: Update pyproject.toml version**

Change `version = "4.1.0"` to `version = "4.2.0"`.
