# Audio Download Feature — Design

## Summary

Add 2 new MCP tools to the audio-mcp server that let Claude download music from YouTube/SoundCloud, so the user doesn't have to manually download tracks.

## Tools

### `download_audio(url, output_dir?)`
- Takes a YouTube or SoundCloud URL
- Downloads audio as WAV to `/Users/josii/Documents/music/music data/` (default)
- Filenames auto-generated from video title, sanitized
- Returns local file path for chaining into other tools

### `search_and_download(query, output_dir?)`
- Takes a search query (e.g. "gunnr - nasty", "sewerperson what if")
- Searches YouTube via yt-dlp
- Downloads best match as WAV
- Returns local file path + metadata (title, duration, source URL)

## Architecture

- **yt-dlp** as the download backend (handles YouTube, SoundCloud, 1000+ sites)
- Added as Python dependency in pyproject.toml
- Downloads run in subprocess to avoid blocking MCP server
- WAV output via ffmpeg (already installed via brew)
- No GPU/model usage — pure download + convert

## File output

- Default directory: `/Users/josii/Documents/music/music data/`
- Format: WAV (uncompressed, matches existing reference tracks)
- Filename pattern: `{artist}_{title}.wav` (sanitized)

## Dependencies

- `yt-dlp` added to pyproject.toml
- FFmpeg already installed on system

## Integration

Both tools return file paths compatible with all existing tools:
`search_and_download("gunnr nasty")` → path → `deep_listen(path)`
