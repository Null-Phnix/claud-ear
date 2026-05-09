# Ableton MCP Server — Design Doc
**Date:** 2026-02-28
**Status:** Approved

## Overview
A separate FastMCP server that gives Claude full control over Ableton Live 12 Suite via the AbletonOSC bridge + real-time audio monitoring via BlackHole. Enables AI-assisted beat making, MIDI generation, vocal recording, mixing, and plugin control — all through plain English conversation.

---

## Architecture

```
Claude Code
    │ MCP tools
ableton-mcp/server.py (FastMCP)
    │ OSC (port 11000)          │ sounddevice capture
AbletonOSC M4L device      BlackHole virtual audio
    │                              │
Ableton Live 12 Suite ─────────────┘
    │
Focusrite Scarlett (user hears output)
```

**Audio routing:** Ableton → macOS Multi-Output Device (Scarlett + BlackHole simultaneously). User hears through Scarlett, Claude captures through BlackHole for real-time analysis.

---

## File Structure

```
/Users/josii/Desktop/ableton-mcp/
  server.py          # FastMCP server — all MCP tools
  osc_client.py      # OSC communication layer with AbletonOSC
  midi_gen.py        # MIDI pattern generation (melody, chords, drums)
  audio_capture.py   # BlackHole capture + analysis
  pyproject.toml
```

---

## MCP Tools

### Transport
- `play()` — start playback
- `stop()` — stop playback
- `record()` — start recording
- `set_tempo(bpm)` — change BPM
- `set_time_signature(numerator, denominator)`

### Session Info
- `get_session_info()` — tempo, key, time sig, track list, current position
- `get_tracks()` — all tracks with names, types, armed status
- `get_devices(track_index)` — list all plugins/devices on a track
- `get_device_params(track_index, device_index)` — all parameters for a device

### Track Management
- `create_midi_track(name)` — add new MIDI track
- `create_audio_track(name)` — add new audio track
- `set_track_volume(track_index, volume)` — 0.0–1.0
- `set_track_pan(track_index, pan)` — -1.0 to 1.0
- `mute_track(track_index)` / `unmute_track(track_index)`
- `solo_track(track_index)` / `unsolo_track(track_index)`
- `arm_track(track_index)` — arm for recording

### MIDI / Clips
- `create_midi_clip(track_index, slot_index, length_bars)` — create empty clip
- `add_notes_to_clip(track_index, slot_index, notes)` — drop notes into clip
- `generate_melody(key, scale, bars, style)` → creates clip with generated melody
- `generate_chord_progression(key, bars, style)` → creates clip with chords
- `generate_drum_pattern(style, bars)` → creates clip with drum pattern

### Plugin / Device Control
- `set_device_param(track_index, device_index, param_name, value)` — tweak any plugin knob
- `add_device(track_index, device_name)` — add built-in Ableton device to track

### Audio Monitoring (BlackHole)
- `capture_output(seconds)` — record Ableton's live output for N seconds
- `analyze_current_mix()` — capture + full audio analysis (EQ, dynamics, stereo, loudness)
- `compare_to_reference(reference_path)` — compare current mix to a reference WAV, suggest adjustments

---

## Setup Steps (for user)

1. Install **AbletonOSC** M4L device → drag into any Ableton session
2. Install **BlackHole 2ch** (free, from Existential Audio)
3. Create **Multi-Output Device** in macOS Audio MIDI Setup (Scarlett + BlackHole)
4. Set Ableton output to the Multi-Output Device
5. Install ableton-mcp server and add to Claude Code MCP settings

---

## Dependencies

- `python-osc` — OSC client for AbletonOSC communication
- `mido` — MIDI file and message generation
- `sounddevice` — audio capture from BlackHole input
- `numpy`, `librosa`, `soundfile` — audio analysis (same stack as existing server)
- `fastmcp` — MCP server framework

---

## Data Flow — Example: "fix the muddiness on the bass"

1. Claude calls `capture_output(10)` → captures 10s of Ableton output via BlackHole
2. Claude calls `analyze_current_mix()` → librosa analysis finds buildup at 180Hz
3. Claude calls `get_tracks()` → finds bass track at index 2
4. Claude calls `get_devices(2)` → finds EQ Eight at device index 0
5. Claude calls `get_device_params(2, 0)` → gets all EQ band params
6. Claude calls `set_device_param(2, 0, "Band 2 Gain", -4.0)` → cuts 180Hz by 4dB
7. Claude reports back: "Cut 4dB at 180Hz on the bass EQ — hit play and let me know if that's better"

---

## Error Handling

- AbletonOSC not running → clear message: "Open Ableton and make sure AbletonOSC is loaded in your session"
- BlackHole not installed → guide user through setup
- Track/device index out of range → refresh session state and retry
- OSC timeout → retry once, then report connection issue

---

## What's Out of Scope (v1)

- Audio recording from microphone (complex routing, defer to v2)
- VST plugin scanning/management
- Ableton browser control
- Automation curve editing
