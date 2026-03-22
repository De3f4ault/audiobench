# Usage Guide

Real-world workflows for the AudioBench CLI.

---

## 1. Basic Transcription

### Print to terminal

```bash
audiobench transcribe meeting.m4a
```

The transcript prints directly to your terminal. Nothing is saved to a file, but it **is** cached in the database for later retrieval via `history` or `search`.

### Save to file

```bash
# Auto-detect format from filename
audiobench transcribe meeting.m4a -o meeting.srt

# Explicitly set format (saves as meeting.srt next to the input file)
audiobench transcribe meeting.m4a -f srt

# Save to a specific directory
audiobench transcribe meeting.m4a -o ~/Documents/transcriptions/
```

### Output format priority

| Flags | Result |
|-------|--------|
| `-o meeting.srt` | Format from extension → SRT |
| `-f srt` (no `-o`) | `<stem>.srt` next to input |
| `-o ./out/` (directory) | `./out/<stem>.<default-format>` |
| Neither `-o` nor `-f` | Print to terminal |

---

## 2. Batch Transcription

Transcribe multiple files at once using glob patterns:

```bash
# All m4a files in current directory
audiobench transcribe *.m4a

# Save all to a directory
audiobench transcribe *.m4a -o ./transcriptions/

# All audio files in a specific folder
audiobench transcribe ~/Music/recordings/*.m4a -f srt
```

A batch summary table is shown at the end with per-file stats.

---

## 3. Multilingual & Code-Switching

### Auto-detect language

By default, the model auto-detects the spoken language:

```bash
audiobench transcribe swahili_speech.m4a
```

### Force a language

```bash
audiobench transcribe interview.m4a --language sw     # Swahili
audiobench transcribe lecture.m4a --language fr        # French
```

### Code-switching (mixed languages)

Use `--prompt` to guide the model when speakers switch between languages:

```bash
audiobench transcribe conversation.m4a \
  --prompt "Conversation in Swahili and English, with occasional Sheng slang"
```

The prompt doesn't force a language — it gives the model context to handle transitions better.

---

## 4. Speed vs Accuracy

### Presets

```bash
audiobench transcribe --fast lecture.mp3       # Quick draft, lower quality
audiobench transcribe meeting.m4a              # Balanced (default)
audiobench transcribe --accurate interview.wav # Best quality, slowest
```

### What the presets change

| | Fast | Balanced | Accurate |
|-----|------|----------|----------|
| Beam size | 1 | 3 | 5 |
| Batch size | 8 | 4 | 1 (sequential) |
| Temperature | 0 (no fallback) | Fallback chain | Fallback chain |
| Context conditioning | No | No | Yes |
| Speed | ~2x faster | Default | ~0.5x slower |

### Choose a different model

Smaller models are faster but less accurate:

```bash
audiobench transcribe lecture.mp3 -m small     # ~461 MB, fastest
audiobench transcribe lecture.mp3 -m medium    # ~1.5 GB, good balance
audiobench transcribe lecture.mp3              # large-v3-turbo (default)
```

---

## 5. Audio Processing

### Audio enhancement flags

AudioBench has three pre-processing flags that can be combined freely:

```bash
# Spectral noise reduction + loudness normalization
audiobench transcribe noisy_recording.m4a --enhance

# Remove leading/trailing silence
audiobench transcribe lecture.m4a --trim

# AI neural noise reduction (RNNoise)
audiobench transcribe interview.m4a --denoise

# All three — maximum quality
audiobench transcribe meeting.m4a --enhance --trim --denoise
```

### Processing order (smart filter chain)

Filters are always applied in the optimal order, regardless of how you specify them:

```
highpass (200Hz) → denoise/enhance → silence trim → loudness normalization
```

| Flags | Actual Filter Chain |
|-------|-------------------|
| `--enhance` | highpass → afftdn → loudnorm |
| `--denoise` | highpass → arnndn → loudnorm |
| `--trim` | silenceremove (standalone) |
| `--enhance --denoise` | highpass → arnndn → loudnorm |
| `--enhance --trim` | highpass → afftdn → trim → loudnorm |
| `--denoise --trim` | highpass → arnndn → trim → loudnorm |
| All three | highpass → arnndn → trim → loudnorm |

> **Key rule**: When `--denoise` is active, it completely replaces the spectral denoiser (`afftdn`) from `--enhance`. This prevents double denoising, which causes metallic artifacts and softened consonants. The `loudnorm` normalization from `--enhance` is still applied.

### When to use each flag

| Situation | Recommended Flags |
|-----------|------------------|
| Clean studio recording | None needed |
| Moderate background noise | `--enhance` |
| Heavy noise (traffic, HVAC, crowd) | `--denoise` |
| Long pauses / dead air | `--trim` |
| Phone recording with noise + silence | `--denoise --trim` |
| Noisy recording for transcription | `--denoise --trim` |

### First-run model download

The `--denoise` flag uses a neural noise reduction model (RNNoise, ~293 KB). On first use, it is automatically downloaded from GitHub to `~/.audiobench/models/rnnoise/bd.rnnn`. Subsequent runs use the cached model instantly.

### Custom FFmpeg filters

```bash
audiobench transcribe recording.m4a --filter "highpass=f=300,lowpass=f=3000"
```

### Preview filter chain before transcribing

```bash
audiobench transcribe --check --enhance --denoise recording.m4a
```

Shows codec, duration, sample rate, channels, bitrate, and the exact filter chain — without starting transcription. Use this to verify your processing pipeline.

---

## 6. Audio Analysis & Utilities

### Analyze audio quality

```bash
audiobench analyze meeting.m4a
```

Shows loudness statistics (integrated LUFS, loudness range, true peak), silence regions, and recommends which flags to use.

### Convert between formats

```bash
audiobench convert recording.m4a -o recording.mp3
audiobench convert recording.wav -o recording.opus --bitrate 48k
audiobench convert recording.m4a -o recording.flac        # Lossless
audiobench convert lecture.mp3 -o fast.mp3 --speed 1.5     # Speed up
```

### Merge audio files

```bash
audiobench merge part1.wav part2.wav part3.wav -o full.wav
```

### Generate waveform / spectrogram images

```bash
audiobench inspect recording.m4a                    # Both images
audiobench inspect recording.m4a --waveform         # Waveform only
audiobench inspect recording.m4a --spectrum          # Spectrogram only
```

---

## 7. History Management

### View past transcriptions

```bash
audiobench history              # Last 20
audiobench history --limit 50   # Last 50
```

Shows ID, filename, language, duration, word count, and timestamp for each.

### Search by content

```bash
audiobench search "yoga"
audiobench search "important meeting" --limit 10
```

Full-text search across all stored transcriptions.

### Re-export in a different format

```bash
audiobench export 3 -f vtt              # Export ID #3 as WebVTT
audiobench export 3 -f json -o data/    # As JSON to data/ directory
```

### Delete

```bash
audiobench delete 3          # Delete transcription #3
audiobench delete --all      # Wipe all history
```

---

## 8. Interactive REPL

The REPL provides a full interactive environment for working with transcriptions.

### Launch

```bash
audiobench repl
```

### Context-aware workflow

```
audiobench> history                     # List transcriptions
audiobench> show 3                      # View transcript → sets context
audiobench [#3 meeting.m4a]> .stats     # Quick stats
audiobench [#3 meeting.m4a]> .find "deadline"  # Search within transcript
audiobench [#3 meeting.m4a]> .play      # Play source audio
audiobench [#3 meeting.m4a]> .play 01:25  # Play from timestamp
audiobench [#3 meeting.m4a]> .edit      # Edit in $EDITOR, saves to DB
audiobench [#3 meeting.m4a]> .next      # Jump to next transcript
audiobench [#3 meeting.m4a]> summarize  # AI summarize (auto-injects ID)
audiobench [#3 meeting.m4a]> .close     # Clear context
audiobench> .help                       # Show all dot-commands
```

All CLI commands work inside the REPL. Context-aware commands (`show`, `ask`, `summarize`, `export`, etc.) auto-inject the active transcript ID.

---

## 9. Piping & Scripting

### Quiet mode for piping

```bash
# Grep for specific words
audiobench transcribe -q meeting.m4a | grep "deadline"

# Pipe to a file
audiobench transcribe -q meeting.m4a > transcript.txt

# Word count
audiobench transcribe -q meeting.m4a | wc -w
```

The `-q` / `--quiet` flag suppresses all UI and prints only the raw transcript.

---

## 10. Caching

Transcriptions are cached automatically by file hash. Re-transcribing the same file returns instantly:

```bash
audiobench transcribe meeting.m4a         # First run: full transcription
audiobench transcribe meeting.m4a         # Second run: instant cache hit
```

Force a fresh transcription:

```bash
audiobench transcribe --no-cache meeting.m4a
```

---

## 11. Pre-downloading Models

Download models ahead of time for offline use:

```bash
audiobench download large-v3-turbo     # Default model (~1.5 GB)
audiobench download small              # Smaller model (~461 MB)
```

Models are stored in `~/.audiobench/models/`.

---

## 12. Configuration Presets

Save and reuse named configurations:

```bash
# Create presets for different use cases
audiobench preset create meeting --model large-v3 --speed accurate --enhance
audiobench preset create podcast --language en --format srt
audiobench preset create quick --speed fast --model base

# Use when transcribing
audiobench transcribe file.m4a --preset meeting

# Manage presets
audiobench preset list
audiobench preset show meeting
audiobench preset delete meeting
```

Presets are stored as TOML files in `data/presets/`.

---

## 13. System Maintenance

### Health check

```bash
audiobench doctor
```

Checks FFmpeg, faster-whisper, piper-tts, ollama, CUDA, disk space, and database connectivity.

### Usage statistics

```bash
audiobench status
```

### Cleanup old data

```bash
audiobench cleanup --older-than 30d        # Delete old transcriptions
audiobench cleanup --cache                  # Remove model cache
audiobench cleanup --temp                   # Remove temp files
audiobench cleanup --older-than 7d --dry-run # Preview deletions
```

### Shell completions

```bash
audiobench install-completion fish     # or bash, zsh
```

---

## 14. Configuration

### Using `.env`

```bash
cp .env.example .env
# Edit .env with your preferred defaults
```

### Using environment variables

```bash
AUDIOBENCH_LANGUAGE=en audiobench transcribe meeting.m4a
AUDIOBENCH_SPEED_PRESET=fast audiobench transcribe *.m4a
```

### Check current settings

```bash
audiobench info
```

---

## Troubleshooting

### "FFmpeg not found"

Install FFmpeg:

```bash
sudo pacman -S ffmpeg       # Arch
sudo apt install ffmpeg     # Ubuntu/Debian
```

### First run is slow

The model (~1.5 GB) downloads on first use. Pre-download with:

```bash
audiobench download large-v3-turbo
```

### Poor accuracy

1. Try `--accurate` preset
2. Use `--prompt` to provide context
3. Use `--denoise` for noisy recordings (better than `--enhance` for heavy noise)
4. Use `--trim` to remove long silences that can cause hallucinations
5. Try forcing the language with `--language en`

### Out of memory

Use a smaller model:

```bash
audiobench transcribe -m small recording.m4a
```
