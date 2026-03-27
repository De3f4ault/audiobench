# Architecture

Technical reference for the AudioBench codebase.

---

## Module Diagram

```
┌─────────────────────────────────────────────────────────┐
│                        CLI Layer                        │
│  cli/commands/           ── 26+ auto-discovered commands  │
│  cli/repl/               ── interactive shell (6 modules) │
│  cli/plugins/            ── user plugin loader + groups   │
│  cli/display/            ── theme, phase tracker          │
│  cli/io/                 ── file collection, output paths │
└──────────────────────────┬──────────────────────────────┘
                           │ on_phase(), on_segment()
┌──────────────────────────▼─────────────────────────────┐
│                     Pipeline Layer                      │
│  transcribe/transcriber.py ── orchestrates full workflow│
│    load → filter → convert → transcribe → save           │
└────────┬──────────────────┬────────────────┬───────────┘
         │                  │                │
┌────────▼────────┐ ┌──────▼───────┐ ┌──────▼───────────┐
│  Audio Loader   │ │    Engine    │ │     Storage      │
│  transcribe/    │ │  transcribe/ │ │  storage/        │
│  audio_         │ │  engines/    │ │  models.py       │
│  converter.py   │ │  whisper_    │ │  repository.py   │
│  FFmpeg probe,  │ │  engine.py   │ │                  │
│  convert, filter│ │  faster-     │ ├──────────────────┤
│  analyze, chain │ │  whisper     │ │  core/           │
│  builder        │ │  batched +   │ │  db_engine.py    │
└─────────────────┘ │  sequential  │ │  db_session.py   │
                    └──────────────┘ └──────────────────┘
```

---

## Data Flow

```
Input file (m4a/mp3/wav/...)
    │
    ▼
┌─ FFmpeg ────────────────────────────────────────────┐
│  1. Probe: extract codec, duration, sample rate     │
│  2. Convert: → 16kHz mono WAV (+ optional filters) │
└────────────────────────────┬────────────────────────┘
                             │
    ▼
┌─ Cache Check ──────────────────────────────────────┐
│  SHA-256 hash of input file                         │
│  If match found in DB → return cached transcript    │
└────────────────────────────┬───────────────────────┘
                             │ (cache miss)
    ▼
┌─ Whisper Engine ───────────────────────────────────┐
│  faster-whisper (CTranslate2 backend)              │
│  Batched mode: BatchedInferencePipeline            │
│  Sequential mode: WhisperModel.transcribe()        │
│                                                     │
│  For each segment:                                  │
│    → Skip low-confidence (avg_logprob < -1.5)      │
│    → collapse_repetitions() + fix_broken_words()   │
│    → progress_callback(pct)                         │
│    → on_segment(segment)                            │
└────────────────────────────┬───────────────────────┘
                             │
    ▼
┌─ Output ───────────────────────────────────────────┐
│  Formatter: txt / srt / vtt / json                  │
│  → File or terminal                                 │
│  → SQLite database (always, for history/search)     │
└────────────────────────────────────────────────────┘
```

---

## Configuration Precedence

```
CLI flags (--language en, --fast, -m small)
    ↓ overrides
Environment variables (AUDIOBENCH_LANGUAGE=en)
    ↓ overrides
.env file (AUDIOBENCH_LANGUAGE=en)
    ↓ overrides
Default values (settings.py)
```

Managed by Pydantic Settings (`src/audiobench/core/settings.py`).

---

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `TranscriptionPipeline` | `transcribe/transcriber.py` | Orchestrates load → transcribe → save |
| `WhisperEngine` | `transcribe/engines/whisper_engine.py` | faster-whisper wrapper (batched + sequential) |
| `AudioBenchSettings` | `core/settings.py` | Pydantic settings with env var binding |
| `TranscriptionRepository` | `storage/repository.py` | SQLAlchemy CRUD for transcription records |
| `PhaseTracker` | `cli/display/phase_tracker.py` | Progress display (Live → streaming transition) |
| `ReplSession` | `cli/repl/session.py` | Interactive shell state, context tracking |
| `Transcript` / `Segment` / `Word` | `transcribe/transcription_result.py` | Pydantic data models |
| `AudioBenchGroup` | `cli/plugins/custom_group.py` | Fuzzy command matching + suggestions |
| `BookmarkRepository` | `storage/bookmark_repository.py` | Bookmark CRUD, Audacity label import/export |

---

## Audio Processing Pipeline

`build_filter_chain()` in `transcribe/audio_converter.py` encodes the optimal filter ordering:

```
highpass (200Hz) → [arnndn | afftdn] → silenceremove → loudnorm
```

Rules:

1. **highpass** always first when any cleaning is active
2. **arnndn** (neural) supersedes **afftdn** (spectral) — no double denoising
3. **silenceremove** after denoise, before normalization
4. **loudnorm** always last (EBU R128, I=-16 LUFS)

The RNNoise model (`bd.rnnn`, ~293 KB) is auto-downloaded on first `--denoise` use to `~/.audiobench/models/rnnoise/`.

---

## Speed Presets (Internal)

Resolved in `core/settings.py`:

| Preset | `beam_size` | `batch_size` | `temperature` | `condition_on_previous_text` |
|--------|-------------|--------------|---------------|------------------------------|
| fast | 1 | 8 | `0` | `False` |
| balanced | 3 | 4 | `[0, 0.2, 0.4, 0.6, 0.8, 1.0]` | `False` |
| accurate | 5 | 1 | `[0, 0.2, 0.4, 0.6, 0.8, 1.0]` | `True` |

---

## Database Schema

SQLite via SQLAlchemy. Three tables: `audio_files`, `transcriptions`, `segments`.

| Table | Key Columns |
|-------|-------------|
| `audio_files` | `id`, `file_hash` (SHA-256), `file_name`, `file_path`, `duration_seconds` |
| `transcriptions` | `id`, `audio_file_id` (FK), `full_text`, `language`, `model_name`, `engine`, `created_at` |
| `segments` | `id`, `transcription_id` (FK), `text`, `start`, `end`, `speaker` |
| `bookmarks` | `id`, `audio_file_id` (FK), `timestamp`, `end_timestamp`, `name`, `type`, `notes` |
| `chat_sessions` | `id`, `title`, `model`, `created_at` |
| `chat_messages` | `id`, `session_id` (FK), `role`, `content`, `thinking` |

---

## REPL Architecture

The interactive shell (`cli/repl/`) is split into 6 focused modules:

| Module | Purpose |
|--------|---------|
| `__init__.py` | Main input loop + Click command |
| `session.py` | State management, context tracking, history |
| `dispatch.py` | Command routing, ID capture |
| `dot_commands.py` | `.stats`, `.segments`, `.find`, `.play`, `.edit`, etc. |
| `slash_commands.py` | `/help`, `/exit`, alias mapping |
| `completion.py` | Tab completion setup |
| `banner.py` | Welcome text, onboarding, goodbye |

Key features:

- **Context tracking**: `set_context(id)` loads a transcript record
- **ID injection**: Commands like `show`, `ask` auto-inject the context ID
- **Variable expansion**: `$last`, `$id` expand to context ID
- **Fuzzy matching**: Typos like `.sarch` suggest `.search`
- **History persistence**: Command history saved to `data/repl_history`

---

## Plugin Architecture

Plugins are Python files in `data/plugins/` loaded via `cli/plugins/loader.py`:

1. `discover_plugins()` — Scans directory for `.py` files (ignores `_` prefixed)
2. `load_plugin()` — Uses `importlib.util` to load each module
3. `register_plugins()` — Calls `register(cli)` if defined, or auto-registers Click commands

Plugins load **after** all built-in commands, so they can extend or override functionality.

---

## Command Auto-Discovery

Built-in commands use `pkgutil`-based auto-discovery (`cli/commands/__init__.py`):

1. Scans `cli/commands/` for `.py` modules
2. Imports each module dynamically
3. Registers any `click.Command` objects found at module level
4. Also registers the REPL command and user plugins

To add a new command: drop a `.py` file with a `@click.command()` in `cli/commands/`. No registration code to edit.

---

## Data Directory Layout

```
data/                    ← Project-local (gitignored)
├── transcriptions.db    ← SQLite database
├── plugins/             ← User plugins
├── presets/             ← Config presets (TOML)
├── logs/                ← App logs
├── sessions/            ← Live STT sessions
└── repl_history         ← REPL history

~/.audiobench/           ← Shared across projects
├── models/              ← Whisper models (multi-GB)
└── voices/              ← Piper TTS voices
```

---

## PhaseTracker Display Architecture

The `PhaseTracker` in `cli/display/phase_tracker.py` uses a two-mode approach:

1. **Live mode** — During loading/converting, Rich Live shows animated spinners (uses `transient=True`)
2. **Streaming mode** — When the first transcript segment arrives, Live stops (frame vanishes), phases print statically at the top, and each segment prints below via `console.print()`. Text grows downward in real-time.
