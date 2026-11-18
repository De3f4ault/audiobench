# Changelog

All notable changes to AudioBench will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-07-24

### Added

- **Daemon & Intelligence Engine**:
  - Full background daemon service (`audiobench daemon`) operating over UNIX domain sockets with single-instance lock file enforcement (`.audiobench.lock`) and startup auto-recovery.
  - Intelligence loop engine featuring continuous drift detection (`drift_detector.py`), automatic indexing sweeps, calibration routines, and scheduled worker tasks.
  - Operator registry system and pipeline executor (`pipeline_executor.py`, `proposal_generator.py`, `proposal_guardrails.py`) for autonomous background intelligence workflows.
  - Alignment worker and pipeline recovery mechanism (`alignment_worker.py`, `pipeline_recovery.py`) for fault-tolerant background audio processing.
  - IPC daemon fast-path timeouts and parent paragraph deduplication in query engines.

- **LanceDB Vector Memory & RAG Pipeline**:
  - Embedded vector database memory store (`memory/memory_store.py`) providing hybrid vector search over audio expressions.
  - Multi-stream retrieval architecture: Reciprocal Rank Fusion (RRF), ColBERT reranker (`answerai-colbert-small-v1`), query reformulator (`query_reformulator.py`), memoir writer (`memoir_writer.py`), knowledge ingester (`knowledge_ingester.py`), and synthesis guardrails.
  - Atomic `merge_insert()` write path replacing the `delete()+add()` two-step to eliminate race condition data loss during concurrent table compactions (`optimize()`).
  - Idempotent scalar index on `expression_id` created during `MemoryStore` initialization for 1.7x faster upserts (12.9ms on 10k rows).
  - Schema-aligned PyArrow table serialization (`pa.Table.from_pylist(..., schema=tbl.schema)`) resolving runtime nullability mismatches with LanceModel.
  - `_force_offline_env()` context manager in `singletons.py` ensuring HuggingFace/SentenceTransformer offline environment variables stay active during deep lazy module initialization.
  - `disable_memory` configuration setting to bypass daemon and semantic embeddings when running in constrained environments.

- **Observatory Dashboard & Rich TUI**:
  - Full Graphical TUI Dashboard (`audiobench observatory`) with interactive Inferences and Proposals panels, confirm modal, and real-time telemetry.
  - Reimport TUI (`reimport_tui.py`) with batch file selection, conflict handling, overwrite support, and reverse import shortcuts.
  - Interactive chapter picker component (`chapter_picker.py`) with visual bookmark indicators.

- **Multi-GPU & Advanced Transcription**:
  - Multi-GPU parallel audio file transcription via `--parallel-gpus` flag.
  - Explicit audio file rename command with Gemini AI fallback (`rename_service.py`).
  - Upgraded default LLM model configuration to `qwen4`.
  - Audio file SHA-256 unique constraint collision handling during high-concurrency worker runs.

- **Database Migrations & Study Schema**:
  - Idempotent migration system executing migrations 006 through 025: FTS5 segment indexing, journal tracking, study projects (`study_models.py`), vector sync status, unified job queue, works schema (`works_schema.py`), and privacy tiers.
  - Note repository (`note_repository.py`) and backfill scripts (`backfill_expression_segment_map.py`).

- **FastAPI REST Web Service & React Frontend UI**:
  - Full REST API server (`src/audiobench/api/`) built with FastAPI providing endpoints for `transcribe`, `chat`, `history`, `jobs`, and `settings`.
  - Modern React + Vite web application (`web/`) featuring dedicated `ChatView`, `HistoryView`, `TranscribeView`, and `SettingsView` components.

- **PDF Export Engine**:
  - PDF document export generator (`export/pdf.py`) with customizable HTML template formatting (`export/templates/pdf_template.html`) for generating clean PDF transcript reports.

- **Decoupled Event Bus**:
  - Asynchronous event bus infrastructure (`events/bus.py`) powering real-time pub/sub telemetry across daemon loops, observatory TUI, and CLI views.

- **Interactive MPV Playback Controller**:
  - IPC-based `MpvController` backend (`playback/`) enabling interactive transcript synchronization, real-time lyrics display, hotkey seeking (±5s/60s), and speed adjustment (±10%, half/double).

- **Speaker Diarization & Voiceprint Verification**:
  - Speaker verification engine (`diarization/verification.py`) and multi-speaker clustering pipeline.

- **Automated Chaptering Engine**:
  - Topic transition and silence-based automatic audio chaptering module (`chapters/`).

- **CLI, REPL & Interactive Shell Enhancements**:
  - New `audiobench db` command group for database migration status, backfills, and database repairs.
  - New `audiobench memory` command group for direct query reformulation, vector store inspection, and knowledge base ingestion.
  - New `audiobench jobs` command group (`list`, `status`, `cancel`, `clear`) for asynchronous queue management.
  - Wizard prompt interactive inline editing powered by `readline` support.
  - Curses layout error handling on terminal resize and bottom-line rendering.
  - REPL session bridging, memory hints in chat, and expanded slash commands (`/memory`, `/daemon`, `/jobs`).

## [0.2.0] — 2026-03-27

### Added

- **Bookmark & Annotation System** — full-featured timestamp marking for audio files
  - Point bookmarks and region markers (Audacity-inspired dual marker model)
  - 5 bookmark types: [BM] bookmark, [HL] highlight, [TODO] todo, [NOTE] note, [CUT] edit
  - Interactive player keybindings (`b` point, `B` region, `n`/`p` jump, `l` cycle type)
  - Zero-interruption UX — bookmarks auto-named from transcript text at current position
  - Visual bookmark indicator bar and green flash feedback during playback
  - `audiobench bookmark` CLI group: `list`, `add`, `rename`, `note`, `type`, `rm`, `search`, `export`, `import`
  - `--bookmark` option to start playback from a saved position (by ID or name)
  - `--bookmarks` flag to list bookmarks before playback
  - `/bookmarks [ID]` slash command in chat REPL
  - Audacity label track export/import (`--format audacity`) alongside JSON
  - Auto-detection of import format (JSON vs TSV)
- Database migration `m005_bookmarks` (idempotent, runs automatically)
- **AI Auto-Bookmarking** — intelligent transcript annotation
  - `audiobench bookmark auto <ID>` — AI identifies key moments and creates structured bookmarks
  - Exact timestamp extraction from transcript segments (no rounding)
  - 5-type classification: [HL] highlight, [TODO] todo, [NOTE] note, [BM] bookmark, [CUT] edit
  - Length-scaled output (3–15 bookmarks based on duration)
  - `--model` override (default: `qwen3-coder:480b-cloud`), `--focus`, `--dry-run`
  - Configurable `AUDIOBENCH_BOOKMARK_MODEL` setting

## [0.1.0] — 2026-03-20

### Added

- Initial release
- Audio transcription with Faster Whisper, Vosk, and Google Gemini engines
- Interactive playback with synchronized lyrics display
- AI-powered chat with transcript context (Ollama, Gemini)
- Side-by-side model comparison mode
- Transcript search, history management, and export
- Speaker diarization support
- Text-to-speech via Piper TTS
- Live microphone transcription (streaming)
- Plugin system for user extensions
