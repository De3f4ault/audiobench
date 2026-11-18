# AudioBench

A personal audio intelligence system. Transcribes audio, chunks it into semantic units, embeds those into a vector store, and runs a background daemon that keeps it all in sync — so that everything you've ever listened to or recorded is searchable, queryable, and available for AI-assisted analysis, entirely offline.

> No API keys required · No cloud · No data leaves your machine

**Python 3.10+** · **MIT License** · **CPU-first** (`int8` quantization, no GPU required)

---

## What It Actually Does

Most transcription tools stop at the transcript. AudioBench treats the transcript as the beginning of a pipeline, not the end.

When you transcribe a file, the result lands in SQLite. A background daemon picks it up, runs it through a semantic chunker, embeds each chunk with a local Nomic model (768-dimensional vectors), and writes the vectors atomically into LanceDB. On startup the daemon loads all those vectors into an in-memory HNSW graph for sub-millisecond nearest-neighbour lookup. When you type a search query, three retrieval streams — BM25 full-text, dense vector ANN, and ColBERT late interaction — run in parallel and their results are merged with Reciprocal Rank Fusion before a cross-encoder re-ranks the top candidates. The daemon also runs a small set of intelligence tasks on a schedule: a pattern detector that finds thematically similar expressions across your corpus, a connection surfer that proposes links between ideas, a drift detector and blind spot detector. These write their observations back into the expression graph as first-class records.

Everything runs locally. The daemon is a long-running asyncio process that holds the ML models resident in RAM and accepts requests from CLI commands over a Unix domain socket.

---

## Architecture

```
Audio file
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Transcription Engine  (faster-whisper, int8 quantization)  │
│  Optional: ECAPA-TDNN speaker diarization                    │
└────────────────────────┬────────────────────────────────────┘
                         │  word-level timestamps
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  SQLite  (single database file: transcriptions.db)          │
│                                                              │
│  audio_files ──< transcriptions ──< segments                │
│  works ──< audio_files                                       │
│  expressions ──< expression_relations                        │
│  bookmarks, chapters, job_queue, chat_conversations, ...     │
└────────────────────────┬────────────────────────────────────┘
                         │  is_indexed = 0  →  daemon picks up
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  RAG Sweep  (background thread, every 300 seconds)          │
│                                                              │
│  1. content_aware_router() cleans and chunks the text       │
│     - Short texts: single chunk                             │
│     - Diarized: split by speaker turn, then chunk each      │
│     - Long texts: AdvancedSemanticChunker                   │
│       (sliding window cosine similarity, NLTK sentences,    │
│        percentile-based breakpoints, SentenceSplitter cap)  │
│                                                              │
│  2. ExpressionRepository.register()                         │
│     - SHA-256 content hash deduplification per source_type  │
│     - Returns existing record if hash matches               │
│     - Writes new ExpressionRecord to SQLite                 │
│                                                              │
│  3. batch_write_nodes() → LanceDB                           │
│     - merge_insert("expression_id") — atomic upsert         │
│     - No delete+add race: one transaction per batch         │
│     - Scalar index on expression_id for O(1) lookup         │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐   ┌─────────────────────────────────┐
│  LanceDB            │   │  AutocompleteIndex (hnswlib)     │
│  (expressions table)│   │                                 │
│  768-dim vectors    │   │  Built at daemon startup from   │
│  nomic-embed-text   │   │  all LanceDB vectors.           │
│  -v1.5              │   │  M=32, ef_construction=200      │
│                     │   │  Cosine space, 768-dim          │
│  Parses queries via │   │  O(log n) ANN lookup            │
│  three streams:     │   │  No model call at lookup time   │
│  - Dense (ANN)      │   │  — query already embedded by    │
│  - BM25 (FTS5)      │   │    the daemon's warm model      │
│  - ColBERT rerank   │   └─────────────────────────────────┘
└─────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Query Engine                                                │
│                                                              │
│  Three retrieval streams in parallel (ThreadPoolExecutor):  │
│  - FTS5Stream: BM25 keyword search over segments_fts        │
│  - DenseStream: ANN search via LanceDB + Nomic              │
│  - ColBERTStream: late-interaction scoring via cross-encoder │
│                                                              │
│  Merged with Reciprocal Rank Fusion (rrf_merge)             │
│  Re-ranked by ms-marco-MiniLM-L-6-v2 cross-encoder          │
│  Synthesised by local Ollama LLM                            │
│  Results cached in QueryCacheStore (semantic similarity)    │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Intelligence Layer  (asyncio scheduler, daemon process)    │
│                                                              │
│  PatternDetector  — hourly, finds expressions with high     │
│    cosine similarity across the corpus, proposes relations  │
│  ConnectionSurfer — finds cross-source thematic links       │
│  DriftDetector    — detects vocabulary/topic shift over time│
│  BlindSpotDetector — identifies underrepresented topics     │
│  ProposalGenerator — surfaces operator proposals from the   │
│    above signals for user review                            │
│                                                              │
│  Outputs are ExpressionRecords with source_type             │
│  'system_inference', 'drift_observation', etc.              │
│  They live in the same graph as transcript expressions.     │
│  User can confirm or reject via CLI.                        │
│                                                              │
│  Note: these tasks run and write outputs, but the volume   │
│  and quality of their proposals depends on corpus size.    │
│  With a small corpus they produce little signal.           │
└─────────────────────────────────────────────────────────────┘
```

---

## The Daemon

The daemon is not an optional add-on — it is the memory and intelligence layer. Without it, you have a transcription tool. With it, you have a persistent knowledge system.

```bash
audiobench daemon start    # start in background
audiobench daemon status   # health, queue depth, model info, uptime
audiobench daemon stop     # graceful shutdown
```

**What happens at startup (in order):**

1. ML models loaded into RAM: Nomic embedder, cross-encoder reranker, boundary embedder for chunking
2. `MemoryStore` connects to LanceDB; creates `expressions` table + scalar index if needed
3. `SweepState` loads from SQLite: all content hashes, all LanceDB expression IDs, all pending segment IDs, all unindexed transcript IDs — into in-memory sets and deques for O(1) sweep-tick access
4. Startup recovery runs three idempotent steps: ghost job cleanup (dead PIDs), unindexed expression backfill (SQLite records absent from LanceDB), work-assignment backfill
5. `AutocompleteIndex` builds an HNSW graph from all LanceDB expression vectors
6. Intelligence scheduler starts: `PatternDetector`, `DriftDetector`, `ConnectionSurfer`, `BlindSpotDetector`, `ProposalGenerator` registered and running on their own intervals
7. Unix socket server opens at the configured path
8. RAG sweep thread starts (every 300 seconds, runs in a background daemon thread)
9. Pipeline recovery sweep thread starts

**The sweep loop** runs every 5 minutes. Each tick:
- Pulls unindexed transcript IDs from the in-memory deque (no SQL)
- Chunks each transcript, deduplicates by content hash, registers expressions in SQLite, batch-embeds and writes to LanceDB
- Pulls pending segment IDs in sub-batches of 64 (measured: 256-segment batches peaked at ~8 GB RSS; 64-segment sub-batches keep each forward pass under the intelligence-task gate threshold)
- After each sub-batch calls `malloc_trim(0)` to return freed glibc arenas to the OS
- Triggers LanceDB compaction automatically when unoptimized write count crosses a threshold

**The socket protocol** is newline-delimited JSON:
```
{"cmd": "search", "args": {"query": "...", "top_k": 5}, "request_id": "abc123"}
→ {"status": "ok", "success": true, "data": {...}, "request_id": "abc123"}
```

Commands served: `ping`, `search`, `embed`, `chunk`, `rerank`, `embed_query`, `embed_segment`, `autocomplete`, `check_cache`, `write_cache`, `pipeline`, `operate`, `get_inferences`, `get_proposals`, `confirm_inference`, `reject_inference`, `authorize_proposal`, `optimize`, `status`, `reload_settings`.

---

## Storage

Two stores, kept in strict parity:

### SQLite (`transcriptions.db`)

The relational spine of the system. Core tables:

| Table | Purpose |
|---|---|
| `audio_files` | Source file metadata, SHA-256 hash, duration, format |
| `transcriptions` | One row per transcription run; `is_indexed` flag drives the sweep |
| `segments` | Individual timed segments within a transcription, word timestamps, `vector_indexed` flag |
| `expressions` | Semantic chunks: content, SHA-256 hash, source type, speaker, embedding model version |
| `expression_relations` | Directed edges between expressions (parent/child, source, relation) |
| `works` | Semantic grouping of audio files by title/author |
| `bookmarks` | Timestamp markers and region annotations |
| `chapters` | Chapter metadata for structured audio |
| `job_queue` | Background job tracking with PID and status |
| `chat_conversations` / `chat_messages` | Persistent AI chat history |
| `notes` | Free-form notes attached to transcriptions |

26 SQL migrations + 12 Python migrations. Migrations are idempotent and run at startup.

### LanceDB (`data/lancedb/`)

The vector store. One table: `expressions`.

Schema (`ExpressionNode`): `expression_id` (int), `content` (str), `vector` (768-dim float32), `embedding_model_version`, `embedded_at`, `source_type`, `speaker`, `audio_file_id`, `confidence`, `original_language`, `work_id`.

Writes use `merge_insert("expression_id")` — a single atomic transaction that inserts new rows and updates existing ones, eliminating the delete+add race condition that previously caused expressions to be lost during concurrent `optimize()` calls.

A scalar index on `expression_id` is created at startup. Idempotent on existing tables (confirmed empirically against lancedb==0.25.2 — re-verify on version bump; not a documented API guarantee).

LanceDB compaction runs automatically in a background thread when write count crosses a configurable threshold, and can be triggered on demand via `audiobench daemon optimize`.

---

## Search

```bash
# Full-text keyword search
audiobench search "keyword"

# Semantic memory search (requires daemon)
audiobench memory search "what did we decide about the API design"

# Ask a question across your entire corpus
audiobench memory ask "what themes appear across all my meetings"
```

The `memory search` path:
1. Query reformulated into BM25 keywords + dense embedding string
2. Three streams run in parallel: FTS5 BM25 over `segments_fts`, dense ANN over LanceDB, ColBERT late-interaction re-scoring
3. Results merged with Reciprocal Rank Fusion
4. Top candidates re-ranked by cross-encoder
5. Semantic cache checked before the LLM synthesis step (cosine distance threshold 0.05 — cache hit skips the model call entirely)
6. Local Ollama synthesises an answer with the ranked segments as context

---

## Transcription

```bash
audiobench transcribe meeting.m4a
audiobench transcribe meeting.m4a -f srt
audiobench transcribe meeting.m4a --diarize --engine whisper
audiobench transcribe --accurate interview.wav
audiobench transcribe --enhance --trim --denoise noisy.m4a
```

Engine: `faster-whisper` with `int8` CPU quantization. Models: `tiny` through `large-v3-turbo`. Speaker diarization via ECAPA-TDNN voice print matching.

Audio preprocessing chain (when flags are combined): `highpass → denoise (RNNoise) → trim silence → EBU R128 loudness normalization`. The `--denoise` flag supersedes the spectral denoiser inside `--enhance` to avoid double-processing. Use `--check` to preview the exact filter chain before running.

---

## What You Can Do

| Command | What it does |
|---|---|
| `audiobench transcribe file.m4a` | Offline speech-to-text with word timestamps |
| `audiobench listen` | Real-time microphone transcription |
| `audiobench transcribe --diarize` | Identify who said what |
| `audiobench chat 3` | AI chat with transcript #3 as context |
| `audiobench search "keyword"` | Full-text search across all transcripts |
| `audiobench memory search "query"` | Semantic vector search over the expression graph |
| `audiobench memory ask "question"` | RAG query: retrieve + synthesise with local LLM |
| `audiobench inspect file.m4a` | Waveform + spectrogram images |
| `audiobench speak "text"` | Offline TTS via Piper |
| `audiobench analyze file.m4a` | Loudness, silence regions, quality report |
| `audiobench bookmark auto 66` | AI auto-bookmarking with type classification |
| `audiobench daemon start` | Start the background intelligence daemon |
| `audiobench daemon status` | Daemon health, queue depth, uptime |
| `audiobench jobs list` | Background audio processing queue |
| `audiobench db status` | SQLite migration state, row counts |

---

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/de3f4ault/audiobench.git
cd audiobench

# 2. Install
make install        # creates venv + installs deps
source venv/bin/activate

# 3. Install FFmpeg (required)
sudo pacman -S ffmpeg        # Arch
# sudo apt install ffmpeg    # Ubuntu/Debian
# brew install ffmpeg        # macOS

# 4. Transcribe
audiobench transcribe meeting.m4a

# 5. Start the daemon (enables search, memory, intelligence)
audiobench daemon start
```

The first transcription run downloads the Whisper model (~1.5 GB) to `~/.audiobench/models/`. The daemon's first startup loads three ML models into RAM (~1–2 GB depending on model size).

---

## Installation

```bash
make install    # base (transcription only)
make dev        # full (all extras: docs, dev tools, TTS, streaming, AI)
```

Or manually:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

**FFmpeg** is required for audio conversion and preprocessing.

---

## Configuration

Settings load in priority order: CLI flags → environment variables (`AUDIOBENCH_*`) → `.env` file → defaults.

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `AUDIOBENCH_MODEL_NAME` | `large-v3-turbo` | Whisper model |
| `AUDIOBENCH_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `AUDIOBENCH_COMPUTE_TYPE` | `int8` | `int8` (CPU), `float16` (CUDA) |
| `AUDIOBENCH_LANGUAGE` | *(auto-detect)* | e.g. `en`, `sw`, `fr` |
| `AUDIOBENCH_SPEED_PRESET` | `balanced` | `fast`, `balanced`, `accurate` |
| `AUDIOBENCH_LOG_LEVEL` | `WARNING` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `AUDIOBENCH_BOOKMARK_MODEL` | `qwen3-coder:480b-cloud` | Ollama model for AI bookmarking |

---

## Directory Layout

```
audiobench/
├── data/                      ← Managed workspace (gitignored)
│   ├── transcriptions.db      ← SQLite: all relational data
│   ├── lancedb/               ← LanceDB: vector store
│   ├── library/               ← Managed audio file collection
│   ├── exports/               ← SRT, VTT, JSON, PDF outputs
│   ├── reports/               ← AI summaries
│   ├── logs/                  ← Daemon and job logs
│   ├── plugins/               ← Custom Python CLI commands
│   └── presets/               ← Named transcription presets (TOML)
├── .env                       ← Configuration (gitignored)
└── .env.example               ← Configuration template

~/.audiobench/
├── models/                    ← Whisper models (~1.5 GB each)
│   └── rnnoise/               ← RNNoise denoise model (~293 KB)
└── voices/                    ← Piper TTS voice models
```

---

## Project Structure

```
src/audiobench/
├── core/                      ← Settings, logging, DB engine, session factory
├── daemon/
│   ├── server.py              ← asyncio Unix socket server, all command handlers
│   ├── sweep_state.py         ← In-memory O(1) state: hash sets, deques
│   ├── startup_recovery.py    ← Idempotent recovery steps run at each startup
│   ├── pipeline_recovery.py   ← Stuck pipeline detection and repair
│   ├── autocomplete.py        ← HNSW in-memory autocomplete index (hnswlib)
│   ├── lancedb_optimizer.py   ← Background LanceDB compaction
│   └── intelligence/
│       ├── scheduler.py       ← Asyncio task scheduler with CPU-gate
│       ├── pattern_detector.py
│       ├── connection_surfer.py
│       ├── drift_detector.py
│       ├── blind_spot_detector.py
│       ├── proposal_generator.py
│       ├── calibration.py     ← Confirm/reject tracking
│       └── operator_registry.py
├── memory/
│   ├── chunking.py            ← content_aware_router, AdvancedSemanticChunker
│   ├── embedding_engine.py    ← Nomic model wrapper, batch embedding
│   ├── memory_store.py        ← LanceDB adapter, merge_insert, scalar index
│   ├── query_engine.py        ← Three-stream retrieval, RRF fusion, synthesis
│   ├── retrieval_streams.py   ← FTS5Stream, DenseStream, ColBERTStream
│   ├── rrf_fusion.py          ← Reciprocal Rank Fusion
│   ├── singletons.py          ← Model lifecycle, inference locks
│   └── query_reformulator.py  ← Query expansion for BM25 + dense
├── storage/
│   ├── models.py              ← SQLAlchemy ORM models
│   ├── expression_repository.py ← Expression CRUD, SHA-256 dedup
│   ├── repository.py          ← Transcription CRUD
│   ├── bookmark_repository.py
│   └── migrations/            ← 26 SQL + 12 Python migrations
├── jobs/                      ← Background job queue (runner, worker, repository)
├── observatory/               ← Event bus, structured logging to journal.db
├── transcribe/                ← faster-whisper pipeline, audio filters, engines
├── chat/                      ← AI chat, Ollama provider, context builder
├── cli/                       ← Click commands, REPL, plugin system
├── tts/                       ← Piper TTS
├── diarization/               ← ECAPA-TDNN speaker diarization
└── streaming/                 ← Live microphone transcription

tests/                         ← Test suite
├── conftest.py                ← Shared fixtures (temp DB, patched settings)
├── test_core/
├── test_cli/
├── test_memory/               ← merge_insert race tests, chunking tests
└── test_storage/
```

---

## Make Targets

```bash
make help              # All targets
make install           # Base install
make dev               # Dev install (editable, all extras)
make test              # Test suite with coverage
make lint              # ruff + mypy
make format            # black + ruff
make transcribe FILE=audio.m4a
make listen            # Live microphone transcription
make speak TEXT="Hello"
make repl              # Interactive shell
make doctor            # System health check
```

> **Testing in Sandboxed Environments:** Tests executed within sandboxed runner environments with filesystem isolation may report ~12 spurious failures in `test_memory/` (e.g., `test_chunking.py`, `test_memory_voice.py`) because NLTK data resides at `~/nltk_data/` (outside the workspace) and the sandbox restricts access there. Run with sandbox bypass or from a standard shell with `~/nltk_data/` access. The memory suite is 192/192 green when run in a standard environment.

---

## Debug Mode

```bash
audiobench -v transcribe meeting.m4a      # INFO logs
audiobench --debug transcribe meeting.m4a # DEBUG logs
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
