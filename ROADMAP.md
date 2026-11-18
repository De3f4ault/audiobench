# AudioBench Roadmap

> **What this document is.**
> A living design intent document. Not a changelog — that is `CHANGELOG.md`.
> Not a task list — that lives in sprint planning.
> This is the *why*, the *where we are*, and the *where we are going*.
> Every future developer who reads this should understand the soul of the system,
> not just its surface.

---

## The Soul of AudioBench

AudioBench is a **personal knowledge retrieval and thinking tool** built around
audio. The recordings it indexes are not podcast content to be consumed — they
are conversations, late-night thoughts, lectures, and ideas that belong to *you*.
The design goal is to make your past thinking available to your present thinking,
without requiring you to remember where something was said or who said it.

The LLM is not the product. The fragments are the product. The LLM is a lens.
This distinction is the most important architectural principle in the system.
When the LLM is unavailable, AudioBench should still be genuinely useful — not
degraded, not apologetic, just different in what it surfaces.

---

## What Is Fully Built

### Transcription Engine
- Multi-engine: **Faster-Whisper** (local, default), **Gemini** (cloud, highest quality)
- Diarization: fast mode (VAD + AHC clustering) and accurate mode (Pyannote)
- Speaker auto-naming via Gemini (`--auto-name`)
- Manual speaker mapping (`--map-speakers`)
- Audio pre-processing: denoise (RNNoise), enhance, trim silence
- Background transcription (`--background`)
- Recursive batch processing with manifest files, extension filters, exclusion patterns
- Chapter-level transcription for large audiobooks
- Multiple output formats: txt, srt, vtt, json, all
- Collision handling: overwrite / skip / rename

### Memory & Retrieval (Search)
- **Three-stream parallel retrieval**: FTS5 (BM25 keyword), Dense (Nomic embeddings),
  ColBERT (answerai-colbert-small-v1 reranker)
- **Reciprocal Rank Fusion (RRF)** with per-source diversity capping
- **Temporal deduplication**: overlapping fragments from same source are merged,
  with merged IDs stashed for reader navigation
- **HyDE** (Hypothetical Document Embeddings) for deep preset
- **MMR** (Maximal Marginal Relevance) for synthesis preset — diversity-aware retrieval
- **Query reformulation**: keyword extraction for BM25, semantic rewrite for dense
- **Semantic cache**: exact-match query caching via daemon
- **Parent-child expression graph**: chunk children expand to parent context for synthesis
- **Session state**: `/set`, `/focus`, `/pin`, `/unpin` within a search session
- **Fragment Reader**: interactive `E` key workspace with prev/next segment context,
  bookmarking, note captures, `\related` cross-transcript search, transcript scroll (H/L)
- **Pinning**: `/pin <n>` persists fragments across re-searches to inject into next query

### Semantic Memory Graph
- **LanceDB** vector store with idempotent scalar index on `expression_id`
- **SQLite** expression graph with typed relations (`corrects`, `infers`, etc.)
- Expression types: TRANSCRIPT_SEGMENT, AUDIO_TRANSCRIPT, SYSTEM_INFERENCE,
  SESSION_MEMOIR, USER_CORRECTION, KNOWLEDGE_FRAGMENT
- **Memoir writer**: structured session summaries (narrative + key_insights + open_threads)
  with 3-tier compression (FULL / DIGEST / KEY_ONLY) for long context windows
- **Knowledge ingester**: manual fact injection into the graph
- **System inferences**: daemon-generated relational proposals surfaced in Observatory
- **Drift detection**: background daemon monitors for embedding alignment drift

### Background Daemon
- Single-instance lock file with startup auto-recovery
- UNIX domain socket IPC with fast-path timeouts
- Intelligence loop: continuous indexing, calibration, pipeline execution
- Operator registry and pipeline executor for autonomous background tasks
- Alignment worker for fault-tolerant background audio processing
- Embedding daemon with circuit breaker and retry backoff

### Playback
- Interactive MPV IPC controller with real-time transcript sync
- Karaoke/lyrics follow mode (terminal-native)
- MPV subtitle overlay mode (`--watch`)
- Hotkeys: seek ±5s/60s, speed ±10%, half/double
- Bookmarks: jump to bookmark on playback start (`--bookmark`)
- Focus gradient, center lock, speaker badges (toggleable in-session)
- Smart transcript selection by engine/model quality priority

### Bookmark & Annotation System
- 5 types: [HL] highlight, [TODO] todo, [NOTE] note, [BM] bookmark, [CUT] edit
- Point and region (start → end) markers
- AI auto-bookmarking (`audiobench bookmark auto`) with focus instruction
- Audacity label track export/import
- Interactive bookmark creation during playback and fragment reading
- Note captures (yanks) stored against segment IDs

### Chat & Ask
- Per-transcript QA with full context loading (`audiobench ask`)
- Multi-transcript chat REPL with loaded context (`audiobench chat`)
- Chapter picker for large audiobooks (auto-triggered at token threshold)
- Session memoir persistence with title generation
- Memoir-compressed context window for multi-session continuity
- `/memory`, `/daemon`, `/jobs`, `/bookmarks` slash commands in chat REPL

### Study Mode
- Project-based deep-dive sessions (`audiobench study new`)
- Chapter selection per study session
- Session tracking across multiple visits to same material

### Works System
- Semantic Works grouping (`audiobench work create/list/assign`)
- Associates multiple audio files under one work (e.g. all chapters of a book)

### Observatory
- Real-time telemetry TUI (`audiobench observatory`)
- Inferences and Proposals panels with confirm modal
- Live event streaming via async event bus
- Subsystem-level metric thresholds and alerts

### CLI Infrastructure
- Wizard prompts with readline inline editing and autocomplete
- Security tier system (`\unlock`, `require_tier`) for destructive commands
- Navigation stack (`\back`, `\home`, `\recent`)
- Phase tracker progress display for long operations
- PDF transcript export
- History with sorting, filtering, format output
- Database migration system (idempotent, m001-m025)
- Semantic cache, vector purge, orphan scan
- FastAPI REST server + React/Vite web UI (parallel access surface)

---

## Where We Are Falling Short

### 1. Fragment Display — The Most Visible Gap

**The problem:** The search result list truncates every fragment to 120 characters.
This is the most-used surface in the entire system and it treats the output like
a Google search result — a snippet to confirm relevance before clicking through.
But AudioBench is not a search engine. The fragment *is* the content.

**What exists:** `_display_results()` in `memory_cmd.py:508`. Flat list, 120-char
truncation, synthesis-failed renders a red error panel as if offline is broken.

**What it should be:**
- Source-grouped layout (Option Z): fragments grouped by source file in bordered
  boxes, ordered by relevance (max rrf_score of group), chronological within group
- Full paragraph text per fragment, wrapped to terminal width
- Compact/rich toggle: `/view compact` (2-line) vs `/view rich` (full paragraph)
- Offline = rich mode automatic (fragments are the reading when no LLM)
- Online = compact default with synthesis panel below
- No red error panel for offline — absence of synthesis panel *is* the signal

**Evolution path:**
- Phase 1 (Now): Option Z source-grouped layout, full text, no synthesis error panel
- Phase 2: `/view compact` toggle, offline auto-rich
- Phase 3: Parent context block inside each fragment card (Option C) — the parent
  chunk is already in the database, just not surfaced at display time

### 2. Offline Mode — Wrong Mental Model in the Code

**The problem:** `synthesis_failed` is treated as an error. The word "instead"
in `"Showing raw fragments instead"` reveals the assumption: synthesis is primary,
fragments are fallback. This is backwards.

**What should change:**
- Rename the mental model: offline mode = "fragment-first mode"
- `ResearchEngine.search()` returns `synthesis_failed=True` when no LLM — this
  is a *state*, not an *error*. Display layer should read it as a display mode switch
- The synthesis panel simply does not render — no banner, no warning, no apology
- Fragment display automatically switches to rich mode in this state

### 3. Corpus Retrieval — Fixed Top-K, No Paragraph Expansion

**The problem:** `top_k=10` is hard-coded in the interactive search wizard.
Segment chunks are typically 3-8 sentences. No way for the user to expand a
fragment to see more surrounding text *before* entering the fragment reader.

**What should change:**
- Expose top_k as `/set topk <n>` in search session state
- Add quick context expansion: pressing `X` on a fragment in the list shows
  +/-1 segment of context inline without entering the full reader
- Consider increasing default chunk size: `chunk_max_tokens=350` produces short
  chunks. 500-600 would give more self-contained fragments at search time.

### 4. Speaker / Diarization — Named but Anonymous

**The problem:** Diarization works and speaker tags are stored. `--auto-name`
uses Gemini to detect real names. But in retrieval, speakers appear as raw
strings (e.g. "SPEAKER_00"), and there is no way to filter or browse by speaker
identity across the library without knowing the exact label used in that file.

**What should change:**
- Speaker registry: a unified table mapping per-file speaker labels to canonical
  identities (e.g. "SPEAKER_00 in krishnamurti_009" -> "Krishnamurti")
- Speaker browse: `audiobench speakers list`, `audiobench speakers rename`
- `/focus speaker:<name>` in search to retrieve only from a named speaker
- Anonymous voice tagging (Idea #4 from session): tag all voices without naming
  them initially; allow user to play a subset and rename on first listen
  -> Implementation: speaker registry with `is_named: bool`, unnamed speakers
  get a deterministic anonymous label (e.g. "Voice A", "Voice B") scoped per file

### 5. Transcript Tagging & Late-Night Context Injection (Idea #2)

**The problem:** High-value recordings (e.g. late-night idea sessions tagged
`@uon`) have no way to be automatically injected as context into searches or
chat sessions. The tag system at the file level is not connected to the retrieval
pipeline.

**What should change:**
- Tag groups: named collections (e.g. `@uon`, `@morning`, `@ideas`)
- `audiobench tag` command group: `add`, `rm`, `list`, `group`
- `/context @uon` in search REPL: pins all fragments from that tag group as
  background context for every query in the session
- In chat REPL: `/load @uon` injects all tagged transcripts as context
- Storage: `audio_file_tags` table with `file_id`, `tag`, `created_at`

### 6. Idea Drilling / Root Cause Prompting (Idea #1)

**The problem:** No structured tool for arriving at the root of an idea.
The chat REPL is a general assistant. What is missing is a mode that gently
guides you *toward* the origin of a thought — not a quiz, not a drill, but
a companion that asks the right next question.

**What should be built:**
- `audiobench idea` command: starts a structured Socratic session
- Prompt sequence (adapts to answers, not rigid):
  1. What is the idea? (free description)
  2. Where did it come from? (provenance — a recording? a person? a feeling?)
  3. What makes you believe it? (evidence)
  4. What makes you doubt it? (counter)
  5. What would it mean if it were true? (implication)
  6. What is the single thing at the root of why this matters to you? (why)
- After each answer, searches the memory graph for related fragments
  and surfaces them as supporting or contrasting material
- Session saved as a `SYSTEM_INFERENCE` expression of type "idea_root"
- Not AI-driven questioning — the questions are fixed but the order and
  phrasing adapt based on what the user has already said

### 7. Gemini Diarization — Not Unified (Idea #5)

**The problem:** Gemini transcription has built-in speaker detection but it is
a separate code path from Pyannote diarization. The speaker labels from Gemini
do not feed into the same speaker registry as Pyannote labels. Results from
both engines are stored but not reconciled.

**What should change:**
- Unified diarization output protocol: regardless of engine, speaker labels
  should be normalised to `SPEAKER_NN` format and written to the same
  `segments.speaker` field
- `diarization/engine.py` should be the single normalisation point
- Gemini speaker detection should go through the same speaker registry
  as Pyannote, enabling cross-engine speaker identity matching

### 8. Graph Connections — Not Visible, Not Browsable

**The problem:** The expression graph exists (expressions + relations tables,
parent-child links, inference edges). But there is no way for the user to *see*
what connections have been made. The graph is used at retrieval time but never
surfaced directly.

**What should change:**
- `audiobench graph` command: show expressions, their relations, and clusters
- In the fragment reader: show which other fragments are graph-connected to
  the current one (not just `\related` semantic search — actual graph edges)
- The Observatory inferences panel already shows proposals — extend it to show
  the full expression graph state, not just pending proposals

### 9. Report Command — Shallow

**The problem:** `audiobench report <id>` dumps a knowledge dossier but it is
a raw dump: transcript text, ask log, chat sessions. It does not synthesise.
It does not show the expression graph. It does not show cross-file connections.

**What should change:**
- Report should produce a structured markdown document with:
  - Source metadata
  - Key bookmarks (highlights only)
  - Memoir narratives from all chat sessions on this file
  - Top system inferences related to this file
  - Graph connections to other files in the library
  - Cross-source fragment clusters (which other recordings discuss the same ideas)

### 10. Study Mode — Under-Developed

**The problem:** `audiobench study` creates a project and session, but the
study session itself is just chat with chapter selection. There is no
progression tracking, no spaced repetition, no recall testing.

**What should change:**
- Study session has a `status` (not_started / in_progress / reviewed / mastered)
- After each session, memoir is auto-generated and stored as study session summary
- "Recall mode": given a fragment, hide it and ask user to reconstruct
- "Connection mode": show two fragments from different sessions, ask user to
  articulate the connection (then AI validates / enriches)

---

## Near-Term Work (Current Sprint)

### P0 — Fragment Display: Option Z Source-Grouped Layout
> **File:** `src/audiobench/cli/commands/memory_cmd.py`

Rewrite `_display_results()` to:
1. Group `result.sources` (list of `FusedResult`) by `source_file`
2. Order groups by max `rrf_score` within group (descending)
3. Order fragments within each group by `start_time` (chronological)
4. Render each group in a bordered section with source name as header
5. Display full `fr.text` per fragment (no truncation), wrapped to terminal width
6. Remove the red synthesis-failed panel — no synthesis panel = offline, no apology
7. Auto-switch to rich (full paragraph) display when `synthesis_failed=True`
8. Keep synthesis answer panel when synthesis succeeded (below the grouped fragments)

Design decisions locked:
- Group ordering: by relevance (max rrf_score)
- Fragment ordering within group: by start_time (chronological)
- Single-source results: still render the source box (consistency)
- Full text, no truncation cap
- Offline = no synthesis panel, not an error

Evolution markers left in code for future phases:
```python
# ROADMAP [Phase 2]: Add `/view compact` toggle — `view_mode: str = "rich"` in
# SearchSessionState. Compact = 2-line snippet per fragment inside group box.
# Auto-rich when synthesis_failed=True; auto-compact when synthesis succeeded.

# ROADMAP [Phase 3]: Surface parent chunk context per fragment (Option C).
# Parent is already fetched in ResearchEngine — pass it through FusedResult
# or fetch at display time via ExpressionRepository.get_parents_batch().
```

### P1 — Offline Mode: Fragment-First Mental Model
> **File:** `src/audiobench/cli/commands/memory_cmd.py`

Remove `synthesis_failed` red error panel. Offline is not broken.
Connected to P0 — done as part of the same display rewrite.

### P2 — `/view` Toggle in Search Session State
> **Files:** `memory_cmd.py`, `SearchSessionState`

Add `view_mode: str = "rich"` to `SearchSessionState`.
`/view rich` and `/view compact` commands in `_parse_slash_command()`.
Default: rich when offline, compact when online.

### P3 — Increase Fragment Retrieval Corpus
> **Files:** `memory_cmd.py`, `memory/query_engine.py`

Expose top_k as `/set topk <n>` in search session. Default stays 10.
Consider bumping default chunk size from 350 to 500 tokens in settings.

---

## Medium-Term Work (Next 2-3 Months)

### M1 — Speaker Registry & Anonymous Voice Tagging
Unified speaker identity system across engines and files.
Anonymous voice tagging with deferred naming — tag voices on transcription,
name them on first listen. Speaker registry with `is_named: bool`.

### M2 — Tag Groups & Context Injection
`audiobench tag` command group: `add`, `rm`, `list`, `group`.
`/context @<tag>` in search REPL pins tagged file fragments as session background.
`/load @<tag>` in chat REPL loads all tagged transcripts as context.
New `audio_file_tags` table in schema.

### M3 — Idea Drilling Mode
`audiobench idea` — Socratic session with memory graph lookups after each answer.
Sessions saved as `idea_root` SYSTEM_INFERENCE expressions.
Questions are fixed but phrasing adapts to what the user has already said.

### M4 — Gemini Diarization Unification
Single normalisation point in `diarization/engine.py`.
All speaker labels -> `SPEAKER_NN` format regardless of engine.
Cross-engine speaker registry reconciliation.

### M5 — Graph Visibility
`audiobench graph` command for browsing expression graph state.
Graph edges visible in fragment reader (actual DB relations, not just semantic search).
Observatory graph panel extension.

---

## Long-Term Vision (6-12 Months)

### L1 — Cross-Recording Idea Clusters
Cluster fragments from across the entire library by underlying idea.
Cluster membership surfaces in search, chat, and report.
Background daemon identifies cluster drift when new recordings are added.

### L2 — Study Mode: Recall & Connection Testing
Spaced repetition over your own recorded thoughts.
Recall mode: given a fragment, can you articulate what comes next?
Connection mode: given two fragments from different recordings, articulate the link.
AI validates and enriches. The cognition is yours.

### L3 — Timeline View
A chronological view of when ideas entered the library and how they evolved.
Fragment timestamps map to recording dates.
"You first discussed X on 2026-03-14. You returned to it on 2026-07-22 with a different angle."

### L4 — Voice-Biometric Library
Every speaker in the library gets a voiceprint (via `_biometric_worker.py`,
which already exists but is not fully plumbed in).
Identify when the same person appears in different recordings.
Surface "same speaker, different context" connections automatically.

### L5 — Web UI Parity
The FastAPI + React web UI exists but is a parallel access surface.
Long-term: bring the fragment reader, search session state, and memoir timeline
into the web UI so it can run on a home server and be accessed from any device.

---

## Architecture Principles for Future Contributors

1. **Fragments are the product. Synthesis is the lens.**
   The LLM synthesises from fragments — it does not replace them.
   Never hide fragments behind synthesis. Always show the source material.

2. **Offline is not degraded.**
   When no LLM is available, the retrieval system still functions fully.
   Design every display surface to work without synthesis.
   Absence of a synthesis panel is information, not an error.

3. **The retrieval pipeline is layered, not monolithic.**
   Three streams (FTS5, Dense, ColBERT) -> RRF fusion -> temporal dedup ->
   parent expansion -> synthesis. Each layer is independently testable.
   Never collapse them.

4. **Chunking decisions are upstream of display.**
   The chunker (`memory/chunking.py`) decides chunk boundaries.
   The display layer should not second-guess those decisions by truncating.
   Show the full chunk. Trust the chunker.

5. **The graph is the long-term memory.**
   Individual fragments are ephemeral search results.
   The expression graph (expressions + relations) is the persistent knowledge
   structure. Every important pattern should eventually become a graph edge.

6. **No bare `except: pass`.**
   Every exception path logs a warning and returns a structured result.
   Use Ok/Err result types for LLM calls (see `query_engine.py` pattern).

7. **Batch everything in the retrieval path.**
   No per-item `get_by_id()` calls inside loops.
   Use `get_by_ids()` and `get_parents_batch()` for all graph lookups.
   See engineering standards EQ-1 through EQ-4 in `query_engine.py`.

---

*Last updated: 2026-08-03*
*Maintained by: the person who built this and the AI that helped them think it through.*
