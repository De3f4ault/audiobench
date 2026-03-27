# Changelog

All notable changes to AudioBench will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-03-27

### Added

- **Bookmark & Annotation System** — full-featured timestamp marking for audio files
  - Point bookmarks and region markers (Audacity-inspired dual marker model)
  - 5 bookmark types: 🔖 bookmark, ⭐ highlight, 📌 todo, 📝 note, ✂️ edit
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
