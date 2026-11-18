.PHONY: install dev test lint format clean help
.PHONY: transcribe transcribe-srt history search info download download-all download-memory
.PHONY: translate subtitle subtitle-hard listen speak download-voice summarize ask diarize
.PHONY: repl chat vocab doctor status cleanup preset preset-create
.PHONY: daemon-start daemon-status daemon-stop memory-search memory-ask db-status jobs-list

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CLI := $(VENV)/bin/audiobench

# ── Help ─────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────

install: ## Install base dependencies
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .

dev: ## Install with dev dependencies (editable)
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[all]"

# ── Transcription ────────────────────────────────────────────

transcribe: ## Transcribe a file (make transcribe FILE=audio.m4a)
	$(CLI) transcribe $(FILE)

transcribe-srt: ## Transcribe → SRT (make transcribe-srt FILE=audio.m4a)
	$(CLI) transcribe -f srt $(FILE)

history: ## View transcription history
	$(CLI) history

search: ## Search transcriptions (make search Q="keyword")
	$(CLI) search "$(Q)"

info: ## Show system info and settings
	$(CLI) info

download: ## Download a model (make download MODEL=large-v3-turbo)
	$(CLI) download $(or $(MODEL),large-v3-turbo)

download-all: ## Download all offline models (Whisper + semantic memory daemon)
	$(CLI) download --all

download-memory: ## Download memory & daemon retrieval models
	$(CLI) download --memory

# ── Feature Commands ─────────────────────────────────────────

translate: ## Translate audio to English (make translate FILE=audio.m4a)
	$(CLI) transcribe --translate $(FILE)

subtitle: ## Add subtitles to video (make subtitle FILE=video.mp4)
	$(CLI) subtitle $(FILE)

subtitle-hard: ## Burn subtitles into video (make subtitle-hard FILE=video.mp4)
	$(CLI) subtitle --hard $(FILE)

listen: ## Live microphone transcription
	$(CLI) listen

speak: ## Speak text aloud (make speak TEXT="Hello world")
	$(CLI) speak "$(TEXT)"

download-voice: ## Download a TTS voice (make download-voice VOICE=en_US-amy-medium)
	$(CLI) download-voice $(or $(VOICE),en_US-amy-medium)

summarize: ## AI-summarize a transcript (make summarize ID=3)
	$(CLI) summarize $(ID)

ask: ## Ask AI about a transcript (make ask ID=3 Q="What decisions?")
	$(CLI) ask $(ID) "$(Q)"

diarize: ## Transcribe with speaker identification (make diarize FILE=meeting.m4a)
	$(CLI) transcribe --diarize $(FILE)

repl: ## Launch interactive shell
	$(CLI) repl

chat: ## Start AI chat (make chat ID=3)
	$(CLI) chat $(ID)

vocab: ## Word frequency analysis (make vocab ID=3)
	$(CLI) vocab $(ID)

preset: ## List presets
	$(CLI) preset list

preset-create: ## Create a preset (make preset-create NAME=meeting)
	$(CLI) preset create $(NAME)

# ── Daemon & Memory ──────────────────────────────────────────

daemon-start: ## Start background daemon
	$(CLI) daemon start

daemon-status: ## Show daemon status & health
	$(CLI) daemon status

daemon-stop: ## Stop background daemon
	$(CLI) daemon stop

memory-search: ## Vector search over memory (make memory-search Q="query")
	$(CLI) memory search "$(Q)"

memory-ask: ## RAG query over memory (make memory-ask Q="query")
	$(CLI) memory ask "$(Q)"

db-status: ## Show database status & migration state
	$(CLI) db status

jobs-list: ## List background processing jobs
	$(CLI) jobs list

# ── System ───────────────────────────────────────────────────

doctor: ## Check system health
	$(CLI) doctor

status: ## Show usage statistics
	$(CLI) status

cleanup: ## Clean old data (make cleanup ARGS="--older-than 30d")
	$(CLI) cleanup $(ARGS)

# ── Development ──────────────────────────────────────────────

test: ## Run test suite with coverage
	$(PYTHON) -m pytest tests/ -v --cov=src/audiobench --cov-report=term-missing

lint: ## Run linters (ruff + mypy)
	$(PYTHON) -m ruff check src/
	$(PYTHON) -m mypy src/

format: ## Auto-format code (black + ruff fix)
	$(PYTHON) -m black src/
	$(PYTHON) -m ruff check --fix src/

clean: ## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
