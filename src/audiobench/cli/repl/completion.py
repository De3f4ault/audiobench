"""REPL tab completion — prompt_toolkit-based context-aware completion.

Three layers work together:
  1. Tab completion  (AudioBenchCompleter)   — full context-aware popup
  2. Ghost suggestion (AudioBenchAutoSuggest) — history-first, then template, then sequential
  3. Chaining        (see __init__.py `;`)   — handled at the dispatch level

Template system:
  Every command has a synopsis string like `transcribe <files> [-m MODEL] [-f FORMAT]`.
  Required arguments are shown as `<name>`, optional as `[name]`.
  These are surfaced as ghost text when you type a full command name + space.
"""

from __future__ import annotations

import glob

import click
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.completion import Completer, Completion

from audiobench.cli.repl.session import ReplSession
from audiobench.core.db_session import get_session
from audiobench.storage.models import NoteCollection

# ── Caches (populated at session start via setup_completion) ──────────────────

_FLAG_MAP: dict[str, list[tuple]] = {}       # "cmd" or "group sub" -> [(opts, val_type, help, choices)]
_SUBCOMMAND_MAP: dict[str, list[str]] = {}   # "group" -> ["sub1", "sub2", ...]
_TEMPLATE_MAP: dict[str, str] = {}           # "cmd" -> "cmd <arg> [--flag VALUE]"
_TRANSCRIPT_CACHE: list[dict] = []
_TRANSITION_MATRIX: dict[str, str] = {}      # "cmd" -> most-frequent next command


# ── Template builder ──────────────────────────────────────────────────────────

# Options worth showing in ghost templates (high-value, commonly used)
_TEMPLATE_OPTS = {
    "transcribe": ["--model", "--format", "--language", "--diarize", "--background"],
    "ask":        ["--interactive", "--chapter"],
    "show":       ["--format", "--timestamps"],
    "export":     ["--format", "--output"],
    "summarize":  ["--model", "--interactive"],
    "chat":       ["--model", "--resume"],
    "history":    ["--limit", "--format"],
    "search":     ["--limit", "--regex"],
    "listen":     ["--language", "--save", "--model"],
    "convert":    ["--output", "--bitrate", "--speed"],
    "play":       ["--from", "--speed", "--bookmarks"],
    "clean":      ["--all", "--model", "--dry-run"],
    "memory search": ["--interactive", "--preset"],
    "bookmark add":  ["--type", "--notes"],
    "bookmark list": ["--type", "--format"],
}


def _build_synopsis(cmd_name: str, cmd_obj: click.BaseCommand) -> str:
    """Build a compact synopsis: `transcribe <files> [-m MODEL] [--diarize]`"""
    if isinstance(cmd_obj, click.Group):
        subs = list(cmd_obj.commands.keys())
        shown = subs[:3]
        ellipsis = "…" if len(subs) > 3 else ""
        return f"{cmd_name} <{'|'.join(shown)}{ellipsis}>"

    parts = [cmd_name]

    # Arguments first
    for param in cmd_obj.params:
        if isinstance(param, click.Argument):
            name = param.human_readable_name.lower()
            if param.required:
                parts.append(f"<{name}>")
            else:
                parts.append(f"[{name}]")

    # Selected options
    wanted = set(_TEMPLATE_OPTS.get(cmd_name, []))
    if wanted:
        for param in cmd_obj.params:
            if isinstance(param, click.Option):
                flag = param.opts[-1]  # longest form
                if flag in wanted:
                    if isinstance(param.type, click.Choice):
                        choices_str = "|".join(param.type.choices)
                        parts.append(f"[{flag} {choices_str}]")
                    elif param.is_flag:
                        parts.append(f"[{flag}]")
                    else:
                        meta = param.type.name.upper() if param.type.name != "text" else "VALUE"
                        parts.append(f"[{flag} {meta}]")

    return " ".join(parts)


def _index_command(cmd_name: str, cmd_obj: click.BaseCommand) -> None:
    """Recursively index a command or group into FLAG_MAP, SUBCOMMAND_MAP, TEMPLATE_MAP."""
    _TEMPLATE_MAP[cmd_name] = _build_synopsis(cmd_name, cmd_obj)

    if isinstance(cmd_obj, click.Group):
        _SUBCOMMAND_MAP[cmd_name] = list(cmd_obj.commands.keys())
        _FLAG_MAP[cmd_name] = []
        for sub_name, sub_obj in cmd_obj.commands.items():
            full_key = f"{cmd_name} {sub_name}"
            _TEMPLATE_MAP[full_key] = _build_synopsis(full_key, sub_obj)
            flags = []
            for param in sub_obj.params:
                if isinstance(param, click.Option):
                    choices = param.type.choices if isinstance(param.type, click.Choice) else None
                    flags.append((param.opts, param.type, param.help or "", choices))
            _FLAG_MAP[full_key] = flags
    else:
        flags = []
        for param in cmd_obj.params:
            if isinstance(param, click.Option):
                choices = param.type.choices if isinstance(param.type, click.Choice) else None
                flags.append((param.opts, param.type, param.help or "", choices))
        _FLAG_MAP[cmd_name] = flags


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_completion(session: ReplSession) -> None:
    """Build all completion caches from live Click command tree."""
    global _FLAG_MAP, _SUBCOMMAND_MAP, _TEMPLATE_MAP
    _FLAG_MAP.clear()
    _SUBCOMMAND_MAP.clear()
    _TEMPLATE_MAP.clear()

    for cmd_name, cmd_obj in session.cli_group.commands.items():
        _index_command(cmd_name, cmd_obj)

    _load_transcript_cache(session)
    _load_transition_matrix()


def _load_transcript_cache(session: ReplSession) -> None:
    global _TRANSCRIPT_CACHE
    try:
        repo = session._get_repo()
        records = repo.get_history(limit=50)
        _TRANSCRIPT_CACHE = [{"id": r["id"], "file_name": r.get("file_name", "?")} for r in records]
    except Exception:
        _TRANSCRIPT_CACHE = []


def _load_transition_matrix() -> None:
    global _TRANSITION_MATRIX
    try:
        from audiobench.observatory.db import get_journal_session
        with get_journal_session() as conn:
            result = conn.execute("""
                SELECT
                  json_extract(metadata, '$.command') as command,
                  LEAD(json_extract(metadata, '$.command'))
                    OVER (PARTITION BY session_id ORDER BY ts) as next_command,
                  COUNT(*) as freq
                FROM system_events
                WHERE subsystem='repl'
                  AND event_type='command_dispatched'
                  AND ts > datetime('now', '-90 days')
                GROUP BY command, next_command
                ORDER BY freq DESC
            """).fetchall()
            matrix: dict[str, str] = {}
            for row in result:
                cmd = row[0]
                next_cmd = row[1]
                if cmd and next_cmd and cmd not in matrix:
                    matrix[cmd] = next_cmd
            _TRANSITION_MATRIX = matrix
    except Exception:
        _TRANSITION_MATRIX = {}


# ── Completer ─────────────────────────────────────────────────────────────────

# Commands that take transcript IDs as first positional arg
_TX_ID_CMDS = {"show", "ask", "export", "summarize", "vocab", "delete", "report", "\\focus"}

# Commands that take audio files as first positional arg
_AUDIO_FILE_CMDS = {"transcribe", "convert", "play", "inspect", "analyze", "subtitle", "merge", "work"}

# Audio extensions to complete
from audiobench.transcribe.audio_converter import ALL_SUPPORTED_FORMATS

_AUDIO_EXTS = tuple(f".{ext}" for ext in ALL_SUPPORTED_FORMATS)


class AudioBenchCompleter(Completer):
    """Context-aware multi-token completer.

    Completion hierarchy for each token position:
      pos 0  : command names (all namespaces)
      pos 1  : subcommand name (for groups) OR first argument/flag
      pos 2+ : flags, argument values, choices
    """

    def __init__(self, session: ReplSession):
        self.session = session
        from audiobench.cli.repl.dispatch import _BACKSLASH_HANDLERS
        self.backslash_cmds = [f"\\{k}" for k in _BACKSLASH_HANDLERS.keys()]
        self.slash_cmds = ["/help", "/commands", "/clear", "/exit", "/context"]
        from audiobench.cli.repl.dot_commands import _DOT_HANDLERS
        self.dot_cmds = [f".{k}" for k in _DOT_HANDLERS.keys()]
        self.meta_words = ["help", "exit", "quit", "clear", "commands"]

    def get_completions(self, document, complete_event):
        text_before = document.text_before_cursor
        tokens = text_before.split()
        word = document.get_word_before_cursor(WORD=True)

        # ── Position 0: command name ──────────────────────────────────────────
        if not tokens or (len(tokens) == 1 and not text_before.endswith(" ")):
            yield from self._complete_command(word)
            return

        cmd = tokens[0]

        # ── Group: position 1 = subcommand name ──────────────────────────────
        if cmd in _SUBCOMMAND_MAP:
            if len(tokens) == 1 and text_before.endswith(" "):
                # Offer all subcommand names with their templates
                for sub in _SUBCOMMAND_MAP[cmd]:
                    full_key = f"{cmd} {sub}"
                    meta = _TEMPLATE_MAP.get(full_key, "")
                    yield Completion(sub, start_position=0, display_meta=meta)
                return
            elif len(tokens) == 2 and not text_before.endswith(" "):
                partial = tokens[1]
                for sub in _SUBCOMMAND_MAP[cmd]:
                    if sub.startswith(partial):
                        full_key = f"{cmd} {sub}"
                        meta = _TEMPLATE_MAP.get(full_key, "")
                        yield Completion(sub, start_position=-len(partial), display_meta=meta)
                return
            elif len(tokens) >= 2:
                # Subcommand already typed — complete its args/flags
                sub = tokens[1]
                full_key = f"{cmd} {sub}"
                remaining_tokens = tokens[2:]
                if text_before.endswith(" "):
                    yield from self._complete_args(full_key, remaining_tokens, tokens[-1])
                else:
                    yield from self._complete_partial(full_key, remaining_tokens, word)
                return

        # ── Flat command: position 1+ = args/flags ───────────────────────────
        if text_before.endswith(" "):
            last = tokens[-1]
            yield from self._complete_args(cmd, tokens[1:], last)
        else:
            yield from self._complete_partial(cmd, tokens[1:], word)

    def _complete_command(self, word: str):
        all_cmds = (
            list(self.session.cli_group.commands.keys()) +
            self.backslash_cmds + self.slash_cmds + self.dot_cmds + self.meta_words
        )
        for cmd in all_cmds:
            if cmd.startswith(word):
                # Prefer Click short_help; fall back to template
                if cmd in self.session.cli_group.commands:
                    meta = self.session.cli_group.commands[cmd].short_help or _TEMPLATE_MAP.get(cmd, "")
                else:
                    meta = _TEMPLATE_MAP.get(cmd, "")
                yield Completion(cmd, start_position=-len(word), display_meta=meta)

    def _complete_args(self, cmd: str, used_tokens: list[str], last_token: str):
        """Complete after a space — offer flags, choices, IDs, files."""

        # ── If previous token was a flag that expects a value, complete the value ──
        if last_token.startswith("-") and cmd in _FLAG_MAP:
            for opts, val_type, help_text, choices in _FLAG_MAP[cmd]:
                if last_token in opts:
                    if choices:
                        for choice in choices:
                            yield Completion(choice, start_position=0, display_meta=help_text)
                        return
                    elif isinstance(val_type, click.Path):
                        files = glob.glob("*")
                        for f in files:
                            yield Completion(f, start_position=0)
                        return
                    # No choices to offer — just return
                    return

        # ── Flags not yet used ────────────────────────────────────────────────
        if cmd in _FLAG_MAP:
            for opts, val_type, help_text, choices in _FLAG_MAP[cmd]:
                main_opt = opts[-1] if opts else ""
                if main_opt and main_opt not in used_tokens:
                    if choices:
                        meta = f"{help_text} [{','.join(choices)}]" if help_text else f"[{','.join(choices)}]"
                    else:
                        meta = help_text
                    yield Completion(main_opt, start_position=0, display_meta=meta)

        # ── Transcript IDs ────────────────────────────────────────────────────
        if cmd in _TX_ID_CMDS:
            for t in _TRANSCRIPT_CACHE:
                yield Completion(str(t["id"]), start_position=0, display_meta=t["file_name"])

        # ── Audio files ───────────────────────────────────────────────────────
        if cmd in _AUDIO_FILE_CMDS:
            for ext in _AUDIO_EXTS:
                for f in glob.glob(f"*{ext}"):
                    yield Completion(f, start_position=0)

        # ── Notes ─────────────────────────────────────────────────────────────
        if cmd == "\\note":
            try:
                with get_session() as db:
                    notes = (
                        db.query(NoteCollection)
                        .order_by(NoteCollection.updated_at.desc())
                        .limit(20)
                        .all()
                    )
                    for n in notes:
                        yield Completion(str(n.id), start_position=0, display_meta=n.title)
            except Exception:
                pass

    def _complete_partial(self, cmd: str, used_tokens: list[str], word: str):
        """Complete a partially-typed token."""

        # ── Partial flag ─────────────────────────────────────────────────────
        if word.startswith("-") and cmd in _FLAG_MAP:
            for opts, val_type, help_text, choices in _FLAG_MAP[cmd]:
                for opt in opts:
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word), display_meta=help_text)
            return

        # ── Partial transcript ID ─────────────────────────────────────────────
        if cmd in _TX_ID_CMDS:
            for t in _TRANSCRIPT_CACHE:
                if str(t["id"]).startswith(word):
                    yield Completion(
                        str(t["id"]), start_position=-len(word), display_meta=t["file_name"]
                    )

        # ── Partial audio filename ────────────────────────────────────────────
        if cmd in _AUDIO_FILE_CMDS:
            for f in glob.glob(word + "*"):
                if f.endswith(_AUDIO_EXTS):
                    yield Completion(f, start_position=-len(word))


# ── Ghost Suggestion ──────────────────────────────────────────────────────────

class AudioBenchAutoSuggest(AutoSuggest):
    """Three-tier ghost suggestion:
      1. History match (AutoSuggestFromHistory) — always first
      2. Command template — when cmd is complete and cursor is at a space
      3. Sequential prediction — when starting a new command, predict from transition matrix
    """

    def __init__(self):
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        self._history = AutoSuggestFromHistory()

    def get_suggestion(self, buffer, document):
        text = document.text

        # ── Tier 1: history ───────────────────────────────────────────────────
        suggestion = self._history.get_suggestion(buffer, document)
        if suggestion:
            return suggestion

        if not text.strip():
            return None

        tokens = text.split()

        # ── Tier 2: template ghost ────────────────────────────────────────────
        # When user just typed a complete command name + space, ghost the synopsis.
        # e.g. typed "transcribe " → ghost "<files> [-m MODEL] [-f FORMAT]"
        if text.endswith(" ") and len(tokens) == 1:
            cmd = tokens[0]
            template = _TEMPLATE_MAP.get(cmd, "")
            if template:
                # The synopsis includes the command name; strip it, ghost the rest
                rest = template[len(cmd):].strip()
                if rest:
                    return Suggestion(rest)

        # For group commands: "bookmark " → ghost "add|list|rm…"
        if text.endswith(" ") and len(tokens) == 1 and tokens[0] in _SUBCOMMAND_MAP:
            subs = _SUBCOMMAND_MAP[tokens[0]]
            if subs:
                return Suggestion(subs[0])  # ghost the first subcommand

        # ── Tier 3: sequential prediction ────────────────────────────────────
        # When typing the start of a new command, check if the previous command
        # predicts this one.
        if len(tokens) == 1 and not text.endswith(" "):
            history_strings = buffer.history.get_strings()
            if history_strings:
                last_cmd = history_strings[-1].split()[0]
                predicted = _TRANSITION_MATRIX.get(last_cmd)
                if predicted and predicted.startswith(text):
                    return Suggestion(predicted[len(text):])

        return None
