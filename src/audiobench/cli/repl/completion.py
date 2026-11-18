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


# ── Backslash Metadata & Subcommand Definitions ──────────────────────────────

_BACKSLASH_METAS: dict[str, tuple[str, str]] = {
    "\\synthesize": ("\\synthesize <thematic query>", "Dialectic synthesis across library via Graph MMR"),
    "\\synth":      ("\\synth <thematic query>", "Alias for \\synthesize"),
    "\\bookmark":   ("\\bookmark [id|cite <n>|list]", "Two-tiered smart contextual bookmark cards"),
    "\\bm":         ("\\bm [id|cite <n>|list]", "Alias for \\bookmark"),
    "\\related":    ("\\related", "Explore cross-work semantic bridges & HNSW connections"),
    "\\suggest":    ("\\suggest [commands|history]", "Next Listen recommendations & triadic consensus"),
    "\\graph":      ("\\graph [seed [limit]|days]", "Seed unlinked islands or show command stats"),
    "\\play":       ("\\play [file|id|cite <n>|pause|resume]", "Audio playback controls & citation seeking"),
    "\\library":    ("\\library", "Interactive Library Command Center"),
    "\\workflow":   ("\\workflow <save|list|run|delete|show>", "Manage and replay command workflows"),
    "\\focus":      ("\\focus <id>", "Focus on audio file, transcript, or work"),
    "\\unfocus":    ("\\unfocus", "Clear active focus"),
    "\\show":       ("\\show [id]", "Print transcript with timestamps"),
    "\\ls":         ("\\ls [n]", "List recent audio files"),
    "\\search":     ("\\search <query>", "FTS5 full-text transcript search"),
    "\\chat":       ("\\chat", "Conversational AI chat over active transcript"),
    "\\ask":        ("\\ask <question>", "Ask AI about active transcript"),
    "\\summarize":  ("\\summarize", "Structured chapter summary"),
    "\\capture":    ("\\capture <note>", "Quick-capture thought or note"),
    "\\info":       ("\\info", "Metadata and stats for active focus"),
    "\\help":       ("\\help [command]", "Show command documentation & syntax"),
    "\\name":       ("\\name <spk> <name>", "Rename speaker globally across transcript"),
    "\\schools":    ("\\schools", "List all emergent intellectual schools of thought"),
    "\\school":     ("\\school <id>", "Inspect school medoid, authors, and core works"),
    "\\scope":      ("\\scope [school:<id>|work:<id>|clear]", "Scope REPL retrieval to a school or work"),
    "\\next":       ("\\next", "Advance to next track in school queue or citation"),
    "\\prev":       ("\\prev", "Return to previous track in school queue or citation"),
    "\\lyrics":     ("\\lyrics", "Interactive live lyrics / karaoke follow HUD"),
    "\\karaoke":    ("\\karaoke", "Alias for \\lyrics"),
    "\\follow":     ("\\follow", "Alias for \\lyrics"),
    "\\lock":       ("\\lock", "Lock privacy tiers"),
    "\\unlock":     ("\\unlock", "Unlock relational privacy tier"),
    "\\proposals":  ("\\proposals [show <id>|all|authorize <id>|defer <id>|reject <id>]", "Review and authorize knowledge gap proposals"),
    "\\prop":       ("\\prop [show <id>|all|authorize <id>|defer <id>|reject <id>]", "Alias for \\proposals"),
    "\\context":    ("\\context", "Display active entity focus and navigation context"),
}

_BACKSLASH_SUBCOMMANDS: dict[str, list[str]] = {
    "\\bookmark": ["cite", "list"],
    "\\bm": ["cite", "list"],
    "\\graph": ["seed"],
    "\\suggest": ["commands", "history"],
    "\\workflow": ["save", "list", "run", "delete", "show"],
    "\\play": ["cite", "school", "pause", "resume", "stop", "next", "prev"],
    "\\scope": ["school:", "work:", "clear"],
    "\\related": ["schools", "works"],
    "\\proposals": ["show", "all", "list", "authorize", "defer", "reject"],
    "\\prop": ["show", "all", "list", "authorize", "defer", "reject"],
}


_BACKSLASH_SUBCOMMAND_TEMPLATES: dict[str, str] = {
    "\\bookmark cite": "\\bookmark cite <n>",
    "\\bookmark list": "\\bookmark list",
    "\\bm cite": "\\bm cite <n>",
    "\\bm list": "\\bm list",
    "\\graph seed": "\\graph seed [limit]",
    "\\suggest commands": "\\suggest commands",
    "\\suggest history": "\\suggest history",
    "\\workflow save": "\\workflow save <name>",
    "\\workflow list": "\\workflow list",
    "\\workflow run": "\\workflow run <name>",
    "\\workflow delete": "\\workflow delete <name>",
    "\\workflow show": "\\workflow show <name>",
    "\\play cite": "\\play cite <n>",
    "\\play school": "\\play school <id>",
    "\\play pause": "\\play pause",
    "\\play resume": "\\play resume",
    "\\play stop": "\\play stop",
    "\\play next": "\\play next",
    "\\play prev": "\\play prev",
    "\\scope school": "\\scope school:<id>",
    "\\scope work": "\\scope work:<id>",
    "\\scope clear": "\\scope clear",
    "\\proposals show": "\\proposals show <id>",
    "\\proposals all": "\\proposals all",
    "\\proposals list": "\\proposals list",
    "\\proposals authorize": "\\proposals authorize <id>",
    "\\proposals defer": "\\proposals defer <id>",
    "\\proposals reject": "\\proposals reject <id>",
    "\\prop show": "\\prop show <id>",
    "\\prop all": "\\prop all",
    "\\prop list": "\\prop list",
    "\\prop authorize": "\\prop authorize <id>",
    "\\prop defer": "\\prop defer <id>",
    "\\prop reject": "\\prop reject <id>",
}


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_completion(session: ReplSession) -> None:
    """Build all completion caches from live Click command tree."""
    global _FLAG_MAP, _SUBCOMMAND_MAP, _TEMPLATE_MAP
    _FLAG_MAP.clear()
    _SUBCOMMAND_MAP.clear()
    _TEMPLATE_MAP.clear()

    for cmd_name, cmd_obj in session.cli_group.commands.items():
        _index_command(cmd_name, cmd_obj)

    # Index backslash commands and subcommands
    for bcmd, (synopsis, _) in _BACKSLASH_METAS.items():
        _TEMPLATE_MAP[bcmd] = synopsis
    for bcmd, subs in _BACKSLASH_SUBCOMMANDS.items():
        _SUBCOMMAND_MAP[bcmd] = subs
    for full_key, tmpl in _BACKSLASH_SUBCOMMAND_TEMPLATES.items():
        _TEMPLATE_MAP[full_key] = tmpl

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


def _pad_meta(meta: str) -> str:
    """Pad metadata string with leading spaces for visual separation."""
    if not meta:
        return ""
    return f"   {meta}"


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
                    yield Completion(sub, start_position=0, display_meta=_pad_meta(meta))
                if cmd in ("\\proposals", "\\prop"):
                    try:
                        from audiobench.cli.repl.backslash_commands import _fetch_daemon_proposals, _parse_proposal_payload
                        for p in _fetch_daemon_proposals(include_all=False):
                            pid = str(p["id"])
                            status = p.get("inference_status", "")
                            _, title, _ = _parse_proposal_payload(p.get("content", ""))
                            meta = f"[{status}] {title[:40]}" if status else title[:40]
                            yield Completion(pid, start_position=0, display_meta=_pad_meta(meta))
                    except Exception:
                        pass
                return
            elif len(tokens) == 2 and not text_before.endswith(" "):
                partial = tokens[1]
                if not (cmd in ("\\scope", "scope") and (partial.startswith("school:") or partial.startswith("work:"))):
                    for sub in _SUBCOMMAND_MAP[cmd]:
                        if sub.startswith(partial):
                            full_key = f"{cmd} {sub}"
                            meta = _TEMPLATE_MAP.get(full_key, "")
                            yield Completion(sub, start_position=-len(partial), display_meta=_pad_meta(meta))
                    if cmd in ("\\proposals", "\\prop"):
                        try:
                            from audiobench.cli.repl.backslash_commands import _fetch_daemon_proposals, _parse_proposal_payload
                            clean_partial = partial.lstrip("#")
                            for p in _fetch_daemon_proposals(include_all=True):
                                pid = str(p["id"])
                                if pid.startswith(clean_partial) or f"#{pid}".startswith(partial):
                                    status = p.get("inference_status", "")
                                    _, title, _ = _parse_proposal_payload(p.get("content", ""))
                                    meta = f"[{status}] {title[:40]}" if status else title[:40]
                                    yield Completion(pid, start_position=-len(partial), display_meta=_pad_meta(meta))
                        except Exception:
                            pass
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
                # Prefer Click short_help; fall back to backslash meta or template
                if cmd in self.session.cli_group.commands:
                    meta = self.session.cli_group.commands[cmd].short_help or _TEMPLATE_MAP.get(cmd, "")
                elif cmd in _BACKSLASH_METAS:
                    meta = _BACKSLASH_METAS[cmd][1]
                else:
                    meta = _TEMPLATE_MAP.get(cmd, "")
                yield Completion(cmd, start_position=-len(word), display_meta=_pad_meta(meta))

    def _complete_args(self, cmd: str, used_tokens: list[str], last_token: str):
        """Complete after a space — offer flags, choices, IDs, files."""

        # ── Citation completions for \bookmark cite <n> and \play cite <n> ────
        is_cite_cmd = (
            cmd in ("\\bookmark cite", "\\bm cite", "\\play cite")
            or (cmd in ("\\bookmark", "\\bm", "\\play") and used_tokens and used_tokens[0] == "cite")
        )
        if is_cite_cmd:
            citations = getattr(self.session, "last_synthesis_citations", {})
            if citations:
                for c_num, expr_id in citations.items():
                    yield Completion(str(c_num), start_position=0, display_meta=_pad_meta(f"Citation [{c_num}] (Expr #{expr_id})"))
            else:
                for c_num in ("1", "2", "3", "4"):
                    yield Completion(str(c_num), start_position=0, display_meta=_pad_meta(f"Citation [{c_num}]"))
            return

        # ── Command completions for \help <cmd> ───────────────────────────────
        if cmd in ("\\help", "help", "/help"):
            candidates = [k.lstrip("\\") for k in self.backslash_cmds] + list(self.session.cli_group.commands.keys())
            for c in sorted(set(candidates)):
                bcmd = f"\\{c}"
                if bcmd in _BACKSLASH_METAS:
                    meta = _BACKSLASH_METAS[bcmd][1]
                elif c in self.session.cli_group.commands:
                    meta = self.session.cli_group.commands[c].short_help or ""
                else:
                    meta = ""
                yield Completion(c, start_position=0, display_meta=_pad_meta(meta) if meta else None)
            return

        # ── If previous token was a flag that expects a value, complete the value ──
        if last_token.startswith("-") and cmd in _FLAG_MAP:
            for opts, val_type, help_text, choices in _FLAG_MAP[cmd]:
                if last_token in opts:
                    if choices:
                        for choice in choices:
                            yield Completion(choice, start_position=0, display_meta=_pad_meta(help_text))
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
                    yield Completion(main_opt, start_position=0, display_meta=_pad_meta(meta))

        # ── Transcript IDs ────────────────────────────────────────────────────
        if cmd in _TX_ID_CMDS:
            for t in _TRANSCRIPT_CACHE:
                yield Completion(str(t["id"]), start_position=0, display_meta=_pad_meta(t["file_name"]))

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
                        yield Completion(str(n.id), start_position=0, display_meta=_pad_meta(n.title))
            except Exception:
                pass

        # ── School IDs ────────────────────────────────────────────────────────
        is_school_target = (
            cmd in ("\\school", "school", "\\play school")
            or (cmd == "\\play" and used_tokens and used_tokens[0] == "school")
        )
        if is_school_target:
            try:
                import sqlite3
                from pathlib import Path
                from audiobench.community.community_service import get_active_school_ids
                from audiobench.core.settings import get_settings

                settings = get_settings()
                db_path = Path(settings.database_url.replace("sqlite:///", "")).resolve()
                with sqlite3.connect(str(db_path)) as conn:
                    for cid, name in get_active_school_ids(conn):
                        yield Completion(str(cid), start_position=0, display_meta=_pad_meta(f"#{cid:02d} {name}"))
            except Exception:
                pass

        # ── Scope subcommands ─────────────────────────────────────────────────
        if cmd in ("\\scope", "scope"):
            yield Completion("school:", start_position=0, display_meta=_pad_meta("Scope to school"))
            yield Completion("work:", start_position=0, display_meta=_pad_meta("Scope to work"))
            yield Completion("clear", start_position=0, display_meta=_pad_meta("Clear scope"))

        # ── Proposal IDs ──────────────────────────────────────────────────────
        is_proposal_target = (
            cmd in (
                "\\proposals show", "\\proposals authorize", "\\proposals defer", "\\proposals reject",
                "\\prop show", "\\prop authorize", "\\prop defer", "\\prop reject",
            )
            or (
                cmd in ("\\proposals", "\\prop")
                and used_tokens
                and used_tokens[0] in ("show", "authorize", "defer", "reject", "auth", "confirm", "accept", "pause", "dismiss", "info", "view", "dossier")
            )
        )
        if is_proposal_target:
            try:
                from audiobench.cli.repl.backslash_commands import _fetch_daemon_proposals, _parse_proposal_payload
                proposals = _fetch_daemon_proposals(include_all=True)
                for p in proposals:
                    pid = str(p["id"])
                    status = p.get("inference_status", "")
                    _, title, _ = _parse_proposal_payload(p.get("content", ""))
                    meta = f"[{status}] {title[:40]}" if status else title[:40]
                    yield Completion(pid, start_position=0, display_meta=_pad_meta(meta))
            except Exception:
                pass
            return


    def _complete_partial(self, cmd: str, used_tokens: list[str], word: str):
        """Complete a partially-typed token."""

        # ── Partial citation number ───────────────────────────────────────────
        is_cite_cmd = (
            cmd in ("\\bookmark cite", "\\bm cite", "\\play cite")
            or (cmd in ("\\bookmark", "\\bm", "\\play") and used_tokens and used_tokens[0] == "cite")
        )
        if is_cite_cmd:
            citations = getattr(self.session, "last_synthesis_citations", {})
            c_keys = [str(k) for k in citations.keys()] if citations else ["1", "2", "3", "4"]
            for c in c_keys:
                if c.startswith(word):
                    if citations and c.isdigit() and int(c) in citations:
                        meta = f"Citation [{c}] (Expr #{citations[int(c)]})"
                    else:
                        meta = f"Citation [{c}]"
                    yield Completion(c, start_position=-len(word), display_meta=_pad_meta(meta))
            return

        # ── Partial help command target ───────────────────────────────────────
        if cmd in ("\\help", "help", "/help"):
            candidates = [k.lstrip("\\") for k in self.backslash_cmds] + list(self.session.cli_group.commands.keys())
            for c in sorted(set(candidates)):
                if c.startswith(word):
                    bcmd = f"\\{c}"
                    if bcmd in _BACKSLASH_METAS:
                        meta = _BACKSLASH_METAS[bcmd][1]
                    elif c in self.session.cli_group.commands:
                        meta = self.session.cli_group.commands[c].short_help or ""
                    else:
                        meta = ""
                    yield Completion(c, start_position=-len(word), display_meta=_pad_meta(meta) if meta else None)
            return

        # ── Partial flag ─────────────────────────────────────────────────────
        if word.startswith("-") and cmd in _FLAG_MAP:
            for opts, val_type, help_text, choices in _FLAG_MAP[cmd]:
                for opt in opts:
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word), display_meta=_pad_meta(help_text))
            return

        # ── Partial transcript ID ─────────────────────────────────────────────
        if cmd in _TX_ID_CMDS:
            for t in _TRANSCRIPT_CACHE:
                if str(t["id"]).startswith(word):
                    yield Completion(
                        str(t["id"]), start_position=-len(word), display_meta=_pad_meta(t["file_name"])
                    )

        # ── Partial audio filename ────────────────────────────────────────────
        if cmd in _AUDIO_FILE_CMDS:
            for f in glob.glob(word + "*"):
                if f.endswith(_AUDIO_EXTS):
                    yield Completion(f, start_position=-len(word))

        # ── Partial school ID ─────────────────────────────────────────────────
        is_school_target = (
            cmd in ("\\school", "school", "\\play school")
            or (cmd == "\\play" and used_tokens and used_tokens[0] == "school")
        )
        if is_school_target:
            try:
                import sqlite3
                from pathlib import Path
                from audiobench.community.community_service import get_active_school_ids
                from audiobench.core.settings import get_settings

                settings = get_settings()
                db_path = Path(settings.database_url.replace("sqlite:///", "")).resolve()
                clean_word = word.lstrip("#")
                with sqlite3.connect(str(db_path)) as conn:
                    for cid, name in get_active_school_ids(conn):
                        if str(cid).startswith(clean_word) or f"#{cid}".startswith(word):
                            yield Completion(
                                str(cid),
                                start_position=-len(word),
                                display_meta=_pad_meta(f"#{cid:02d} {name}"),
                            )
            except Exception:
                pass
            return

        # ── Partial scope or focus targets (school:<id>, s<id>, w<id>) ────────
        if cmd in ("\\scope", "scope", "\\focus", "focus"):
            if word.startswith("school:") or word.startswith("s"):
                prefix = "school:" if word.startswith("school:") else "s"
                val = word[len(prefix):]
                try:
                    import sqlite3
                    from pathlib import Path
                    from audiobench.community.community_service import get_active_school_ids
                    from audiobench.core.settings import get_settings

                    settings = get_settings()
                    db_path = Path(settings.database_url.replace("sqlite:///", "")).resolve()
                    with sqlite3.connect(str(db_path)) as conn:
                        for cid, name in get_active_school_ids(conn):
                            if str(cid).startswith(val):
                                yield Completion(
                                    f"{prefix}{cid}",
                                    start_position=-len(word),
                                    display_meta=_pad_meta(f"#{cid:02d} {name}"),
                                )
                except Exception:
                    pass
                return

            if word.startswith("work:") or word.startswith("w"):
                prefix = "work:" if word.startswith("work:") else "w"
                val = word[len(prefix):]
                try:
                    with get_session() as sdb:
                        from audiobench.storage.models import WorkRecord
                        works = sdb.query(WorkRecord).limit(20).all()
                        for w in works:
                            if str(w.id).startswith(val):
                                yield Completion(
                                    f"{prefix}{w.id}",
                                    start_position=-len(word),
                                    display_meta=_pad_meta(w.title[:30]),
                                )
                except Exception:
                    pass
                return

        # ── Partial proposal ID ───────────────────────────────────────────────
        is_proposal_target = (
            cmd in (
                "\\proposals show", "\\proposals authorize", "\\proposals defer", "\\proposals reject",
                "\\prop show", "\\prop authorize", "\\prop defer", "\\prop reject",
            )
            or (
                cmd in ("\\proposals", "\\prop")
                and used_tokens
                and used_tokens[0] in ("show", "authorize", "defer", "reject", "auth", "confirm", "accept", "pause", "dismiss", "info", "view", "dossier")
            )
        )
        if is_proposal_target:
            try:
                from audiobench.cli.repl.backslash_commands import _fetch_daemon_proposals, _parse_proposal_payload
                clean_word = word.lstrip("#")
                proposals = _fetch_daemon_proposals(include_all=True)
                for p in proposals:
                    pid = str(p["id"])
                    if pid.startswith(clean_word) or f"#{pid}".startswith(word):
                        status = p.get("inference_status", "")
                        _, title, _ = _parse_proposal_payload(p.get("content", ""))
                        meta = f"[{status}] {title[:40]}" if status else title[:40]
                        yield Completion(pid, start_position=-len(word), display_meta=_pad_meta(meta))
            except Exception:
                pass
            return




# ── Ghost Suggestion ──────────────────────────────────────────────────────────

class AudioBenchAutoSuggest(AutoSuggest):
    """Three-tier ghost suggestion:
      1. History match (AutoSuggestFromHistory) — always first
      2. Command template — when cmd is complete and cursor is at a space
      3. Sequential prediction — when starting a new command, predict from transition matrix
    """

    def __init__(self, session: ReplSession | None = None):
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        self._history = AutoSuggestFromHistory()
        self.session = session

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

        # For subcommands with args: "\bookmark cite " → ghost "1" or "<n>"
        if text.endswith(" ") and len(tokens) == 2:
            full_key = f"{tokens[0]} {tokens[1]}"
            if full_key in ("\\bookmark cite", "\\bm cite", "\\play cite"):
                if self.session and getattr(self.session, "last_synthesis_citations", None):
                    return Suggestion("1")
            template = _TEMPLATE_MAP.get(full_key, "")
            if template and template.startswith(full_key):
                rest = template[len(full_key):].strip()
                if rest:
                    return Suggestion(rest)

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
