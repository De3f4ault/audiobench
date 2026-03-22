"""REPL session state — holds mutable state across the interactive session.

Provides:
    - ReplSession: Mutable state container (context, history, navigation)
    - CONTEXT_AWARE_COMMANDS: Set of commands that accept a transcript ID
"""

from __future__ import annotations

import contextlib
import readline

import click

from audiobench.cli.display.theme import WARNING, console

# ── Commands that accept a transcript ID as first positional arg ──
# When context is set and no ID is given, the REPL auto-injects it.

CONTEXT_AWARE_COMMANDS = {
    "show",  # show <id>
    "ask",  # ask <id> <question>
    "summarize",  # summarize <id>
    "vocab",  # vocab <id>
    "export",  # export <id>
    "delete",  # delete <id>
    "chat",  # chat <id...>  (nargs=-1, optional)
    "speak",  # speak <id>
}


class ReplSession:
    """Holds mutable REPL state across the interactive session."""

    def __init__(self, cli_group: click.Group) -> None:
        self.cli_group = cli_group
        self.last_id: int | None = None
        self.context_record: dict | None = None

        # Use settings.data_dir for REPL history (project-local)
        from audiobench.core.settings import get_settings

        self._history_file = get_settings().data_dir / "repl_history"
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._repo = None
        self._command_count = 0
        self._history_ids: list[int] = []  # for .next / .prev navigation
        self._history_cursor: int = -1

    def _get_repo(self):
        """Lazy-init the transcription repository."""
        if self._repo is None:
            from audiobench.core.db_engine import init_db
            from audiobench.storage.repository import TranscriptionRepository

            init_db()
            self._repo = TranscriptionRepository()
        return self._repo

    def _load_history_ids(self) -> None:
        """Load all transcript IDs for .next/.prev navigation."""
        try:
            repo = self._get_repo()
            records = repo.get_history(limit=500)
            # Oldest first so .next goes forward in time
            self._history_ids = [r["id"] for r in reversed(records)]
        except Exception:
            self._history_ids = []

    # ── Prompt ──

    @property
    def prompt(self) -> str:
        if self.last_id is not None:
            name = ""
            if self.context_record:
                fname = self.context_record.get("file_name", "")
                if fname and len(fname) > 25:
                    fname = fname[:22] + "..."
                name = f" {fname}" if fname else ""
            return f"\001\033[36m\002ab\001\033[0m\002 #{self.last_id}{name} > "
        return "\001\033[36m\002ab\001\033[0m\002 > "

    # ── Variable expansion ──

    def expand_vars(self, args: list[str]) -> list[str]:
        """Replace $last / $last_id / $id with the current context ID."""
        if self.last_id is None:
            return args
        sid = str(self.last_id)
        return [sid if a in ("$last", "$last_id", "$id") else a for a in args]

    # ── Context auto-injection ──

    def auto_inject_id(self, args: list[str]) -> list[str]:
        """For context-aware commands, inject the current ID if missing."""
        if not args or self.last_id is None:
            return args

        cmd_name = args[0]
        if cmd_name not in CONTEXT_AWARE_COMMANDS:
            return args

        rest = args[1:]

        # If the next arg is already a number, user specified an ID
        if rest and rest[0].isdigit():
            return args

        # If next arg is a flag, or there's no next arg — inject ID
        return [cmd_name, str(self.last_id)] + rest

    # ── Set context ──

    def set_context(self, record_id: int) -> None:
        """Set the current context to a transcription ID."""
        repo = self._get_repo()
        record = repo.get_by_id(record_id)
        if record:
            self.last_id = record_id
            self.context_record = record
            # Update navigation cursor
            if record_id in self._history_ids:
                self._history_cursor = self._history_ids.index(record_id)
        else:
            console.print(f"  [{WARNING}]Transcript #{record_id} not found[/]")

    def clear_context(self) -> None:
        """Clear the current context, return to bare prompt."""
        self.last_id = None
        self.context_record = None
        self._history_cursor = -1

    def refresh_context(self) -> None:
        """Refresh the current context record from DB."""
        if self.last_id is not None:
            repo = self._get_repo()
            self.context_record = repo.get_by_id(self.last_id)

    # ── Readline history ──

    def load_history(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            readline.read_history_file(str(self._history_file))
        readline.set_history_length(1000)

    def save_history(self) -> None:
        with contextlib.suppress(OSError):
            readline.write_history_file(str(self._history_file))
