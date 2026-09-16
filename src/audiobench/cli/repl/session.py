"""REPL session state — holds mutable state across the interactive session.

Provides:
    - NavigationFrame: Immutable snapshot of one level of the navigation stack
    - ReplSession: Mutable state container (context, history, navigation stack)
    - CONTEXT_AWARE_COMMANDS: Set of commands that accept a transcript ID
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import click
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory

from audiobench.cli.display.theme import DIM, SUCCESS, WARNING, console
from audiobench.core.focused_entity import FocusedEntity
from audiobench.core.logger_factory import get_logger

logger = get_logger("cli.repl.session")


# ── Navigation Stack ─────────────────────────────────────────


@dataclass
class NavigationFrame:
    """Frozen snapshot of one level of the navigation stack.

    When a command pushes a frame before launching a sub-command,
    the frame captures the context label and any state needed to
    resume (scroll position, selected ID, wizard step, etc.).

    The ``resumed`` flag is set to True when the frame is popped,
    so the relaunched command knows it is re-entering, not starting fresh.
    """

    context: str
    state: dict = field(default_factory=dict)
    resumed: bool = False
    intent: str = ""

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
        self._root_focus: FocusedEntity | None = None

        # Use settings.data_dir for REPL history (project-local)
        from audiobench.core.settings import get_settings

        _data_dir = get_settings().data_dir
        _data_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = _data_dir / "repl_history"
        self._repo = None
        self._command_count = 0
        self._history_ids: list[int] = []  # for .next / .prev navigation
        self._history_cursor: int = -1

        # ── Navigation stack ──
        self.navigation_stack: deque[NavigationFrame] = deque()
        self._stack_path = _data_dir / "session_stack.json"

        # ── Security / Privacy Tier ──────────────────────────────────────
        # 0 = public (default), 1 = relational (10-min TTL), 2+ = intimate
        self.unlocked_tier: int = 0
        self.tier1_unlocked_at: datetime | None = None
        self.last_keystroke_at: datetime = datetime.now(UTC)
        # Set to True by the idle-lock watcher thread; cleared on next prompt
        self._idle_lock_fired: bool = False
        # Session-level unlock: stays active for the whole REPL session.
        # Only cleared by \lock or the idle-lock watcher.
        # Tier 3 is intentionally excluded — it is always per-query.
        self.session_unlock_tier: int = 0

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

    # ── Focus ──
    @property
    def focus(self) -> FocusedEntity | None:
        for frame in reversed(self.navigation_stack):
            focus_dict = frame.state.get("focus")
            if focus_dict:
                return FocusedEntity(
                    id=focus_dict["id"],
                    type=focus_dict["type"],
                    label=focus_dict.get("label", f"{focus_dict['type'].title()} #{focus_dict['id']}"),
                    chapter_index=focus_dict.get("chapter_index"),
                    chapter_title=focus_dict.get("chapter_title")
                )
        return self._root_focus

    @focus.setter
    def focus(self, value: FocusedEntity | None) -> None:
        if not self.navigation_stack:
            self._root_focus = value
        else:
            if value is None:
                self.navigation_stack[-1].state.pop("focus", None)
            else:
                self.navigation_stack[-1].state["focus"] = {
                    "id": value.id,
                    "type": value.type,
                    "label": value.label,
                    "chapter_index": value.chapter_index,
                    "chapter_title": value.chapter_title
                }
            self._persist_stack()

    # ── Prompt ──

    @property
    def last_id(self) -> int | None:
        """Derived property returning the relevant transcript ID for the current focus."""
        if not self.focus:
            return None
        if self.focus.type == "transcript":
            return self.focus.id
        if self.focus.type == "file":
            repo = self._get_repo()

            # If a chapter is focused, get the transcript ID of THAT chapter!
            if self.focus.chapter_index is not None:
                from audiobench.core.db_session import get_session
                from audiobench.storage.chapter_repository import get_chapter_repo
                from audiobench.storage.models import ChapterRecord

                chap = get_chapter_repo().get_chapter_by_index(
                    self.focus.id, self.focus.chapter_index
                )
                if chap and chap.id:
                    with get_session() as db:
                        rec = db.query(ChapterRecord).filter_by(id=chap.id).first()
                        if rec and rec.transcription_id:
                            return rec.transcription_id
                return None

            tx = repo.get_latest_transcript_for_file(self.focus.id)
            return tx["id"] if tx else None
        return None

    def focus_chapter(self, chapter_index: int, chapter_title: str) -> bool:
        """Set focus to a specific chapter of the currently focused file."""
        if not self.focus or self.focus.type != "file":
            console.print(f"  [{WARNING}]You must focus on an audio file first.[/]")
            return False

        self.focus.chapter_index = chapter_index
        self.focus.chapter_title = chapter_title
        return True

    def clear_chapter_focus(self) -> None:
        """Clear the currently focused chapter."""
        if self.focus:
            self.focus.chapter_index = None
            self.focus.chapter_title = None

    @property
    def prompt(self) -> Any:
        """Return the dynamic prompt, formatted for prompt_toolkit.

        Badge order (left → right):
            [N job(s)]   background job indicator
            ★N           session-unlock tier (only when session_unlock_tier > 0)
            ↓N           navigation stack depth
            label ❯      focus label + arrow
        """
        job_badge = ""
        try:
            from pathlib import Path

            from audiobench.core.settings import get_settings

            active_file = Path(get_settings().data_dir) / "jobs.active"
            if active_file.exists():
                count = active_file.read_text().strip()
                if count.isdigit() and int(count) > 0:
                    n = int(count)
                    label = "job" if n == 1 else "jobs"
                    job_badge = f"<ansiyellow>[{n} {label}]</ansiyellow> "
        except Exception:
            pass

        # ★N — permanent session-unlock indicator.
        # Visible whenever session_unlock_tier > 0, reminding the user their
        # session is elevated even when not actively typing security commands.
        sec_badge = ""
        if self.session_unlock_tier > 0:
            sec_badge = f"<ansigreen>★{self.session_unlock_tier}</ansigreen> "

        depth = len(self.navigation_stack)
        depth_badge = f"<style color='gray'>↓{depth}</style> " if depth > 0 else ""

        if self.focus:
            label = self.focus.display_label
            label = label.replace("<", "&lt;").replace(">", "&gt;")
            return HTML(
                f"{job_badge}"
                f"{sec_badge}"
                f"<ansicyan><b>{label}</b></ansicyan> "
                f"{depth_badge}"
                f"<ansimagenta><b>❯</b></ansimagenta> "
            )
        return HTML(
            f"{job_badge}{sec_badge}{depth_badge}"
            f"<ansimagenta><b>❯</b></ansimagenta> "
        )

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
        rest = args[1:]

        # Auto-inject file path and chapter for 'transcribe' if focused on a file
        if cmd_name == "transcribe" and self.focus and self.focus.type == "file":
            # Check if user already provided files (those without a leading dash, before any double dash)
            has_files = False
            for arg in rest:
                if not arg.startswith("-"):
                    has_files = True
                    break

            injections = []
            if not has_files:
                repo = self._get_repo()
                audio_file = repo.get_audio_file(self.focus.id)
                if audio_file and audio_file.get("file_path"):
                    injections.append(str(audio_file["file_path"]))

            if self.focus.chapter_index is not None and "--chapters" not in rest:
                injections.extend(["--chapters", str(self.focus.chapter_index)])

            return [cmd_name] + injections + rest

        if cmd_name not in CONTEXT_AWARE_COMMANDS:
            return args

        # If the next arg is already a number, user specified an ID
        if rest and rest[0].isdigit():
            return args

        # If next arg is a flag, or there's no next arg — inject ID
        return [cmd_name, str(self.last_id)] + rest

    # ── Set context ──

    def set_context(self, record_id: int) -> None:
        """Set the current context using a transcript ID."""
        repo = self._get_repo()
        record = repo.get_by_id(record_id)

        if not record:
            console.print(f"  [{WARNING}]Transcript #{record_id} not found[/]")
            return

        # If transcript is linked to a file, focus on the file instead
        audio_file_id = record.get("audio_file_id")
        if audio_file_id:
            audio_file = repo.get_audio_file(audio_file_id)
            if audio_file:
                self.focus = FocusedEntity(
                    type="file", id=audio_file_id, label=audio_file["file_name"]
                )
            else:
                # Fallback if DB is inconsistent
                self.focus = FocusedEntity(
                    type="transcript",
                    id=record_id,
                    label=record.get("file_name", f"Transcript #{record_id}"),
                )
        else:
            self.focus = FocusedEntity(
                type="transcript",
                id=record_id,
                label=record.get("file_name", f"Transcript #{record_id}"),
            )

        # Update navigation cursor
        if record_id in self._history_ids:
            self._history_cursor = self._history_ids.index(record_id)

    def clear_context(self) -> None:
        """Clear the current context, return to bare prompt."""
        self.focus = None
        self._history_cursor = -1

    def refresh_context(self) -> None:
        """Refresh the current context if needed."""
        # Refresh logic is implicit now since last_id fetches live from DB
        pass

    # ── Security / Privacy Tier ──────────────────────────────────────────

    def effective_tier(self) -> int:
        """Return the current effective unlocked tier.

        Priority (highest wins):
          1. session_unlock_tier  — set by \\unlock N --session, persists for
                                    the whole REPL session (cleared by \\lock
                                    or idle-lock only).
          2. unlocked_tier        — set by \\unlock N without --session.
                                    Tier 1 has a 10-minute TTL; Tier 2+ is
                                    per-query (no TTL, re-auth each time).

        Tier 3 can never get a session_unlock_tier — it is always per-query.
        """
        # Session unlock wins if it's higher than the per-query unlock
        if self.session_unlock_tier > 0:
            return max(self.session_unlock_tier, self._per_query_tier())
        return self._per_query_tier()

    def _per_query_tier(self) -> int:
        """Evaluate the per-query unlock tier with TTL logic."""
        now = datetime.now(UTC)
        if self.unlocked_tier >= 2:
            return self.unlocked_tier
        if self.unlocked_tier == 1 and self.tier1_unlocked_at is not None:
            elapsed = (now - self.tier1_unlocked_at).total_seconds()
            if elapsed < 600:  # 10 minutes
                return 1
            # TTL expired — silently demote
            self.unlocked_tier = 0
            self.tier1_unlocked_at = None
        return 0

    def start_idle_lock_watcher(self, idle_seconds: int = 180) -> None:
        """Start a daemon thread that drops the unlock tier after N seconds idle.

        Idle is defined as time elapsed since the last keypress recorded
        in ``last_keystroke_at``. Background jobs, daemons, and AI responses
        do NOT reset the idle clock — only physical keyboard input does.

        When the lock fires, ``_idle_lock_fired`` is set to True so the
        next prompt render can display a non-intrusive notice.

        Args:
            idle_seconds: Inactivity threshold in seconds (0 = disabled).
        """
        if idle_seconds <= 0:
            return

        def _watcher() -> None:
            while True:
                time.sleep(15)          # poll every 15 s
                if self.unlocked_tier == 0 and self.session_unlock_tier == 0:
                    continue            # nothing to lock, skip cheaply
                idle = (datetime.now(UTC) - self.last_keystroke_at).total_seconds()
                if idle >= idle_seconds:
                    self.unlocked_tier = 0
                    self.tier1_unlocked_at = None
                    self.session_unlock_tier = 0   # session unlock cleared on idle
                    self._idle_lock_fired = True
                    logger.debug("Idle-lock fired after %.0fs of inactivity", idle)

        t = threading.Thread(target=_watcher, daemon=True, name="ab-idle-lock")
        t.start()
        logger.debug("Idle-lock watcher started (threshold: %ds)", idle_seconds)

    # ── Navigation stack ─────────────────────────────────────

    def push_frame(self, frame: NavigationFrame) -> None:
        """Push a context frame onto the navigation stack and persist to disk."""
        if "focus" not in frame.state and self.focus:
            f = self.focus
            frame.state["focus"] = {
                "id": f.id,
                "type": f.type,
                "chapter_index": f.chapter_index,
                "chapter_title": f.chapter_title
            }
        self.navigation_stack.append(frame)
        self._persist_stack()

    def pop_frame(self) -> NavigationFrame | None:
        """Pop the top frame. Sets resumed=True on it. Returns None at top level."""
        if not self.navigation_stack:
            return None
        frame = self.navigation_stack.pop()
        frame.resumed = True
        self._persist_stack()
        return frame

    def _persist_stack(self) -> None:
        """Serialize the navigation stack to disk for crash recovery."""
        frames = [
            {"context": f.context, "state": f.state, "resumed": f.resumed}
            for f in self.navigation_stack
        ]
        try:
            self._stack_path.write_text(json.dumps(frames, default=str))
        except Exception:
            pass

    def maybe_resume(self) -> bool:
        """On startup, check for an interrupted session and offer to restore it."""
        if not self._stack_path.exists():
            return False
        try:
            data = self._stack_path.read_text().strip()
            if not data:
                return False
            frames = json.loads(data)
            if not frames:
                return False
            count = len(frames)
            console.print(
                f"\n  [{WARNING}]Interrupted session found[/] [{DIM}]({count} frame(s)).[/]"
            )
            confirm = input("  Resume? [Y/n] ").strip().lower()
            if confirm in ("", "y", "yes"):
                self.navigation_stack = deque(
                    NavigationFrame(**f) for f in frames
                )
                console.print(f"  [{SUCCESS}]✓ Session restored.[/]\n")
                return True
            self._stack_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False

    # ── History ──

    def get_history(self) -> FileHistory:
        """Return a prompt_toolkit FileHistory instance."""
        return FileHistory(str(self._history_file))
