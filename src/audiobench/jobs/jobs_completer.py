"""Context-aware autocomplete and ghost suggestions for Jobs REPL slash commands.

Provides:
  - JobsCompleter: 2-column popup menu with command descriptions, flags, and candidate Job IDs.
  - JobsAutoSuggest: History-first ghost text falling back to command synopsis.
"""

from __future__ import annotations

from prompt_toolkit.auto_suggest import AutoSuggest, AutoSuggestFromHistory, Suggestion
from prompt_toolkit.completion import Completer, Completion

from audiobench.cli.shared.repl_shell import _pad_meta
from audiobench.core.db_session import get_session
from audiobench.jobs.jobs_meta import (
    JOBS_COMMAND_REGISTRY,
    CommandMeta,
    get_all_job_command_names,
    get_job_command_meta,
)
from audiobench.storage.models import UnifiedJob


class JobsCompleter(Completer):
    """Context-aware autocompleter for the interactive Jobs REPL."""

    def get_completions(self, document, complete_event):
        text_before = document.text_before_cursor
        if not text_before.startswith("/"):
            return

        parts = text_before.split(None, 1)

        # Position 0: completing the slash command itself
        if len(parts) == 1 and not text_before.endswith(" "):
            prefix = parts[0].lower()
            for meta in JOBS_COMMAND_REGISTRY:
                for name in meta.all_names:
                    if name.startswith(prefix):
                        yield Completion(
                            name,
                            start_position=-len(prefix),
                            display=name,
                            display_meta=_pad_meta(meta.summary),
                        )
            return

        # Position 1+: completing arguments for a specific command
        cmd_name = parts[0].lower()
        arg_part = parts[1] if len(parts) > 1 else ""

        if cmd_name in ("/retry", "retry", "/rerun"):
            yield from self._complete_retry(arg_part)
        elif cmd_name in ("/cancel", "cancel", "/stop", "/kill"):
            yield from self._complete_cancel(arg_part)
        elif cmd_name in ("/tail", "tail", "/follow", "/log", "log", "/fg", "fg", "/logs"):
            yield from self._complete_active(arg_part)
        elif cmd_name in ("/clean", "clean", "/prune"):
            yield from self._complete_clean(arg_part)

    def _complete_retry(self, arg_part: str):
        if "--all".startswith(arg_part):
            yield Completion(
                "--all",
                start_position=-len(arg_part),
                display="--all",
                display_meta=_pad_meta("Retry all failed and cancelled jobs"),
            )

        try:
            with get_session() as session:
                failed_jobs = (
                    session.query(UnifiedJob)
                    .filter(UnifiedJob.status.in_(["failed", "cancelled"]))
                    .order_by(UnifiedJob.id.desc())
                    .limit(20)
                    .all()
                )
                for j in failed_jobs:
                    jid_str = str(j.id)
                    label = j.file_label or j.command_display or j.job_type
                    if len(label) > 28:
                        label = label[:25] + "..."
                    meta = f"[{j.status}] {label}"
                    if jid_str.startswith(arg_part) or f"#{jid_str}".startswith(arg_part):
                        insert_val = jid_str if not arg_part.startswith("#") else f"#{jid_str}"
                        yield Completion(
                            insert_val,
                            start_position=-len(arg_part),
                            display=f"#{jid_str}",
                            display_meta=_pad_meta(meta),
                        )
        except Exception:
            pass

    def _complete_cancel(self, arg_part: str):
        if "--all".startswith(arg_part):
            yield Completion(
                "--all",
                start_position=-len(arg_part),
                display="--all",
                display_meta=_pad_meta("Cancel all running and pending jobs"),
            )

        if "--batch".startswith(arg_part):
            yield Completion(
                "--batch",
                start_position=-len(arg_part),
                display="--batch",
                display_meta=_pad_meta("Cancel all jobs in a batch (--batch <id>)"),
            )

        try:
            with get_session() as session:
                active_jobs = (
                    session.query(UnifiedJob)
                    .filter(UnifiedJob.status.in_(["running", "pending"]))
                    .order_by(UnifiedJob.priority.asc(), UnifiedJob.id.asc())
                    .limit(30)
                    .all()
                )
                for j in active_jobs:
                    jid_str = str(j.id)
                    label = j.file_label or j.command_display or j.job_type
                    if len(label) > 28:
                        label = label[:25] + "..."
                    meta = f"[{j.status}] {label}"
                    if jid_str.startswith(arg_part) or f"#{jid_str}".startswith(arg_part):
                        insert_val = jid_str if not arg_part.startswith("#") else f"#{jid_str}"
                        yield Completion(
                            insert_val,
                            start_position=-len(arg_part),
                            display=f"#{jid_str}",
                            display_meta=_pad_meta(meta),
                        )
        except Exception:
            pass

    def _complete_active(self, arg_part: str):
        try:
            with get_session() as session:
                active_jobs = (
                    session.query(UnifiedJob)
                    .filter(UnifiedJob.status.in_(["running", "pending", "done", "failed"]))
                    .order_by(UnifiedJob.id.desc())
                    .limit(20)
                    .all()
                )
                for j in active_jobs:
                    jid_str = str(j.id)
                    label = j.file_label or j.command_display or j.job_type
                    if len(label) > 28:
                        label = label[:25] + "..."
                    meta = f"[{j.status}] {label}"
                    if jid_str.startswith(arg_part) or f"#{jid_str}".startswith(arg_part):
                        insert_val = jid_str if not arg_part.startswith("#") else f"#{jid_str}"
                        yield Completion(
                            insert_val,
                            start_position=-len(arg_part),
                            display=f"#{jid_str}",
                            display_meta=_pad_meta(meta),
                        )
        except Exception:
            pass

    def _complete_clean(self, arg_part: str):
        if "--all".startswith(arg_part):
            yield Completion(
                "--all",
                start_position=-len(arg_part),
                display="--all",
                display_meta=_pad_meta("Also purge running tasks (dangerous)"),
            )


class JobsAutoSuggest(AutoSuggest):
    """Two-tier ghost suggestion for Jobs REPL:

    1. History-first (previous user commands)
    2. Command synopsis template when user types a command + space
    """

    def __init__(self):
        self._history = AutoSuggestFromHistory()

    def get_suggestion(self, buffer, document):
        suggestion = self._history.get_suggestion(buffer, document)
        if suggestion:
            return suggestion

        text = document.text
        if not text.startswith("/") or not text.endswith(" "):
            return None

        tokens = text.split()
        if len(tokens) == 1:
            cmd_name = tokens[0].lower()
            meta = get_job_command_meta(cmd_name)
            if meta and meta.synopsis:
                return Suggestion(meta.synopsis)

        return None
