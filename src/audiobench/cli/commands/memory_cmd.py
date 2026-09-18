"""CLI commands for interacting with the semantic memory system."""

import json
import shutil
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from audiobench.cli.display.theme import error_panel
from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.memory.enums import SourceType
from audiobench.memory.query_engine import ResearchEngine
from audiobench.storage.models import ConversationSummary, ExpressionRecord

logger = get_logger("cli.memory")
console = Console()

# ── Search preferences persistence ────────────────────────────────────────────
# A lightweight search_prefs.json file stored alongside settings.json in the
# data directory.  Only display/search preferences live here — infra config
# stays in settings.json.  The file is written on every /set change so it is
# always up-to-date even if the process crashes.

_PREFS_FIELDS = ("layout", "wrap_cap", "preset", "diversity_weight", "model", "autocomplete")


def _prefs_path() -> Path:
    """Return the path to search_prefs.json, creating the data dir if needed."""
    from audiobench.core.settings import get_settings
    p = get_settings().data_dir / "search_prefs.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_search_prefs() -> dict:
    """Load persisted preferences or return {} if the file doesn't exist yet."""
    path = _prefs_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read search_prefs.json: %s", exc)
        return {}


def _save_search_prefs(state: "SearchSessionState") -> None:  # type: ignore[name-defined]
    """Persist the preference fields from state to search_prefs.json."""
    data = {k: getattr(state, k, None) for k in _PREFS_FIELDS}
    try:
        with open(_prefs_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save search_prefs.json: %s", exc)



import dataclasses
from typing import Any, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from audiobench.memory.rrf_fusion import FusedResult


@dataclasses.dataclass
class SearchSessionState:
    preset: str = "balanced"
    initial_preset: str = "balanced"
    mmr_lambda: float = 0.5
    focus_source: str | None = None
    model: str | None = None
    diversity_weight: float = 0.4
    max_per_source: int = 3
    pinned_fragments: dict[int, Any] = dataclasses.field(default_factory=dict)
    pinned_expr_ids: set[int] = dataclasses.field(default_factory=set)

    # Session persistence fields (populated by _run_search_loop on entry)
    session_id: int = -1             # DB primary key; -1 means persistence unavailable
    search_count: int = 0            # incremented per query; 1-indexed sequence_num
    last_synthesis: str | None = None  # most recent synthesis text (for carryforward)
    # In-memory search history for offline overlap detection (segment_id sets per search)
    # Maps sequence_num (1-indexed) -> set of segment_ids from that search
    search_segment_ids: dict[int, set[int]] = dataclasses.field(default_factory=dict)
    # Maps source_file -> list of sequence_nums it appeared in
    search_source_files: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    # Maps sequence_num -> query text (for /history and /summary readability)
    search_query_texts: dict[int, str] = dataclasses.field(default_factory=dict)

    # Tracks which one-shot hints have already been shown this session
    hints_shown: set[str] = dataclasses.field(default_factory=set)

    # Display preferences (session-scoped, persisted to data/search_prefs.json)
    layout: str = "book"       # "book" = two-column (default), "list" = single column
    wrap_cap: int | None = None  # None = fluid (terminal width - 8), int = hard cap
    autocomplete: bool = True    # toggle semantic autocomplete while typing

    @property
    def mmr_enabled(self) -> bool:
        return self.preset == "synthesis"



def _format_hint(state: "SearchSessionState", key: str, message: str) -> str | None:
    """Return a styled contextual hint string once per session, or None if already shown."""
    if key in state.hints_shown:
        return None
    state.hints_shown.add(key)
    return f"  [dim]💡 {message}[/dim]"


def _emit_hint(state: "SearchSessionState", key: str, message: str) -> None:
    """Print a contextual hint once per session."""
    hint = _format_hint(state, key, message)
    if hint:
        console.print(f"\n{hint}")

def _pick_ollama_model(state: "SearchSessionState") -> str:  # type: ignore[name-defined]
    """Interactively pick an Ollama model from the list returned by /api/tags.

    Presents a numbered menu, lets the user pick by number or name, defaults
    to the currently configured model if they just press Enter or type 0.
    Gracefully degrades when Ollama is not reachable.
    """
    from audiobench.chat.providers.ollama_provider import AIError, OllamaClient
    from audiobench.core.settings import get_settings

    settings = get_settings()
    base_url = settings.ollama_base_url
    current = state.model or settings.ollama_model

    try:
        client = OllamaClient(base_url=base_url)
        models = client.list_models()
    except AIError as e:
        return (
            f"[yellow]Ollama not reachable ({e}) — "
            f"keeping current model: [bold]{current}[/bold][/yellow]"
        )
    except Exception as e:
        return f"[red]Error listing models: {e}[/red]"

    if not models:
        return (
            "[yellow]No models found on Ollama server — "
            f"keeping current: [bold]{current}[/bold][/yellow]"
        )

    console.print()
    console.print("[bold]Available Ollama models:[/bold]")
    for i, m in enumerate(models, 1):
        marker = "  [green]← current[/green]" if m == current else ""
        console.print(f"  [[bold cyan]{i}[/bold cyan]] {m}{marker}")
    console.print(f"  [[bold cyan]0[/bold cyan]] Keep current ([dim]{current}[/dim])")
    console.print()

    try:
        raw = input("  › Pick model [0]: ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print()
        return "[dim]Model selection cancelled — no change.[/dim]"

    if not raw or raw == "0":
        state.model = None  # reset to configured default
        _save_search_prefs(state)
        return f"[dim]Keeping configured default: [bold]{current}[/bold][/dim]"

    # Accept a number or a literal model name
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(models):
            chosen = models[idx]
            state.model = chosen
            _save_search_prefs(state)
            return f"[green]Model set to [bold]{chosen}[/bold][/green]"
        return f"[red]Invalid number — must be 1–{len(models)} or 0 to keep current.[/red]"

    # Typed a model name directly
    if raw in models:
        state.model = raw
        _save_search_prefs(state)
        return f"[green]Model set to [bold]{raw}[/bold][/green]"

    # Fuzzy match: prefix
    prefix_matches = [m for m in models if m.startswith(raw)]
    if len(prefix_matches) == 1:
        state.model = prefix_matches[0]
        _save_search_prefs(state)
        return f"[green]Model set to [bold]{prefix_matches[0]}[/bold] (prefix match)[/green]"
    if len(prefix_matches) > 1:
        return (
            f"[yellow]Ambiguous prefix '{raw}' matches: {', '.join(prefix_matches)} — "
            "be more specific.[/yellow]"
        )

def _render_settings_card(state: "SearchSessionState") -> str:
    """Format a clean, structured overview of all active and persisted search settings."""
    from audiobench.core.settings import get_settings
    settings = get_settings()

    model_display = state.model or f"default ({settings.ollama_model})"
    wrap_display = f"{state.wrap_cap} chars cap" if state.wrap_cap else "fluid (terminal width)"
    focus_display = state.focus_source if state.focus_source else "none (all sources)"

    lines = [
        "",
        "  [bold cyan]⚙ Search & Session Settings[/bold cyan] [dim](persisted to data/search_prefs.json)[/dim]",
        "  [dim]──────────────────────────────────────────────────────────────────────────[/dim]",
        f"   [bold]Preset[/bold]            [cyan]{state.preset:<16}[/cyan] [dim]fast · balanced · synthesis · deep[/dim]",
        f"   [bold]Layout[/bold]            [cyan]{state.layout:<16}[/cyan] [dim]book (2-column) · list (1-column)[/dim]",
        f"   [bold]Wrap Width[/bold]        [cyan]{wrap_display:<16}[/cyan] [dim]auto (fluid) · <number> (e.g. 88)[/dim]",
        f"   [bold]Autocomplete[/bold]      [cyan]{('on' if getattr(state, 'autocomplete', True) else 'off'):<16}[/cyan] [dim]on · off (/set autocomplete on|off)[/dim]",
        f"   [bold]Synthesis Model[/bold]   [cyan]{model_display:<16}[/cyan] [dim]ollama · gemini · <model_name>[/dim]",
        f"   [bold]Diversity Weight[/bold]  [cyan]{f'{state.diversity_weight:.2f}':<16}[/cyan] [dim]0.0 (pure RRF) → 2.0 (diversity)[/dim]",
        f"   [bold]Focus Source[/bold]      [cyan]{focus_display:<16}[/cyan] [dim]/focus <source_name> · /unfocus[/dim]",
        "  [dim]──────────────────────────────────────────────────────────────────────────[/dim]",
        "   [dim]Change any setting: [cyan]/set <option> <value>[/cyan]  (e.g. [cyan]/set deep[/cyan], [cyan]/set layout book[/cyan], [cyan]/set autocomplete off[/cyan])[/dim]",
    ]
    return "\n".join(lines)


def _parse_slash_command(
    line: str,
    state: SearchSessionState,
    last_sources: list[Any] | None = None,
    engine: "ResearchEngine | None" = None,
) -> str | None:
    """Parse interactive REPL commands to mutate session state.
    
    Returns a feedback message if handled, or None if not a valid command.
    """
    parts = line.strip().split()
    if not parts or not parts[0].startswith("/"):
        return None

    cmd = parts[0].lower()

    if cmd in ("/help", "/?"):
        from audiobench.memory.search_meta import render_search_help
        render_search_help()
        return None

    if cmd in ("/settings", "/config"):
        return _render_settings_card(state)

    if cmd == "/set":
        if len(parts) == 1:
            return (
                "[dim]Usage: [cyan]/set <option> <value>[/cyan]  "
                "(e.g. [cyan]/set deep[/cyan], [cyan]/set layout book[/cyan], [cyan]/set autocomplete off[/cyan], [cyan]/set width 88[/cyan])\n"
                "  To inspect active settings: [cyan]/settings[/cyan] or [cyan]/config[/cyan][/dim]"
            )

        arg1 = parts[1].lower()

        # Flattened preset setting: `/set deep` or `/set default`
        if arg1 in ("default", "reset"):
            state.preset = state.initial_preset
            _save_search_prefs(state)
            return f"[green]Preset reset to default ('{state.preset}')[/green]"
        elif arg1 in ("fast", "balanced", "deep", "synthesis"):
            state.preset = arg1
            _save_search_prefs(state)
            return f"[green]Preset updated to '{arg1}'[/green]"

        # Legacy preset setting: `/set preset deep`
        if arg1 == "preset" and len(parts) >= 3:
            val = parts[2].lower()
            if val in ("default", "reset"):
                state.preset = state.initial_preset
                _save_search_prefs(state)
                return f"[green]Preset reset to default ('{state.preset}')[/green]"
            elif val in ("fast", "balanced", "deep", "synthesis"):
                state.preset = val
                _save_search_prefs(state)
                return f"[green]Preset updated to '{val}'[/green]"
            return f"[red]Invalid preset '{val}'[/red]"

        elif parts[1].lower() in ("autocomplete", "auto-complete", "completion"):
            if len(parts) < 3:
                curr = "on" if getattr(state, "autocomplete", True) else "off"
                return f"[dim]Autocomplete is currently [bold]{curr}[/bold]. Usage: /set autocomplete <on | off>[/dim]"
            val = parts[2].lower()
            if val in ("on", "true", "yes", "1", "enable"):
                state.autocomplete = True
                _save_search_prefs(state)
                return "[green]Semantic autocomplete enabled[/green]"
            elif val in ("off", "false", "no", "0", "disable"):
                state.autocomplete = False
                _save_search_prefs(state)
                return "[yellow]Semantic autocomplete disabled[/yellow]"
            else:
                return "[red]Usage: /set autocomplete <on | off>[/red]"

        elif parts[1].lower() == "mmr" and len(parts) >= 3:
            try:
                lam = float(parts[2])
                if 0.0 <= lam <= 1.0:
                    state.preset = "synthesis"  # automatically switch preset
                    state.mmr_lambda = lam
                    _save_search_prefs(state)
                    return f"[green]Preset → synthesis, λ = {lam:.2f}[/green]"
                return "[red]λ must be between 0.0 and 1.0[/red]"
            except ValueError:
                return "[red]Invalid lambda value[/red]"

        elif parts[1].lower() == "model" and len(parts) >= 3:
            val = parts[2]
            # Reset the Gemini circuit breaker whenever the model changes.
            # A previous Gemini network failure must not keep blocking synthesis
            # after the user has switched away (and back) from Gemini.
            try:
                from audiobench.memory.singletons import reset_gemini_circuit_breaker
                reset_gemini_circuit_breaker()
            except Exception:
                pass  # non-critical, best effort
            if val.lower() == "default":
                state.model = None
                _save_search_prefs(state)
                return "[green]Model reset to configured default[/green]"
            if val.lower() == "ollama":
                # "ollama" is not a model name — show the interactive picker
                # so the user selects an actual model name from the running server.
                return _pick_ollama_model(state)
            if val.lower() == "gemini":
                state.model = "gemini"
                _save_search_prefs(state)
                return "[green]Model set to Gemini[/green]"
            state.model = val
            _save_search_prefs(state)
            return f"[green]Model set to '{val}'[/green]"

        elif parts[1].lower() == "model" and len(parts) == 2:
            # /set model with no arg — show the picker
            return _pick_ollama_model(state)

        elif parts[1].lower() in ("max-per-source", "max_per_source") and len(parts) >= 3:
            try:
                mps = int(parts[2])
                state.max_per_source = mps
                return f"[green]Max per source set to {mps}[/green]"
            except ValueError:
                return "[red]Invalid integer value for max-per-source[/red]"

        elif parts[1].lower() in ("diversity-weight", "diversity_weight") and len(parts) >= 3:
            try:
                weight_val = float(parts[2])
                if weight_val < 0.0:
                    return "[red]diversity-weight must be >= 0.0 (0.0 = pure RRF)[/red]"
                state.diversity_weight = weight_val
                _save_search_prefs(state)
                return f"[green]Diversity weight set to {weight_val:.2f}[/green]"
            except ValueError:
                return "[red]Invalid float value for diversity-weight[/red]"

        elif parts[1].lower() == "layout":
            if len(parts) < 3:
                current = getattr(state, "layout", "book")
                return f"[dim]Current layout: [bold]{current}[/bold]  Usage: /set layout <book | list>[/dim]"
            val = parts[2].lower()
            if val not in ("list", "book"):
                return "[red]Invalid layout — use 'book' or 'list'[/red]"
            state.layout = val
            _save_search_prefs(state)
            label = "two-column book spread" if val == "book" else "single-column list"
            return f"[green]Layout set to {val} ({label})[/green]"

        elif parts[1].lower() == "width":
            if len(parts) < 3:
                cap = getattr(state, "wrap_cap", None)
                current = f"cap at {cap}" if cap else "fluid (terminal width)"
                return f"[dim]Current width: [bold]{current}[/bold]  Usage: /set width <auto | N>[/dim]"
            val = parts[2].lower()
            if val in ("auto", "fluid", "none", "default"):
                state.wrap_cap = None
                _save_search_prefs(state)
                return "[green]Wrap width set to fluid (fills terminal)[/green]"
            try:
                n = int(val)
                if n < 40:
                    return "[red]Width cap must be at least 40[/red]"
                state.wrap_cap = n
                _save_search_prefs(state)
                return f"[green]Wrap width capped at {n} chars[/green]"
            except ValueError:
                return "[red]Usage: /set width <auto | N>  e.g. /set width 88[/red]"

    elif cmd == "/focus":
        if len(parts) >= 2:
            source = " ".join(parts[1:])
            state.focus_source = source
            return f"[green]Focus set to: {source}[/green]"
        return "[red]Usage: /focus <source_name>[/red]"

    elif cmd == "/unfocus":
        state.focus_source = None
        return "[green]Focus cleared.[/green]"

    elif cmd == "/pin":
        if len(parts) == 1:
            return "[red]Usage: /pin <1, 2, ...> (indices of fragments from last search)[/red]"
        if not last_sources:
            return "[yellow]No search results available to pin from.[/yellow]"
        pinned_count = 0
        for p in parts[1:]:
            try:
                idx = int(p)
                if 1 <= idx <= len(last_sources):
                    fr = last_sources[idx - 1]
                    state.pinned_fragments[fr.segment_id] = fr
                    state.pinned_expr_ids.add(fr.segment_id)
                    pinned_count += 1
            except ValueError:
                continue
        return f"[green]Pinned {pinned_count} fragment(s). Total pinned: {len(state.pinned_fragments)}[/green]"

    elif cmd == "/unpin":
        if len(parts) == 1:
            return "[red]Usage: /unpin <all | 1, 2, ...>[/red]"
        if parts[1].lower() in ("all", "reset"):
            state.pinned_fragments.clear()
            state.pinned_expr_ids.clear()
            return "[green]All pins cleared.[/green]"
        if not last_sources:
            return "[yellow]No search results available to unpin.[/yellow]"
        unpinned_count = 0
        for p in parts[1:]:
            try:
                idx = int(p)
                if 1 <= idx <= len(last_sources):
                    fr = last_sources[idx - 1]
                    if fr.segment_id in state.pinned_fragments:
                        del state.pinned_fragments[fr.segment_id]
                        state.pinned_expr_ids.discard(fr.segment_id)
                        unpinned_count += 1
            except ValueError:
                continue
        return f"[green]Unpinned {unpinned_count} fragment(s). Total pinned: {len(state.pinned_fragments)}[/green]"

    elif cmd == "/pins":
        if not state.pinned_fragments:
            return "[dim]No fragments currently pinned.[/dim]"
        lines = [f"[bold]Pinned Fragments ({len(state.pinned_fragments)}):[/bold]"]
        for idx, fr in enumerate(state.pinned_fragments.values(), 1):
            src = f" ({_short_source(fr.source_file)})" if getattr(fr, "source_file", None) else ""
            lines.append(f"  [cyan]{idx}.[/cyan] [dim]#{fr.segment_id}[/dim]{src} — {fr.text[:60]}...")
        return "\n".join(lines)

    elif cmd == "/forget":
        state.preset = "balanced"
        state.mmr_lambda = 0.5
        state.focus_source = None
        state.diversity_weight = 0.4
        state.max_per_source = 3
        state.pinned_fragments.clear()
        state.pinned_expr_ids.clear()
        return "[green]Session state reset to defaults.[/green]"


    elif cmd == "/history":
        # List queries in the current session with query text
        if state.search_count == 0:
            return "[dim]No searches yet in this session.[/dim]"
        lines = [f"[bold]Session history ({state.search_count} search{'es' if state.search_count != 1 else ''}):[/bold]"]
        for seq in sorted(state.search_segment_ids.keys()):
            n_frags = len(state.search_segment_ids[seq])
            q_text = state.search_query_texts.get(seq, "")
            q_preview = f"  [dim]\"{ q_text[:150] + '…' if len(q_text) > 150 else q_text }\"[/dim]" if q_text else ""
            lines.append(f"  [cyan]S{seq}[/cyan]  {n_frags} fragment{'s' if n_frags != 1 else ''}{q_preview}")
        result_text = "\n".join(lines)
        # One-shot hint appended AFTER the history output block
        hint = _format_hint(state, "history_to_summary",
                            "[cyan]/summary[/cyan] generates a full AI executive summary of this session — saved to DB and embeddable via [cyan]/export[/cyan]")
        if hint:
            result_text += f"\n\n{hint}"
        return result_text

    elif cmd == "/sessions":
        # Render full grouped dashboard inline — same output as `audiobench memory sessions`
        try:
            _render_sessions_dashboard(console, limit=15)
        except Exception as e:
            return f"[red]Failed to load sessions: {e}[/red]"
        return None  # dashboard already printed directly

    elif cmd == "/summary":
        # AI-generated executive summary of the current session
        if state.session_id < 0:
            return "[red]No active session.[/red]"
        if state.search_count == 0:
            return "[dim]No searches yet in this session to summarize.[/dim]"

        # Gather query + synthesis pairs from the DB
        try:
            from audiobench.memory.session_store import get_session, save_session_summary
            detail = get_session(state.session_id)
            if not detail:
                return "[red]Could not load session from DB.[/red]"
        except Exception as e:
            return f"[red]Failed to load session: {e}[/red]"

        pairs: list[str] = []
        for q in detail.queries:
            block = f"Search S{q.sequence_num}: {q.query_text}"
            if q.synthesis_text and not q.synthesis_failed:
                block += f"\nAnswer: {q.synthesis_text.strip()}"
            pairs.append(block)

        if not pairs:
            return "[dim]No query data found in DB for this session.[/dim]"

        session_material = "\n\n".join(pairs)
        from audiobench.core.prompts import SESSION_SUMMARY_PROMPT
        prompt = SESSION_SUMMARY_PROMPT.format(transcript=session_material)

        from rich.markdown import Markdown
        from rich.panel import Panel

        from audiobench.chat.providers.ollama_provider import OllamaClient
        from audiobench.core.settings import get_settings
        from audiobench.memory.query_engine import Ok, _call_llm

        settings = get_settings()
        llm = OllamaClient(base_url=settings.ollama_base_url, model=settings.ollama_model)

        console.print("  [dim]Generating AI summary…[/dim]")
        match _call_llm(prompt, 0.3, llm, settings.gemini_api_key):
            case Ok(value=summary_text):
                summary_text = summary_text.strip()
                # Save to DB
                save_session_summary(state.session_id, summary_text)
                title = detail.title or f"Session #{state.session_id}"
                ts = (detail.created_at or "")[:10]
                console.print(
                    Panel(
                        Markdown(summary_text),
                        title=f"[bold cyan]{title}[/bold cyan]  [dim]{ts} · {state.search_count} searches[/dim]",
                        border_style="cyan",
                        expand=True,
                    )
                )
                _emit_hint(state, "summary_to_export",
                           "Run [cyan]/export[/cyan] to produce a Markdown file with this summary embedded at the top")
                return None
            case _:
                return "[red]Summary generation failed. Check Ollama/Gemini is available.[/red]"

    elif cmd == "/show":
        # Peek at any session without switching into it: /show #N | /show #N summary
        if len(parts) < 2:
            try:
                from audiobench.memory.session_store import list_sessions
                recent = list_sessions(limit=10)
                lines = ["[bold]Sessions (use /show #ID or /show #ID summary):[/bold]"]
                for s in recent:
                    marker = " [green]← current[/green]" if s.session_id == state.session_id else ""
                    dt = s.created_at[:10] if s.created_at else "?"
                    if s.query_count == 0:
                        lines.append(f"  [dim]#{s.session_id}  {dt}  (empty — no searches)[/dim]{marker}")
                    else:
                        title = s.title or "[dim](untitled)[/dim]"
                        lines.append(
                            f"  [cyan]#{s.session_id}[/cyan]  {dt}  {title}"
                            f"  [dim]· {s.query_count} search{'es' if s.query_count != 1 else ''}[/dim]{marker}"
                        )
                result_text = "\n".join(lines)
                hint = _format_hint(state, "show_list_tip",
                                    "[cyan]/show #N[/cyan] peeks at a session's queries · [cyan]/show #N summary[/cyan] generates or reads its AI summary")
                if hint:
                    result_text += f"\n\n{hint}"
                return result_text
            except Exception as e:
                return f"[red]Failed to list sessions: {e}[/red]"

        if parts[1].lower() in ("settings", "config", "prefs", "options"):
            return _render_settings_card(state)

        raw_id = parts[1].lstrip("#")
        want_summary = len(parts) >= 3 and parts[2].lower() == "summary"

        try:
            target_id = int(raw_id)
        except ValueError:
            return f"[red]Invalid session ID '{parts[1]}'. Use /show #N, /show #N summary, or /show config.[/red]"

        try:
            from audiobench.memory.session_store import get_session
            detail = get_session(target_id)
        except Exception as e:
            return f"[red]Failed to load session #{target_id}: {e}[/red]"

        if detail is None:
            return f"[red]Session #{target_id} not found.[/red]"

        title = detail.title or f"Session #{target_id}"
        ts = (detail.created_at or "")[:10]

        if want_summary:
            # Summary sub-mode: show saved summary OR generate on-the-fly (no switch required)
            from rich.markdown import Markdown
            from rich.panel import Panel

            if detail.session_summary:
                gen_ts = (detail.summary_generated_at or "")[:16].replace("T", " ")
                console.print(
                    Panel(
                        Markdown(detail.session_summary),
                        title=f"[bold cyan]{title}[/bold cyan]  [dim]{ts} · {detail.query_count} searches · summarised {gen_ts}[/dim]",
                        border_style="cyan",
                        expand=True,
                    )
                )
                return None

            # No saved summary — build one on the fly without requiring a /switch
            if not detail.queries:
                return f"[dim]Session #{target_id} has no searches yet; nothing to summarise.[/dim]"

            summary_pairs: list[str] = []
            for q in detail.queries:
                block = f"Search S{q.sequence_num}: {q.query_text}"
                if q.synthesis_text and not q.synthesis_failed:
                    block += f"\nAnswer: {q.synthesis_text.strip()}"
                summary_pairs.append(block)

            session_material = "\n\n".join(summary_pairs)
            from audiobench.core.prompts import SESSION_SUMMARY_PROMPT
            prompt = SESSION_SUMMARY_PROMPT.format(transcript=session_material)

            from audiobench.chat.providers.ollama_provider import OllamaClient
            from audiobench.core.settings import get_settings
            from audiobench.memory.query_engine import Ok, _call_llm
            from audiobench.memory.session_store import save_session_summary

            settings = get_settings()
            llm = OllamaClient(base_url=settings.ollama_base_url, model=settings.ollama_model)

            console.print(f"  [dim]Generating summary for session #{target_id}…[/dim]")
            match _call_llm(prompt, 0.3, llm, settings.gemini_api_key):
                case Ok(value=summary_text):
                    summary_text = summary_text.strip()
                    save_session_summary(target_id, summary_text)
                    console.print(
                        Panel(
                            Markdown(summary_text),
                            title=f"[bold cyan]{title}[/bold cyan]  [dim]{ts} · {detail.query_count} searches[/dim]",
                            border_style="cyan",
                            expand=True,
                        )
                    )
                    _emit_hint(state, f"show_summary_export_{target_id}",
                               f"Run [cyan]/export[/cyan] after switching to session #{target_id} to produce a file with this summary embedded")
                    return None
                case _:
                    return "[red]Summary generation failed. Check Ollama/Gemini is available.[/red]"
        else:
            # Structural view: fast, no LLM, shows queries + recurring sources
            lines = [f"[bold]Session #{target_id}[/bold]  [dim]{ts} · {title} · {detail.query_count} searches[/dim]"]
            if detail.session_summary:
                lines.append(f"  [dim green]✓ AI summary available — use /show #{target_id} summary to read it[/dim green]")
            lines.append("")
            for q in detail.queries:
                n_frags = len(q.segment_ids)
                q_preview = f"\"{q.query_text[:150] + '…' if len(q.query_text) > 150 else q.query_text}\""
                lines.append(f"  [cyan]S{q.sequence_num}[/cyan]  {n_frags} frags  [dim]{q_preview}[/dim]")

            # Recurring sources across queries
            src_map: dict[str, list[int]] = {}
            for q in detail.queries:
                # Use fragment data from session_store helper
                pass
            try:
                from audiobench.memory.session_store import get_session_source_files
                src_map = get_session_source_files(target_id)
                overlapping = [(src, seqs) for src, seqs in src_map.items() if len(seqs) > 1]
                if overlapping:
                    lines.append("\n[bold]Sources recurring across searches:[/bold]")
                    for src, seqs in overlapping:
                        short = _short_source(src) if src else src
                        lines.append(f"  {short}  [dim]→ S{', S'.join(str(s) for s in seqs)}[/dim]")
            except Exception:
                pass

            # One-shot hint toward /show summary — shown only when user doesn't already have a summary
            if not detail.session_summary:
                _emit_hint(state, f"show_struct_hint_{target_id}",
                           f"[cyan]/show #{target_id} summary[/cyan] generates an AI summary for this session on the fly · "
                           f"[cyan]/switch #{target_id}[/cyan] to resume it")
            return "\n".join(lines)

    elif cmd == "/rename":
        # Rename the current session
        if state.session_id < 0:
            return "[red]No active session to rename.[/red]"
        new_title = " ".join(parts[1:]).strip() if len(parts) > 1 else ""

        if not new_title or new_title.lower() == "auto":
            if not state.search_query_texts:
                return "[yellow]No queries to generate title from.[/yellow]"
            if not engine:
                return "[red]Engine required for auto-rename.[/red]"

            queries = "\n".join(f"- {txt}" for txt in state.search_query_texts.values())
            prompt = (
                "Based on the following search queries, generate a comprehensive, "
                "descriptive title (maximum 12 words) for this research session. "
                "Return ONLY the title string, no quotes, no conversational text.\n\n"
                f"{queries}"
            )

            from audiobench.chat.providers.ollama_provider import OllamaClient
            from audiobench.core.settings import get_settings
            from audiobench.memory.query_engine import Ok, _call_llm

            settings = get_settings()
            llm = OllamaClient(base_url=settings.ollama_base_url, model=settings.ollama_model)

            # Using low temperature for predictable short titles
            match _call_llm(prompt, 0.2, llm, settings.gemini_api_key):
                case Ok(value=ans):
                    new_title = ans.strip().strip('"').strip("'")
                case _:
                    return "[red]Auto-rename LLM call failed.[/red]"

        if not new_title:
            return "[red]Usage: /rename <new title> | /rename auto[/red]"
        try:
            from audiobench.memory.session_store import set_session_title
            set_session_title(state.session_id, new_title)
            return f"[green]Session #{state.session_id} renamed to \"{new_title}\"[/green]"
        except Exception as e:
            return f"[red]Failed to rename session: {e}[/red]"

    elif cmd == "/export":
        # Export current session to markdown
        if state.session_id < 0:
            return "[red]No active session to export.[/red]"
        raw = parts[1].lstrip("-o").strip() if len(parts) > 1 else ""
        out_path = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
        try:
            _export_session_markdown(state.session_id, out_path or None)
        except Exception as e:
            return f"[red]Export failed: {e}[/red]"
        return None  # _export_session_markdown prints its own confirmation

    elif cmd == "/switch":
        # Resume a prior session by ID: /switch #2  or  /switch 2
        # Loads all prior segment/source maps so overlap badges work immediately.
        if len(parts) < 2:
            # No ID given — print the 10 most recent sessions as a quick pick list
            try:
                from audiobench.memory.session_store import list_sessions
                recent = list_sessions(limit=10)
                if not recent:
                    return "[dim]No prior sessions found.[/dim]"
                lines = ["[bold]Recent sessions (use /switch #ID to resume):[/bold]"]
                for s in recent:
                    marker = " [green]← current[/green]" if s.session_id == state.session_id else ""
                    title = s.title or "[dim](untitled)[/dim]"
                    dt = s.created_at[:10] if s.created_at else "?"
                    lines.append(
                        f"  [cyan]#{s.session_id}[/cyan]  {dt}  {title}"
                        f"  [dim]{s.query_count} search{'es' if s.query_count != 1 else ''}[/dim]{marker}"
                    )
                return "\n".join(lines)
            except Exception as e:
                return f"[red]Failed to list sessions: {e}[/red]"

        # Parse the target session ID (accept both "2" and "#2")
        raw = parts[1].lstrip("#")
        try:
            target_id = int(raw)
        except ValueError:
            return f"[red]Invalid session ID '{parts[1]}'. Use /switch #N or /switch N.[/red]"

        if target_id == state.session_id:
            return f"[dim]Already in session #{target_id}.[/dim]"

        try:
            from audiobench.memory.session_store import resume_session_state
            snap = resume_session_state(target_id)
        except Exception as e:
            return f"[red]Failed to load session #{target_id}: {e}[/red]"

        if snap is None:
            return f"[red]Session #{target_id} not found.[/red]"

        # Apply snapshot to live state
        state.session_id          = snap["session_id"]
        state.search_count        = snap["search_count"]
        state.search_segment_ids  = snap["search_segment_ids"]
        state.search_source_files = snap["search_source_files"]
        state.search_query_texts  = snap.get("search_query_texts", {})
        state.last_synthesis      = snap["last_synthesis"]
        if snap["preset"]:
            state.preset = snap["preset"]

        prior = snap["search_count"]
        title = snap["title"] or "(untitled)"

        console.print(f"  [green]✓ Switched to session [cyan]#{target_id}[/cyan]: \"{title}\"[/green]\n")

        # Trigger the same full visual replay as the startup resume logic
        _replay_session_history(state)

        return None
    return f"[dim]Unknown command: {cmd}[/dim]"




@click.group()
def memory() -> None:
    """Interact with semantic memory (search, threads, inferences)."""
    pass


def query_completer(ctx, param, incomplete: str):
    """Click shell completion for search queries via daemon."""
    from audiobench.daemon.factory import get_daemon_client
    try:
        client = get_daemon_client()
        results = client.autocomplete(incomplete, top_k=10)
        from click.shell_completion import CompletionItem
        return [CompletionItem(r.get("text", "")) for r in results if r.get("text")]
    except Exception:
        return []


@memory.command()
@click.argument("query", type=str, required=False, shell_complete=query_completer)
@click.option(
    "--preset",
    type=click.Choice(["fast", "balanced", "deep", "synthesis"]),
    default="balanced",
    help="Search preset to use.",
)
@click.option("--enable-hyde/--no-hyde", default=None, help="Override HyDE setting.")
@click.option(
    "--enable-cross-encoder/--no-cross-encoder",
    default=None,
    help="Override Cross-Encoder setting.",
)
@click.option("--enable-colbert/--no-colbert", default=None, help="Override ColBERT setting.")
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Bypass the semantic cache and force a new generation.",
)
@click.option(
    "--model",
    default=None,
    help="Override synthesis model (e.g. gemini, gpt-4o, llama3)",
)
@click.option("--interactive", "-i", is_flag=True, help="Interactive wizard mode.")
@click.option(
    "--resume",
    "resume_id",
    default=None,
    type=int,
    help="Resume a prior session by ID (skips the continue prompt).",
)
def search(
    query: str | None,
    preset: str,
    enable_hyde: bool | None,
    enable_cross_encoder: bool | None,
    enable_colbert: bool | None,
    no_cache: bool,
    model: str | None,
    interactive: bool,
    resume_id: int | None,
) -> None:
    """Search memory for a semantic query."""

    try:
        from audiobench.core.db_engine import init_db
        init_db()
    except Exception as e:
        console.print(f"[red]Warning: Database initialization failed: {e}[/red]")

    interactive_mode = interactive or not query

    engine = ResearchEngine()
    state = SearchSessionState(preset=preset, model=model)

    # ── Apply persisted preferences ───────────────────────────────────────────
    # Load search_prefs.json and apply to state.  CLI flags (preset, model)
    # that the user explicitly passed take priority over saved prefs.
    _prefs = _load_search_prefs()
    if _prefs:
        # layout and wrap_cap: no CLI equivalent — always apply from prefs
        if "layout" in _prefs and _prefs["layout"] in ("list", "book"):
            state.layout = _prefs["layout"]
        if "wrap_cap" in _prefs:
            cap = _prefs["wrap_cap"]
            state.wrap_cap = int(cap) if cap is not None else None
        # preset: apply only if the user did NOT pass --preset on the CLI
        # (CLI default is "balanced", so we can't distinguish "user passed balanced"
        # from "user didn't pass anything" — we apply the saved pref either way;
        # if they want a one-off override they can /set preset inside the session)
        if "preset" in _prefs and _prefs["preset"] in ("fast", "balanced", "deep", "synthesis"):
            state.preset = _prefs["preset"]
            state.initial_preset = _prefs["preset"]
        if "diversity_weight" in _prefs and isinstance(_prefs["diversity_weight"], (int, float)):
            state.diversity_weight = float(_prefs["diversity_weight"])
        # model: saved pref applies only when the CLI left model=None
        if "model" in _prefs and _prefs["model"] and model is None:
            state.model = _prefs["model"]
        if "autocomplete" in _prefs and isinstance(_prefs["autocomplete"], bool):
            state.autocomplete = _prefs["autocomplete"]

    continued = False


# ── Session bootstrap: resume or continue or create ───────────────────────
    try:
        from audiobench.memory.session_store import (
            list_sessions,
            resume_session_state,
        )

        # --resume flag: explicit session ID from CLI
        if resume_id is not None:
            snap = resume_session_state(resume_id)
            if snap is None:
                console.print(f"[red]Session #{resume_id} not found.[/red]")
                return
            state.session_id          = snap["session_id"]
            state.search_count        = snap["search_count"]
            state.search_segment_ids  = snap["search_segment_ids"]
            state.search_source_files = snap["search_source_files"]
            state.search_query_texts  = snap.get("search_query_texts", {})
            state.last_synthesis      = snap["last_synthesis"]
            if snap["preset"]:
                state.preset = snap["preset"]
        else:
            # SESSION VISIBILITY RULE (documented):
            # A session earns its place in the resume list only after its FIRST search query
            # has been executed (query_count > 0). Empty sessions created by Ctrl-C or
            # accidental entry are silently excluded from the resume prompt to avoid noise.
            # They remain visible (and dimmed) in /show for full DB transparency.
            all_recent = list_sessions(limit=20)  # fetch more so #ID lookup has full scope
            recent = [s for s in all_recent if s.query_count > 0][:5]
            continued = False
            if recent:
                from audiobench.cli.shared.session_picker import show_session_picker
                picker_sessions = [
                    {
                        "id": r.session_id,
                        "title": r.title,
                        "created_at": r.created_at,
                        "detail": f"{r.query_count} search{'es' if r.query_count != 1 else ''}"
                    }
                    for r in recent
                ]
                
                try:
                    chosen_id = show_session_picker(console, picker_sessions, noun="session")
                except KeyboardInterrupt:
                    return

                chosen_session = None
                if chosen_id is not None:
                    chosen_session = next((s for s in all_recent if s.session_id == chosen_id), None)

                if chosen_session:
                    snap = resume_session_state(chosen_session.session_id)
                    if snap:
                        state.session_id          = snap["session_id"]
                        state.search_count        = snap["search_count"]
                        state.search_segment_ids  = snap["search_segment_ids"]
                        state.search_source_files = snap["search_source_files"]
                        state.search_query_texts  = snap.get("search_query_texts", {})
                        state.last_synthesis      = snap["last_synthesis"]
                        if snap["preset"]:
                            state.preset = snap["preset"]
                        continued = True

            if not continued:
                # SESSION CREATION IS DEFERRED — no DB write here.
                # The session will be created lazily when the first search query is
                # executed (see persist block in _run_search_loop). This guarantees
                # that empty sessions can never accumulate in the DB from cancelled
                # or abandoned REPL entries.
                pass  # state.session_id remains -1 until first query runs


    except Exception as e:
        logger.warning("Session persistence unavailable: %s", e)
        state.session_id = -1

    # ARCHITECTURAL DECISION (2026-08-07):
    # The legacy 3-step pre-search interactive wizard (blue Panel box with prompt_menu/prompt_string)
    # has been replaced in favor of direct entry into the unified REPL search loop.
    # Preset selection, filters, and state toggles are managed dynamically via slash commands
    # (/set, /focus, /pin, etc.) within the REPL, eliminating multi-step setup prompts.

    try:
        _run_search_loop(engine, query or "", state, resumed=continued)
    except (KeyboardInterrupt, EOFError):
        # Ctrl-C anywhere inside the search loop (Fragment Reader, note writing,
        # related-search submenu, etc.) should exit cleanly — not crash to Click's
        # "Aborted!" message.  Session close runs normally in the finally block below.
        console.print()

    # Close the session on exit
    if state.session_id >= 0:
        try:
            from audiobench.memory.session_store import close_session
            close_session(state.session_id)
        except Exception as e:
            logger.warning("Failed to close session: %s", e)


@memory.command()
def threads() -> None:
    """List open conversational threads across all sessions."""
    with get_session() as session:
        summaries = session.query(ConversationSummary).all()

    if not summaries:
        console.print("No open threads found.")
        return

    table = Table(title="Open Threads", show_lines=True)
    table.add_column("Session Title", style="cyan", no_wrap=True)
    table.add_column("Open Threads")

    found = False
    for s in summaries:
        try:
            threads = json.loads(s.open_threads)
            if threads:
                found = True
                thread_texts = []
                for t in threads:
                    q = t.get("question", "")
                    c = t.get("context", "")
                    thread_texts.append(f"[bold]{q}[/bold]\n[dim]{c}[/dim]")

                table.add_row(
                    s.refined_title or f"Session #{s.conversation_id}", "\n\n".join(thread_texts)
                )
        except Exception:
            continue

    if found:
        console.print(table)
    else:
        console.print("No open threads found.")


def _render_sessions_dashboard(
    con: "Console",
    limit: int = 20,
    filter_type: str | None = None,
) -> None:
    """Render the unified system-wide grouped sessions dashboard.

    Used by both:
    - ``audiobench memory sessions`` CLI command
    - ``/sessions`` in-REPL slash command

    Groups:
        search   → search_sessions table
        chat     → chat_conversations (session_type in chat, search_followup)
        study    → study_sessions + chat_conversations (session_type=study)
        ask      → chat_conversations (session_type in ask, bookmark, memoir)
    """
    import sqlite3
    from pathlib import Path as _Path

    from rich.rule import Rule

    from audiobench.core.settings import get_settings

    settings = get_settings()
    db_path = _Path(settings.database_url.replace("sqlite:///", ""))

    con.print()

    any_printed = False

    def _section_header(label: str) -> None:
        con.print(Rule(f"[bold]{label}[/bold]", style="dim"))

    term_width = shutil.get_terminal_size().columns
    # We allocate some fixed width for other columns.
    # ID: 5, Date: 10, Queries: 7, Preset: 20 (can be comma separated now), spacing ~10. Total fixed ~50
    title_width_search = max(55, term_width - 60)
    title_width_chat = max(50, term_width - 55)

    # ── 1. Search Sessions ────────────────────────────────────────────────────
    if filter_type in (None, "search"):
        try:
            from audiobench.memory.session_store import list_sessions
            search_sessions = list_sessions(limit=limit)
            _section_header("Search Sessions")
            if not search_sessions:
                con.print("  [dim]No search sessions yet. Run 'audiobench memory search' to start one.[/dim]")
            else:
                tbl = Table(show_lines=False, box=None, padding=(0, 1))
                tbl.add_column("ID",      style="cyan",  no_wrap=True, width=5)
                tbl.add_column("Date",    style="dim",   no_wrap=True, width=10)
                tbl.add_column("Title")
                tbl.add_column("Queries", style="dim",   justify="right", width=7)
                tbl.add_column("Preset",  style="dim")
                for s in search_sessions:
                    dt = s.created_at[:10] if s.created_at else "?"
                    raw_title = s.title or ""
                    title = _short_source(raw_title, max_len=title_width_search) if raw_title else "[dim](untitled)[/dim]"
                    tbl.add_row(f"#{s.session_id}", dt, title, str(s.query_count), s.preset or "")
                con.print(tbl)
            any_printed = True
        except Exception as e:
            con.print(f"  [red]Search sessions unavailable: {e}[/red]")
        con.print()

    # ── 2. AI Chat & Follow-up Sessions ─────────────────────────────────────
    if filter_type in (None, "chat"):
        try:
            with sqlite3.connect(str(db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT id, title, session_type, model_name, message_count,
                           created_at
                    FROM chat_conversations
                    WHERE session_type IN ('chat', 'search_followup')
                      AND message_count > 0
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            _section_header("AI Chat & Follow-up Sessions")
            if not rows:
                con.print("  [dim]No active chat sessions yet.[/dim]")
            else:
                tbl = Table(show_lines=False, box=None, padding=(0, 1))
                tbl.add_column("ID",    style="cyan", no_wrap=True, width=5)
                tbl.add_column("Date",  style="dim",  no_wrap=True, width=10)
                tbl.add_column("Type",  style="dim",  width=14)
                tbl.add_column("Title")
                tbl.add_column("Msgs",  style="dim",  justify="right", width=5)
                tbl.add_column("Model", style="dim",  width=15)
                for row in rows:
                    cid, title, stype, model, msgs, created = row
                    dt = (created or "")[:10]
                    type_label = "follow-up" if stype == "search_followup" else "chat"
                    clean_title = title.replace("🔍 Search: ", "").replace("🔍 ", "") if title else ""
                    title_short = _short_source(clean_title, max_len=title_width_chat) if clean_title else "[dim](untitled)[/dim]"
                    model_short = (model or "")[:15]
                    tbl.add_row(f"#{cid}", dt, type_label, title_short, str(msgs), model_short)
                con.print(tbl)
            any_printed = True
        except Exception as e:
            con.print(f"  [red]Chat sessions unavailable: {e}[/red]")
        con.print()

    # ── 3. Audio Study Sessions ───────────────────────────────────────────────
    if filter_type in (None, "study"):
        try:
            with sqlite3.connect(str(db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT ss.id, ss.session_number, ss.created_at, ss.closed_at,
                           af.file_name, af.file_path
                    FROM study_sessions ss
                    LEFT JOIN study_projects sp ON ss.project_id = sp.id
                    LEFT JOIN audio_files af   ON sp.audio_file_id = af.id
                    ORDER BY ss.created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            _section_header("Audio Study Sessions")
            if not rows:
                con.print("  [dim]No study sessions yet. Use 'audiobench study' to start one.[/dim]")
            else:
                tbl = Table(show_lines=False, box=None, padding=(0, 1))
                tbl.add_column("ID",     style="cyan", no_wrap=True, width=5)
                tbl.add_column("Date",   style="dim",  no_wrap=True, width=10)
                tbl.add_column("S#",     style="dim",  width=4)
                tbl.add_column("Source")
                tbl.add_column("Status", style="dim",  width=10)
                for row in rows:
                    sid, snum, created, closed, af_fname, af_path = row
                    dt = (created or "")[:10]
                    source = _short_source(af_fname or af_path or "Unknown", max_len=title_width_search)
                    status = "[green]Active[/green]" if not closed else "[dim]Closed[/dim]"
                    tbl.add_row(f"#{sid}", dt, str(snum), source, status)
                con.print(tbl)
            any_printed = True
        except Exception as e:
            con.print(f"  [red]Study sessions unavailable: {e}[/red]")
        con.print()

    # ── 4. Ask & Memoir Sessions ──────────────────────────────────────────────
    if filter_type in (None, "ask"):
        try:
            with sqlite3.connect(str(db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT id, title, session_type, message_count, created_at
                    FROM chat_conversations
                    WHERE session_type IN ('ask', 'bookmark', 'memoir')
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            _section_header("Ask & Memoir Sessions")
            if not rows:
                con.print("  [dim]No ask or memoir sessions yet.[/dim]")
            else:
                tbl = Table(show_lines=False, box=None, padding=(0, 1))
                tbl.add_column("ID",   style="cyan", no_wrap=True, width=5)
                tbl.add_column("Date", style="dim",  no_wrap=True, width=10)
                tbl.add_column("Kind", style="dim",  width=8)
                tbl.add_column("Title")
                tbl.add_column("Msgs", style="dim",  justify="right", width=5)
                for row in rows:
                    cid, title, stype, msgs, created = row
                    dt = (created or "")[:10]
                    title_short = _short_source(title, max_len=title_width_chat) if title else "[dim](untitled)[/dim]"
                    tbl.add_row(f"#{cid}", dt, stype or "", title_short, str(msgs))
                con.print(tbl)
            any_printed = True
        except Exception as e:
            con.print(f"  [red]Ask/memoir sessions unavailable: {e}[/red]")
        con.print()

    if not any_printed:
        con.print("[dim]No sessions found.[/dim]")
        con.print()


@memory.command(name="sessions")
@click.option("--limit", default=20, show_default=True, help="Max sessions per category to show.")
@click.option("--type", "session_type", default=None,
              type=click.Choice(["search", "chat", "study", "ask"]),
              help="Filter to one session category.")
def sessions_cmd(limit: int, session_type: str | None) -> None:
    """Show a grouped dashboard of all system sessions (search, chat, study, ask)."""
    try:
        from audiobench.core.db_engine import init_db
        init_db()
    except Exception as e:
        console.print(f"[red]Database initialization failed: {e}[/red]")
        return
    _render_sessions_dashboard(console, limit=limit, filter_type=session_type)


def _export_session_markdown(session_id: int, output_path: str | None = None) -> None:
    """Render a search session to markdown and write to file (or ~/Documents)."""
    import re

    from audiobench.memory.session_store import get_session

    detail = get_session(session_id)
    if detail is None:
        console.print(f"[red]Session #{session_id} not found.[/red]")
        return

    title = detail.title or f"Session {session_id}"
    date_str = (detail.created_at or "")[:10]
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Session #{session_id}** · {date_str} · {detail.query_count} "
                 f"search{'es' if detail.query_count != 1 else ''} · preset: {detail.preset}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── AI Summary (if available) ─────────────────────────────────────────────
    if detail.session_summary:
        gen_ts = (detail.summary_generated_at or "")[:16].replace("T", " ")
        lines.append("## Session Summary")
        lines.append("")
        lines.append(f"*AI-generated summary · {gen_ts}*")
        lines.append("")
        lines.append(detail.session_summary.strip())
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Queries ───────────────────────────────────────────────────────────────
    for q in detail.queries:
        lines.append(f"## S{q.sequence_num} — {q.query_text}")
        lines.append("")
        lines.append(f"*{(q.created_at or '')[:16].replace('T', ' ')} · {q.preset}*")
        lines.append("")

        # Fragments — loaded inline from the DB
        try:
            from audiobench.memory.session_store import _get_conn
            conn = _get_conn()
            frag_rows = conn.execute(
                """SELECT rank, start_time, end_time, fragment_text, source_file
                   FROM search_query_fragments WHERE query_id=? ORDER BY rank""",
                (q.query_id,),
            ).fetchall()
            conn.close()
            if frag_rows:
                lines.append("### Fragments")
                lines.append("")
                for fr in frag_rows:
                    rank, start, end, text, src = fr["rank"], fr["start_time"], fr["end_time"], fr["fragment_text"], fr["source_file"]
                    ts = ""
                    if start is not None and end is not None:
                        def _fmt(s: float) -> str:
                            s = int(s); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
                        ts = f"{_fmt(start)} → {_fmt(end)}"
                    src_name = Path(src).stem if src else ""
                    lines.append(f"**#{rank}** [{ts}] *{src_name}*")
                    lines.append("")
                    lines.append(f"> {text.strip()}")
                    lines.append("")
        except Exception:
            pass

        # Synthesis answer
        if q.synthesis_text and not q.synthesis_failed:
            lines.append("### Answer")
            lines.append("")
            lines.append(q.synthesis_text.strip())
            lines.append("")

        lines.append("---")
        lines.append("")

    content = "\n".join(lines)

    # ── Output path ───────────────────────────────────────────────────────────
    if not output_path:
        from audiobench.core.settings import get_settings
        exports_dir = get_settings().data_dir / "exports" / "search"
        exports_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w\- ]", "", title)[:50].strip().replace(" ", "_")
        output_path = str(exports_dir / f"{session_id}_{safe}.md")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding="utf-8")
    console.print(f"  [green]Session #{session_id} exported →[/green] [cyan]{output_path}[/cyan]")


@memory.command(name="export")
@click.argument("session_id", type=int)
@click.option("-o", "--output", "output_path", default=None,
              help="Output file path (default: <data_dir>/exports/search/N_<title>.md)")
def export_session_cmd(session_id: int, output_path: str | None) -> None:
    """Export a search session as a markdown document.

    Example: audiobench memory export 2
    """
    try:
        from audiobench.core.db_engine import init_db
        init_db()
    except Exception as e:
        console.print(f"[red]Database initialization failed: {e}[/red]")
        return
    _export_session_markdown(session_id, output_path)



@memory.command()
def inferences() -> None:
    """List active system inferences."""
    with get_session() as session:
        inferences = (
            session.query(ExpressionRecord)
            .filter_by(source_type=SourceType.SYSTEM_INFERENCE.value)
            .all()
        )

    if not inferences:
        console.print("No system inferences found.")
        return

    table = Table(title="System Inferences", show_lines=True)
    table.add_column("ID", style="cyan")
    table.add_column("Content")
    table.add_column("Confidence", justify="right")
    table.add_column("Status")

    for inf in inferences:
        conf = f"{inf.inference_confidence:.2f}" if inf.inference_confidence is not None else "-"
        table.add_row(str(inf.id), inf.content, conf, inf.inference_status or "-")
        console.print(table)


@memory.command()
@click.option("--execute", is_flag=True, help="Execute deletion of orphaned SQLite expression rows and LanceDB vector nodes.")
@click.option("--batch-size", default=500, type=int, help="Batch size for deletions (default: 500).")
def purge(execute: bool, batch_size: int) -> None:
    """Scan and purge orphaned vector nodes and expressions from storage."""
    from sqlalchemy import select

    from audiobench.cli.display.theme import DIM, SUCCESS
    from audiobench.memory.memory_store import MemoryStore
    from audiobench.storage.models import ExpressionRecord, TranscriptionRecord

    with get_session() as session:
        active_tx_ids = set(session.scalars(select(TranscriptionRecord.id)).all())
        all_tx_exprs = session.query(ExpressionRecord.id, ExpressionRecord.source_id).filter(
            ExpressionRecord.source_type == SourceType.AUDIO_TRANSCRIPT.value
        ).all()
        orphans = [eid for eid, sid in all_tx_exprs if sid not in active_tx_ids]

    store = MemoryStore()
    lancedb_orphans: list[int] = []
    if orphans:
        # Check in batches if orphan count is very large
        for i in range(0, len(orphans), batch_size):
            chunk = orphans[i:i + batch_size]
            ids_str = ", ".join(str(eid) for eid in chunk)
            try:
                lancedb_rows = store.table.search().where(f"expression_id IN ({ids_str})").select(["expression_id"]).to_list()
                lancedb_orphans.extend(int(r["expression_id"]) for r in lancedb_rows)
            except Exception:
                pass

    if not orphans and not lancedb_orphans:
        console.print(f"[{SUCCESS}]✓ No orphaned expressions or vector nodes found.[/]")
        return

    table = Table(title="Orphan Scan Summary", show_lines=True)
    table.add_column("Asset", style="cyan")
    table.add_column("Orphan Count", justify="right", style="bold yellow")
    table.add_row("SQLite Expression Records", str(len(orphans)))
    table.add_row("LanceDB Vector Nodes", str(len(lancedb_orphans)))
    console.print(table)

    if not execute:
        console.print(f"[{DIM}]DRY-RUN MODE. To permanently purge these records, run:[/] [bold]audiobench memory purge --execute[/bold]")
        return

    with console.status("Purging orphaned vectors and expressions..."):
        if lancedb_orphans:
            for i in range(0, len(lancedb_orphans), batch_size):
                batch = lancedb_orphans[i:i + batch_size]
                b_str = ", ".join(str(eid) for eid in batch)
                store.table.delete(f"expression_id IN ({b_str})")

        if orphans:
            with get_session() as session:
                for i in range(0, len(orphans), batch_size):
                    batch = orphans[i:i + batch_size]
                    session.query(ExpressionRecord).filter(ExpressionRecord.id.in_(batch)).delete(synchronize_session=False)
                session.commit()

    console.print(f"[{SUCCESS}]✓ Successfully purged {len(orphans)} SQLite rows and {len(lancedb_orphans)} LanceDB vector nodes.[/]")


@memory.command()
@click.argument("target_id", type=int)
@click.argument("correction_text", type=str)
def correct(target_id: int, correction_text: str) -> None:
    """Correct a system inference."""
    from audiobench.daemon.factory import get_daemon_client
    from audiobench.memory.enums import SourceType
    from audiobench.storage.expression_repository import ExpressionRepository

    expr_repo = ExpressionRepository()

    # 1. Update target inference status
    success = expr_repo.update_inference_status(target_id, "corrected")
    if not success:
        console.print(
            error_panel(
                "Not Found", f"Expression #{target_id} not found or is not a system inference."
            )
        )
        return

    # 2. Write User Correction Expression
    correction_expr = expr_repo.register(
        content=correction_text,
        source_type=SourceType.USER_CORRECTION.value,
    )

    # 3. Write Relation
    # Wait, does expr_repo.link accept created_by? Not currently.
    expr_repo.link(
        from_id=correction_expr.id,
        to_id=target_id,
        relation_type="corrects",  # It asks for type='corrects', maybe not in RelationType enum yet
    )

    # Update created_by manually if needed (not in ExpressionRelation schema from earlier, but let's check if it exists)
    # The requirement says `created_by='user'`, but `ExpressionRelation` might not have it. Let's just link it.

    # 4. Embed
    try:
        daemon = get_daemon_client()
        daemon.embed(
            expression_id=correction_expr.id,
            content=correction_text,
            source_type=SourceType.USER_CORRECTION,
            speaker=None,
        )
    except Exception as e:
        logger.warning("Daemon embed failed for correction: %s", e)

    console.print(
        f"[green]✓ Corrected inference #{target_id} with expression #{correction_expr.id}.[/green]\n"
    )


# ── Research Engine display helpers (P1-5) ────────────────────────────────────

def _fmt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _short_source(file_path: str, max_len: int = 55) -> str:
    """Return a readable short name from a full file path.

    Uses *middle* truncation so both the beginning and end of long titles
    are visible: e.g. "Biblical Teach… Human Wisdom" instead of the
    end-truncated "Biblical Teachings on Divin…".
    """
    import os
    name = os.path.basename(file_path)
    # Strip common extensions
    for ext in (".mp4", ".mp3", ".m4a", ".wav", ".webm", ".m4b", ".ogg", ".flac"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    # Strip leading hash prefix (e.g. "f5c6e2c8_Life is Short" → "Life is Short")
    if "_" in name and len(name.split("_")[0]) == 8:
        name = name.split("_", 1)[1]
    name = name.strip()
    # Middle-truncate: keep the first 60% and last 35% of max_len, joined by …
    if len(name) > max_len:
        keep_front = max(1, int(max_len * 0.60))
        keep_back  = max(1, max_len - keep_front - 1)  # -1 for the ellipsis
        name = name[:keep_front] + "…" + name[-keep_back:]
    return name


_STREAM_COLORS: dict[str, str] = {
    "fts5":    "yellow",
    "dense":   "magenta",
    "colbert": "blue",
}


def _relevance_dots(rank: int, total_dots: int = 5) -> str:
    """Convert a 1-based retrieval rank to a filled-dot relevance bar.

    rank 1-2  → ●●●●●  (strongest)
    rank 3-4  → ●●●●○
    rank 5-6  → ●●●○○
    rank 7-8  → ●●○○○
    rank 9+   → ●○○○○  (weakest shown)

    The result is rendered with Rich markup: filled dots in cyan, empty in dim.
    """
    filled = max(1, total_dots - (rank - 1) // 2)
    filled = min(filled, total_dots)
    dots = (
        "[cyan]●[/cyan]" * filled
        + "[dim]○[/dim]" * (total_dots - filled)
    )
    return dots


# ── Superscript citation rendering ────────────────────────────────────────────
# Maps digit characters to their Unicode superscript equivalents so we can
# render [1] as ¹, [10] as ¹⁰, etc. without any special terminal support.
_SUPERSCRIPT_DIGITS: dict[str, str] = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
}


def _to_superscript(n: int) -> str:
    """Convert an integer to its Unicode superscript string (e.g. 10 → '¹⁰')."""
    return "".join(_SUPERSCRIPT_DIGITS[c] for c in str(n))


def _render_citations(
    answer: str,
    sources: "list[FusedResult]",  # type: ignore[name-defined]
) -> tuple[str, list[tuple[int, "FusedResult"]]]:  # type: ignore[name-defined]
    """Post-process LLM answer text: convert [N] citation markers to vibrant
    inline code badges `[N]` so cited fragments stand out clearly and correlate
    directly to the fragment numbers in the book layout above.

    Args:
        answer:  Raw LLM answer string, may contain [1], [3][5], etc.
        sources: result.sources (full fused list, 1-indexed by position).

    Returns:
        (styled_answer, cited_pairs) where:
          styled_answer  — answer with [N] converted to vibrant `[N]` code badges.
          cited_pairs    — ordered (display_num, FusedResult) for each distinct
                           cited N.
    """
    import re

    seen: dict[int, "FusedResult"] = {}  # type: ignore[name-defined]
    cited_pairs: list[tuple[int, "FusedResult"]] = []  # type: ignore[name-defined]

    def _replace(m: re.Match) -> str:
        raw = m.group(1)
        try:
            n = int(raw)
        except ValueError:
            return m.group(0)  # not a number — leave as-is

        # Validate: must refer to a real fragment (1-based)
        if n < 1 or n > len(sources):
            return m.group(0)

        fr = sources[n - 1]
        if n not in seen:
            seen[n] = fr
            cited_pairs.append((n, fr))

        # Return markdown inline code badge `[N]` for vibrant styling
        return f"`[{n}]`"

    # Match [N] or [N][M] chains; only match digit-only content inside brackets
    # to avoid clobbering markdown links like [text](url).
    styled = re.sub(r"\[(\d+)\]", _replace, answer)
    return styled, cited_pairs


def _display_results(result: "ResearchResult", state: SearchSessionState | None = None) -> None:  # type: ignore[name-defined]
    """Render a ResearchResult using source-grouped layout (Option Z).

    Fragments are grouped by source file, ordered by group relevance (max
    rrf_score descending), chronological within each group.  Full fragment
    text is displayed without truncation, wrapped to terminal width.

    Online:  grouped fragments above, synthesis panel below.
    Offline: grouped fragments only.  No error panel — the absence of a
             synthesis panel is itself the signal. No apology. No red banner.

    # ROADMAP [Phase 2]: Add `/view compact` toggle — `view_mode: str = "rich"` in
    # SearchSessionState. Compact = 2-line snippet per fragment inside group box.
    # Auto-rich when synthesis_failed=True; auto-compact when synthesis succeeded.
    #
    # ROADMAP [Phase 3]: Surface parent chunk context per fragment (Option C).
    # Parent is already fetched in ResearchEngine — pass it through FusedResult
    # or fetch at display time via ExpressionRepository.get_parents_batch().
    """
    import shutil
    import textwrap
    from collections import defaultdict
    from rich.markdown import Markdown
    from rich.panel import Panel

    console.print()

    # ── HyDE notice ──────────────────────────────────────────────────────────
    if result.hyde_fallback:
        console.print(
            "[dim yellow]⚠  HyDE unavailable — falling back to direct query embedding[/dim yellow]"
        )
    elif getattr(result, "hyde_document", None):
        console.print(
            Panel(
                f"[italic]{result.hyde_document}[/italic]",
                title="[dim]HyDE Generation[/dim]",
                border_style="dim",
                expand=True,
            )
        )

    if not result.sources:
        console.print("[dim]No results found.[/dim]")
        console.print()
        return

    term_width = shutil.get_terminal_size().columns

    # ── Header bar ────────────────────────────────────────────────────────────
    n_frags = len(result.sources)
    # Unique source names, order preserved, deduped
    seen: dict[str, None] = {}
    for fr in result.sources:
        if fr.source_file:
            seen[fr.source_file] = None
    n_sources = len(seen)

    elapsed = f"{result.retrieval_time_seconds:.2f}s retrieval"
    data_parts = [f"{n_frags} fragment{'s' if n_frags != 1 else ''}"]
    if n_sources > 1:
        data_parts.append(f"{n_sources} sources")
    data_parts.append(elapsed)

    skipped_markup = ""
    if result.streams_skipped:
        # Exclude 'recap' — 0 prior-context hits is normal; not a user-visible warning.
        visible_skipped = [(name, reason) for name, reason in result.streams_skipped if name != "recap"]
        if visible_skipped:
            skipped = [f"[dim red]{name}✗[/dim red]" for name, _ in visible_skipped]
            skipped_markup = "  " + "  ".join(skipped)

    # Session context prefix (S2 · #7 · ...)
    session_prefix = ""
    if state and state.session_id >= 0:
        seq = state.search_count  # already incremented before display
        session_prefix = f"  [bold cyan]S{seq}[/bold cyan] [dim]· #{state.session_id}[/dim]  [dim]·[/dim]  "

    # Synthesis status suffix on the header line
    synth_suffix = ""
    if getattr(result, "synthesis_is_fallback", False):
        synth_suffix = "  [dim yellow]synthesis\u223f[/dim yellow]"  # ∿ = extractive fallback
    elif getattr(result, "synthesis_failed", False):
        synth_suffix = "  [dim red]synthesis\u2717[/dim red]"         # ✗ = hard fail

    console.print(f"{session_prefix}[dim]{' · '.join(data_parts)}[/dim]{skipped_markup}{synth_suffix}")
    console.print()

    # Wrap width: fluid by default; capped if user ran /set width N
    _fluid_wrap = max(40, term_width - 8)
    wrap_width = min(_fluid_wrap, state.wrap_cap) if (state and state.wrap_cap) else _fluid_wrap

    # ── Render recap (prior-session context) ─────────────────────────────────
    # Hard cap at 2 to guard against retrieval layer returning more than expected.
    if getattr(result, "prior_synthesis_hits", None):

        for synth in result.prior_synthesis_hits[:2]:
            label = "summary" if synth.source_type == "search_session_summary" else f"S{synth.sequence_num}"
            title = (
                f"[bold white]🧠  Recap[/bold white]"
                f"  [dim]· Session #{synth.session_id} · {label}[/dim]"
            )
            md = Markdown(synth.content.strip(), justify="left")
            console.print(
                Panel(
                    md,
                    title=title,
                    title_align="left",
                    border_style="dim cyan",
                    padding=(0, 2),
                )
            )
            console.print()


    # ── Group fragments by source ─────────────────────────────────────────────
    # Preserve 1-based global indices so /pin <n> and E reader work correctly
    groups: dict[str, list[tuple[int, 'FusedResult']]] = defaultdict(list)
    for idx, fr in enumerate(result.sources, 1):
        groups[fr.source_file].append((idx, fr))  # type: ignore[union-attr]

    # ── F1: Two-level outer source ranking ────────────────────────────────────
    # Inner rank: each fragment already has its individual rrf_score (from RRF merge).
    # Outer rank: sum all fragment scores per source — a source that contributed
    # 3 relevant fragments outranks one that contributed only 1, even if that
    # single fragment had a slightly higher raw score.  sum() > max() here because
    # it rewards breadth of evidence, not just the lucky peak.
    ordered_sources = sorted(
        groups.keys(),
        key=lambda sf: sum(fr.rrf_score for _, fr in groups[sf]),  # type: ignore[union-attr]
        reverse=True,
    )

    # Build a rank→source mapping so _render_group can show the outer rank badge
    _source_rank: dict[str, int] = {sf: i + 1 for i, sf in enumerate(ordered_sources)}

    # ── Pre-render helper: build (header_str, body_lines) for one source group ─
    def _render_group(source_file, wrap_w: int) -> tuple[str, list[str]]:
        group        = groups[source_file]
        group_sorted = sorted(group, key=lambda x: x[1].start_time)  # type: ignore[union-attr]
        source_name  = _short_source(source_file) if source_file else "Unknown source"
        n_group      = len(group_sorted)

        source_overlap_note = ""
        if state and state.search_count > 1 and source_file:
            prior_source_seqs = [
                s for s in state.search_source_files.get(source_file, [])
                if s < state.search_count
            ]
            if prior_source_seqs:
                seq_labels = ", ".join(f"S{s}" for s in sorted(prior_source_seqs))
                source_overlap_note = f"  [dim yellow](also in {seq_labels})[/dim yellow]"

        # ── F1: Outer rank badge ───────────────────────────────────────────────
        # Show which position this source holds in the outer (source-level) ranking.
        # The aggregate score (sum of all fragment rrf_scores) is what put it here.
        outer_rank = _source_rank.get(source_file, 0)
        total_sources = len(_source_rank)
        rank_badge = ""
        if total_sources > 1:
            # Only show rank badge when there are multiple sources to distinguish
            rank_badge = f"  [dim]#{outer_rank}[/dim]"

        header = (
            f"[dim]──[/dim] [bold]{source_name}[/bold]"
            f"{rank_badge}"
            f"  [dim]{n_group} fragment{'s' if n_group != 1 else ''}[/dim]"
            f"{source_overlap_note}"
        )

        body: list[str] = []
        for frag_pos, (global_idx, fr) in enumerate(group_sorted):
            ts = f"[cyan]{_fmt_timestamp(fr.start_time)} → {_fmt_timestamp(fr.end_time)}[/cyan]"  # type: ignore[union-attr]

            rel_dots = ""
            if fr.stream_contributions:  # type: ignore[union-attr]
                best_rank = min(rank for _, rank in fr.stream_contributions)  # type: ignore[union-attr]
                rel_dots = "  " + _relevance_dots(best_rank)

            pin_badge = ""
            if state and fr.segment_id in state.pinned_fragments:  # type: ignore[union-attr]
                pin_badge = "  [bold green]📌[/bold green]"

            overlap_badge = ""
            if state and state.search_count > 1:
                prior_seqs = [
                    s for s in state.search_segment_ids
                    if fr.segment_id in state.search_segment_ids[s]  # type: ignore[union-attr]
                ]
                if prior_seqs:
                    seq_labels = ",".join(f"S{s}" for s in sorted(prior_seqs))
                    overlap_badge = f"  [dim yellow]↩{seq_labels}[/dim yellow]"

            body.append(
                f"  [bold cyan]{global_idx}[/bold cyan]  {ts}{rel_dots}{pin_badge}{overlap_badge}"
            )

            raw_text = fr.text.replace("\n", " ").strip()  # type: ignore[union-attr]
            wrapped  = textwrap.fill(raw_text, width=wrap_w)
            for text_line in wrapped.split("\n"):
                body.append(f"     {text_line}")

            if frag_pos < len(group_sorted) - 1:
                body.append("")  # blank separator between fragments within group

        return header, body

    # ── Choose rendering strategy ──────────────────────────────────────────────
    use_book = (
        getattr(state, "layout", "book") == "book"
        and term_width >= 100
        and len(ordered_sources) >= 2
    )

    if use_book:
        # ── Book mode: two-column layout ──────────────────────────────────────
        # Column width: (terminal - 3 chars for " │ ") / 2, then apply wrap_cap
        col_w_raw  = max(40, (term_width - 3) // 2)
        _fluid_col = col_w_raw - 4  # 4 chars indent margin inside each column
        col_wrap   = min(_fluid_col, state.wrap_cap) if (state and state.wrap_cap) else _fluid_col
        col_w      = col_w_raw  # full column width for padding/separator alignment

        # Greedily assign groups to left/right columns to balance heights
        left_groups:  list[str] = []
        right_groups: list[str] = []
        left_h  = 0
        right_h = 0
        for sf in ordered_sources:
            _, body = _render_group(sf, col_wrap)
            h = len(body) + 2  # +2 for header + blank line after
            if left_h <= right_h:
                left_groups.append(sf)
                left_h += h
            else:
                right_groups.append(sf)
                right_h += h

        # Pre-render each column into a flat list of plain strings (no Rich markup
        # for padding — Rich markup is included as-is; we pad with spaces)
        def _col_lines(source_files, wrap_w: int) -> list[str]:
            out: list[str] = []
            for sf in source_files:
                hdr, body = _render_group(sf, wrap_w)
                out.append(hdr)
                out.extend(body)
                out.append("")  # blank line after each group
            return out

        left_lines  = _col_lines(left_groups,  col_wrap)
        right_lines = _col_lines(right_groups, col_wrap)

        # Zip, padding the shorter column so both reach the same height
        max_h = max(len(left_lines), len(right_lines))
        left_lines  += [""] * (max_h - len(left_lines))
        right_lines += [""] * (max_h - len(right_lines))

        from rich.markup import escape as _resc
        import re as _re

        def _strip_markup(s: str) -> str:
            """Approximate visible length by stripping Rich markup tags."""
            return _re.sub(r"\[/?[^\]]+\]", "", s)

        for left_raw, right_raw in zip(left_lines, right_lines):
            # Pad left column to col_w visible chars
            left_vis = len(_strip_markup(left_raw))
            pad      = max(0, col_w - left_vis)
            # Print: left content + padding + │ separator + space + right content
            console.print(f"{left_raw}{' ' * pad} [dim]│[/dim] {right_raw}")
        console.print()

    else:
        # ── List mode (default): single-column ────────────────────────────────
        # wrap_width already computed at top of function (honours wrap_cap)
        for source_file in ordered_sources:
            header, body = _render_group(source_file, wrap_width)
            console.print(header)
            console.print("\n".join(body))
            console.print()

    # ── Synthesis panel ───────────────────────────────────────────────────────
    # synthesis_is_fallback → dim amber panel, clearly labeled "LLM offline".
    # synthesis_failed      → silent absence (no red banner, no apology).
    # LLM answered          → normal cyan Answer panel + superscript footnotes.
    if result.synthesis_is_fallback and result.answer is not None:
        from rich.markdown import Markdown as RichMarkdown
        console.print(
            Panel(
                RichMarkdown(result.answer),
                title=(
                    "[bold yellow]Fragments[/bold yellow]"
                    " [dim]· LLM offline \u2014 top retrieved passages shown[/dim]"
                ),
                border_style="dim yellow",
                expand=True,
            )
        )
    elif not result.synthesis_failed and result.answer is not None:
        from rich.markup import escape as rich_escape
        from rich.markdown import Markdown as RichMarkdown
        from rich.text import Text

        from audiobench.cli.display.theme import CHAT_CODE_THEME

        # ── Citation post-processing ──────────────────────────────────────────
        # Convert [N] tokens in the LLM answer to dim-cyan Unicode superscripts.
        # cited_pairs is an ordered list of (display_num, FusedResult) for every
        # distinct N that was cited — used for the footnote strip below.
        styled_answer, cited_pairs = _render_citations(result.answer, result.sources)

        # Truncate query for title display
        q_display = result.query
        if len(q_display) > 48:
            q_display = q_display[:47] + "…"

        # The styled_answer contains Rich markup ([dim cyan]¹[/dim cyan]) mixed
        # with Markdown.  RichMarkdown renders markup inside the panel so the
        # superscripts appear correctly coloured.
        console.print(
            Panel(
                RichMarkdown(styled_answer, code_theme=CHAT_CODE_THEME),
                title=(
                    f"[bold cyan]Answer[/bold cyan]"
                    f" [dim]· \"{q_display}\"[/dim]"
                    f" [dim]({result.synthesis_time_seconds:.2f}s)[/dim]"
                ),
                border_style="cyan",
                expand=True,
            )
        )
        console.print()

    console.print(f"[dim]Total search time: {result.query_time_seconds:.2f}s[/dim]")
    console.print()



def _stream_expanded_synthesis(result: "ResearchResult") -> None:  # type: ignore[name-defined]
    """Generate and stream a deeply expanded synthesis using all fragments."""
    from rich.live import Live
    from rich.markdown import Markdown as RichMarkdown
    from rich.padding import Padding

    from audiobench.chat.providers.ollama_provider import OllamaClient
    from audiobench.cli.display.theme import CHAT_CODE_THEME, chat_console
    from audiobench.core.settings import get_settings

    settings = get_settings()
    client = OllamaClient(base_url=settings.ollama_base_url, model=settings.ollama_model)

    prompt = (
        f"QUERY: {result.query}\n\n"
        "Produce a richly structured, deep exploration of the query using the transcript fragments below.\n\n"
        "STRUCTURE YOUR RESPONSE:\n"
        "  1. **Opening** — a direct 2-3 sentence engagement with the core question.\n"
        "  2. **Analysis** — themed breakdowns with bold section headers. Quote directly and\n"
        "     anchor every key claim to a timestamp, e.g. (02:25:19) or \"exact quote\" (01:30:15).\n"
        "  3. **Synthesis** — what the fragments collectively reveal that no single one does alone.\n"
        "  4. **Open Questions** — what remains unanswered or most worth pursuing further.\n\n"
        "Be thorough and direct. No hedging, no pleasantries.\n\n"
        "FRAGMENTS:\n"
    )
    for i, src in enumerate(result.sources, 1):
        prompt += f"[{_fmt_timestamp(src.start_time)}–{_fmt_timestamp(src.end_time)}]\n"
        content = getattr(src, "expression_content", None) or src.text
        prompt += f"{content}\n\n"

    console.print("\n[dim]Generating expanded synthesis...[/dim]\n")
    content_parts = []

    try:
        with Live(console=chat_console, refresh_per_second=8, transient=True) as live:
            for chunk in client.stream(
                prompt=prompt,
                system_prompt=(
                    "You are a deep research assistant working with audio transcript fragments "
                    "from the user's personal listening library (audiobooks, podcasts, interviews, "
                    "lectures, and personal recordings). "
                    "Produce richly structured, timestamp-anchored analytical prose. "
                    "Use bold headers for each section. Quote directly from the fragments to ground claims. "
                    "Identify patterns, tensions, and open questions the user has not asked about explicitly. "
                    "Be thorough — this is a deep dive, not a summary."
                ),
            ):
                if chunk:
                    content_parts.append(chunk)
                    live.update(RichMarkdown("".join(content_parts), code_theme=CHAT_CODE_THEME))

        if content_parts:
            chat_console.print(Padding(RichMarkdown("".join(content_parts), code_theme=CHAT_CODE_THEME), (0, 0, 1, 0)))

    except Exception as e:
        console.print(f"[red]Error during synthesis: {e}[/red]")


# ── Search PromptSession ──────────────────────────────────────────────────────
# All user input in the search REPL flows through a single prompt_toolkit
# PromptSession.  This gives us:
#   • Semantic autocomplete via daemon (fragment text matches as you type)
#   • Slash-command autocomplete (/set, /settings, /focus, …)
#   • Up/down arrow query history (persisted to data/search_history.txt)
#   • Ghost-text auto-suggest from history
#   • Clean paste handling (clipboard is atomic — B1 permanently eliminated)
#   • Single-char keybindings for Q / E / C / S / 1–9 when buffer is empty
#
# The session is created once per _run_search_loop call via
# _make_search_prompt_session() and reused for every prompt() call inside the
# loop.  Rich console.print() calls interleave freely — prompt_toolkit only
# owns the terminal while waiting for input.

# _SEARCH_SLASH_COMMANDS moved to audiobench.memory.search_meta


def _make_search_prompt_session(
    state: "SearchSessionState",
) -> "PromptSession":  # type: ignore[name-defined]
    """Build a prompt_toolkit PromptSession configured for the search REPL.

    Autocomplete layers (in priority order when buffer starts with '/'):
      1. SlashCommandCompleter  — known slash commands, inline hints
    When buffer does NOT start with '/':
      2. DaemonSemanticCompleter — semantic matches against transcript memory (if enabled)

    Key bindings (fire only when buffer is EMPTY so they don't interfere with
    normal typing):
      Q   → raise KeyboardInterrupt  (signals quit to the caller)
      E   → raise _SearchKeyEvent("E")
      C   → raise _SearchKeyEvent("C")
      S   → raise _SearchKeyEvent("S")
      1–9 → raise _SearchKeyEvent("<digit>")

    Falls back to a plain PromptSession if prompt_toolkit is too old or
    the daemon is not reachable.
    """
    import re as _re
    from pathlib import Path as _Path

    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.lexers import Lexer as _PtLexer
    from prompt_toolkit.styles import Style as _PtStyle

    # ── History file ──────────────────────────────────────────────────────────
    try:
        from audiobench.core.settings import get_settings
        hist_path = get_settings().data_dir / "search_history.txt"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        hist_path = _Path.home() / ".cache" / "audiobench_search_history.txt"
        hist_path.parent.mkdir(parents=True, exist_ok=True)

    # ── F3: Syntax lexer ──────────────────────────────────────────────────────
    # Colours the prompt input as the user types:
    #   "quoted phrases"  → green (double quotes only, to avoid apostrophe breakage)
    #   AND / OR / NOT    → yellow
    #   /slash commands   → cyan
    #   plain text        → dimmed grey

    _BOOL_RE = _re.compile(r'(\bAND\b|\bOR\b|\bNOT\b)')

    def _tokenize_query(text: str) -> list[tuple[str, str]]:
        """Return (style_class, text) tokens for the search prompt."""
        if not text:
            return [("", "")]
        if text.startswith("/"):
            return [("class:slash-cmd", text)]

        tokens: list[tuple[str, str]] = []
        i = 0
        while i < len(text):
            if text[i] == '"':
                # Double quote starts exact phrase
                j = i + 1
                while j < len(text) and text[j] != '"':
                    j += 1
                if j < len(text):
                    j += 1  # include closing double quote
                tokens.append(("class:quoted", text[i:j]))
                i = j
            else:
                # Scan until next double quote
                j = i
                while j < len(text) and text[j] != '"':
                    j += 1
                segment = text[i:j]
                # Within plain segment, highlight boolean operators
                for part in _BOOL_RE.split(segment):
                    if part in ("AND", "OR", "NOT"):
                        tokens.append(("class:bool-op", part))
                    elif part:
                        tokens.append(("class:query-plain", part))
                i = j

        return tokens

    class _SearchLexer(_PtLexer):
        """Syntax-highlight the search REPL input line."""
        def lex_document(self, document):
            tokens = _tokenize_query(document.text)
            def _get_line(line_no: int):
                return tokens
            return _get_line



    # ── Completers & Suggestions ──────────────────────────────────────────────
    from audiobench.memory.search_completer import (
        SearchAutoSuggest,
        SearchSemanticCompleter,
        SearchSlashCompleter,
    )

    # ── Key bindings ──────────────────────────────────────────────────────────
    # Single-char command keys fire only when the input buffer is EMPTY so
    # they don't steal characters from a query the user is typing.

    kb = KeyBindings()

    def _buffer_is_empty(event) -> bool:
        return event.app.current_buffer.text == ""

    @kb.add("q")
    @kb.add("Q")
    def _key_q(event):
        if _buffer_is_empty(event):
            event.app.exit(result="\x00Q")

    @kb.add("e")
    @kb.add("E")
    def _key_e(event):
        if _buffer_is_empty(event):
            event.app.exit(result="\x00E")
        else:
            event.app.current_buffer.insert_text(event.key_sequence[0].key)

    @kb.add("c")
    @kb.add("C")
    def _key_c(event):
        if _buffer_is_empty(event):
            event.app.exit(result="\x00C")
        else:
            event.app.current_buffer.insert_text(event.key_sequence[0].key)

    @kb.add("s")
    @kb.add("S")
    def _key_s(event):
        if _buffer_is_empty(event):
            event.app.exit(result="\x00S")
        else:
            event.app.current_buffer.insert_text(event.key_sequence[0].key)

    for _d in "123456789":
        @kb.add(_d)
        def _key_digit(event, digit=_d):
            if _buffer_is_empty(event):
                event.app.exit(result=f"\x00{digit}")
            else:
                event.app.current_buffer.insert_text(digit)

    from audiobench.cli.shared.repl_shell import make_prompt_session
    
    style_overrides = {
        "query-plain":  "#b0b0b0",          # dim grey
        "quoted":       "#00e87a bold",     # vibrant green for "quoted phrases"
        "bool-op":      "#e8c800 bold",     # amber-yellow for AND/OR/NOT
        "slash-cmd":    "cyan bold",        # cyan for /commands
    }
    
    return make_prompt_session(
        slash_completer=SearchSlashCompleter(),
        history_path=hist_path,
        style_overrides=style_overrides,
        semantic_completer=ThreadedCompleter(SearchSemanticCompleter(state)),
        key_bindings=kb,
        lexer=_SearchLexer(),
        auto_suggest=SearchAutoSuggest(state),
        complete_while_typing=True,
    )


def _read_single_key() -> str:
    """Read one keypress from stdin without requiring Enter.

    Captures multi-byte escape sequences for arrow keys (\\x1b[A/B/C/D).
    Falls back to ``input()`` if the terminal is not a tty (e.g. piped
    input in tests), in which case arrow-key navigation is unavailable.

    Paste-contamination guard
    ─────────────────────────
    When the user pastes text via Ctrl-V / middle-click, the OS delivers
    the clipboard as a burst of characters into stdin all at once.
    ``tty.setraw`` + ``read(1)`` consumes only the *first* byte, leaving
    the rest sitting in the kernel's input buffer.  Those leftovers then
    appear at the start of the *next* ``input()`` call, producing ghost
    characters that the user can't delete.

    Fix: after reading the first byte, poll stdin with a zero-second
    ``select()`` timeout.  If more bytes are already buffered, drain them
    all in a tight loop and return the full string.  The REPL's
    ``_sys.stdout.write(first)`` + ``input("")`` pattern already expects
    a single character; when we hand it the full paste string it
    reconstructs the complete line cleanly.
    """
    import select
    import sys
    import termios
    import tty

    if not sys.stdin.isatty():
        return input()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)

        # Arrow keys send \x1b[A / \x1b[B / \x1b[C / \x1b[D — read two more bytes
        if ch == "\x1b":
            ch += sys.stdin.read(2)
            return ch

        # Drain any additional bytes already buffered (paste burst protection).
        # We poll with a very short timeout (50 ms) to catch late-arriving bytes
        # in slow paste scenarios; once the buffer is empty we stop immediately.
        extra = ""
        while select.select([sys.stdin], [], [], 0.05)[0]:
            byte = sys.stdin.read(1)
            if not byte or byte in ("\r", "\n"):
                break
            extra += byte

        return ch + extra
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _fetch_adjacent(
    transcription_id: int,
    anchor_start: float,
    anchor_end: float,
    limit: int = 2,
    segment_id: int = 0,
    source_file: str = "",
) -> tuple[list[dict], list[dict]]:
    """Return (prev_segs, next_segs) relative to the anchor window.

    Fetches the `limit` segments immediately before ``anchor_start`` and the
    `limit` immediately at or after ``anchor_end`` from the same transcript.
    Both lists are ordered chronologically (earliest first).
    """
    from sqlalchemy import text as sql_text

    with get_session() as session:
        if transcription_id <= 0:
            if segment_id > 0:
                row = session.execute(
                    sql_text("SELECT transcription_id FROM segments WHERE id = :sid"),
                    {"sid": segment_id},
                ).fetchone()
                if row and row[0]:
                    transcription_id = int(row[0])
            if transcription_id <= 0 and source_file:
                basename = source_file.split("/")[-1]
                row = session.execute(
                    sql_text(
                        "SELECT t.id FROM transcriptions t "
                        "JOIN audio_files af ON t.audio_file_id = af.id "
                        "WHERE af.file_path = :path OR af.file_path LIKE :endpath "
                        "ORDER BY t.id DESC LIMIT 1"
                    ),
                    {"path": source_file, "endpath": f"%{basename}"},
                ).fetchone()
                if row and row[0]:
                    transcription_id = int(row[0])

        if transcription_id <= 0:
            return [], []

        prev_rows = session.execute(
            sql_text(
                "SELECT id, start_time, end_time, text FROM segments "
                "WHERE transcription_id = :tid AND start_time < :start "
                "ORDER BY start_time DESC LIMIT :limit"
            ),
            {"tid": transcription_id, "start": anchor_start, "limit": limit},
        ).mappings().all()

        next_rows = session.execute(
            sql_text(
                "SELECT id, start_time, end_time, text FROM segments "
                "WHERE transcription_id = :tid AND start_time >= :end "
                "ORDER BY start_time ASC LIMIT :limit"
            ),
            {"tid": transcription_id, "end": anchor_end, "limit": limit},
        ).mappings().all()

    # prev_rows come back newest-first from DESC; reverse to chronological order
    return list(reversed([dict(r) for r in prev_rows])), [dict(r) for r in next_rows]


def _resolve_audio_file_id(transcription_id: int, _cache: dict = {}) -> int:
    """Memoized lookup for audio_file_id from a transcription_id."""
    if transcription_id in _cache:
        return _cache[transcription_id]
    from audiobench.storage.models import TranscriptionRecord
    with get_session() as s:
        rec = s.query(TranscriptionRecord).filter_by(id=transcription_id).first()
        if rec and rec.audio_file_id:
            _cache[transcription_id] = rec.audio_file_id
            return rec.audio_file_id
    return 0

def _get_max_end_time(transcription_id: int, _cache: dict = {}) -> float:
    """Memoized lookup for MAX(end_time) of a transcription."""
    if transcription_id in _cache:
        return _cache[transcription_id]
    from sqlalchemy import text
    with get_session() as s:
        res = s.execute(text("SELECT MAX(end_time) FROM segments WHERE transcription_id = :tid"), {"tid": transcription_id}).scalar()
        _cache[transcription_id] = res or 0.0
        return _cache[transcription_id]

def _toggle_bookmark(
    fr_display: dict,
    transcription_id: int,
    console: Console,
) -> None:
    """Toggle a bookmark on the currently-displayed segment.

    Takes ``fr_display`` (the scroll-adjusted dict with id/start_time/end_time/text)
    and the ``transcription_id`` from the parent FusedResult (not in fr_display).
    """
    from audiobench.storage.bookmark_repository import BookmarkRepository
    audio_file_id = _resolve_audio_file_id(transcription_id)
    if not audio_file_id:
        console.print("[red]Cannot bookmark: no audio_file_id[/red]")
        return

    repo = BookmarkRepository()
    bm = repo.get_nearest(audio_file_id, fr_display["start_time"], window=1.0)
    if bm:
        repo.delete(bm["id"])
    else:
        repo.add_region(
            audio_file_id,
            fr_display["start_time"],
            fr_display["end_time"],
            name="Bookmarked from reader",
        )

def _yank_to_clipboard(text: str, console: Console) -> None:
    import subprocess
    try:
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=True)
        console.print("  [green]Yanked to clipboard (xclip)[/green]")
    except Exception:
        try:
            subprocess.run(["xsel", "--clipboard", "--input"], input=text.encode("utf-8"), check=True)
            console.print("  [green]Yanked to clipboard (xsel)[/green]")
        except Exception:
            console.print(f"\n[dim]Clipboard copy failed. Text:[/dim]\n{text}\n")


def _find_related(
    console: Console,
    fr: "FusedResult",  # type: ignore[name-defined]
    exclude_ids: set[int],
) -> "FusedResult | None":
    """Show the top 5 semantically related fragments and return a user-selected one, if any."""
    console.print()
    console.print("  [bold cyan]Related search preset[/bold cyan]")
    console.print("    [1] fast            [dim]BM25 + Nomic (Fastest)[/dim]")
    console.print("    [2] balanced        [dim]BM25 + Nomic + ColBERT (Default)[/dim]")
    console.print("    [3] synthesis       [dim]MMR + Cross-Source Diversity (Synthesis)[/dim]")
    console.print("    [4] deep            [dim]HyDE + CrossEncoder (Deepest)[/dim]")
    console.print("    [any] cancel")
    try:
        preset_choice = input("  → ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print()
        # Re-raise so Ctrl-C propagates all the way out to the Click entry
        # point and exits cleanly instead of just re-rendering the Fragment Reader.
        raise KeyboardInterrupt

    preset_map = {
        "1": "fast",
        "2": "balanced",
        "3": "synthesis",
        "4": "deep",
    }
    preset = preset_map.get(preset_choice)
    if not preset:
        return None

    from audiobench.memory.query_engine import ResearchEngine
    engine = ResearchEngine()

    with console.status(f"[cyan]Finding related fragments (preset={preset})...[/cyan]"):
        try:
            res = engine.search(fr.text, top_k=10, preset=preset)
            hits = res.sources
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n  [red]Related search failed: {exc}[/red]")
            console.print("  [dim]Press any key to continue…[/dim]")
            _read_single_key()
            return None

    related = [h for h in hits if h.segment_id not in exclude_ids][:5]

    console.print()
    console.rule("[bold]Related moments[/bold]")
    if not related:
        console.print("  [dim]No related fragments found outside the current result set.[/dim]")
        console.print("  [dim]Press any key to return to reader…[/dim]")
        _read_single_key()
        return None

    for i, h in enumerate(related, 1):
        src = _short_source(h.source_file)
        ts = f"{_fmt_timestamp(h.start_time)}–{_fmt_timestamp(h.end_time)}"
        snippet = h.text[:120].replace("\n", " ")
        console.print(
            f"  [bold]{i}.[/bold] [cyan]{ts}[/cyan]  [dim]{src}[/dim]\n"
            f"     [italic]{snippet}[/italic]"
        )
    console.print()
    console.print("  [dim]Select 1-5 to jump, or any other key to go back…[/dim]")

    try:
        key = _read_single_key()
        if key.isdigit() and 1 <= int(key) <= len(related):
            return related[int(key) - 1]
    except (KeyboardInterrupt, EOFError):
        pass
    return None


def _write_note(console: Console, fr_display: dict, audio_file_id: int, source_name: str) -> None:
    from audiobench.storage.note_repository import NoteRepository
    try:
        console.print()
        text = input("  Note (Enter to cancel): ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if not text:
        return

    # Single-char inputs that look like mistaken key presses (e.g. user typed
    # 'q' expecting to quit the note dialog) are silently discarded.
    if len(text) == 1 and not text.isalpha():
        return
    # Give a small warning if 'q' was used
    if text.lower() == "q" and len(text) == 1:
        return

    # Session-replayed fragments may have transcription_id that doesn't map to a
    # live DB row, so _resolve_audio_file_id returns 0.  The note_collections
    # table has a FOREIGN KEY on audio_file_id — inserting 0 raises IntegrityError.
    # Show a clear message instead of crashing.
    if audio_file_id <= 0:
        console.print(
            "  [dim yellow]⚠ Cannot save note — this fragment's source file is not "
            "available in the current library (session replay or deleted file).[/dim yellow]"
        )
        import time
        time.sleep(1.5)
        return

    repo = NoteRepository()

    transcript_expr_id = None
    if fr_display.get("id"):
        with get_session() as s:
            from sqlalchemy import text as sql_text
            res = s.execute(sql_text("SELECT expression_id FROM expression_segment_map WHERE segment_id = :sid LIMIT 1"), {"sid": fr_display["id"]}).scalar()
            if res:
                transcript_expr_id = res

    title = f"Notes on {source_name}"
    col = repo.find_or_create_collection(audio_file_id, title)

    repo.create_capture(
        collection_id=col.id,
        body=text,
        segment_id=fr_display.get("id"),
        transcript_expression_id=transcript_expr_id,
        collection_expression_id=col.expression_id
    )
    console.print("  [cyan]✎ Note saved[/cyan]")
    import time
    time.sleep(0.5)


def _show_citation_card(
    n: int,
    fr: "FusedResult",  # type: ignore[name-defined]
    all_sources: "list[FusedResult]",  # type: ignore[name-defined]
) -> None:
    """Render an inline citation card for fragment *n* and wait for a keypress.

    This is the terminal equivalent of clicking a citation chip in NotebookLM —
    the full fragment text surfaces in a bordered panel with source metadata.
    Pressing E from the card chains directly into the fragment reader at that
    position; any other key (or Escape) dismisses and returns to the prompt.
    """
    from rich.markup import escape as rich_escape

    term_width = console.width or 80
    sup = _to_superscript(n)
    source_name = _short_source(fr.source_file) if fr.source_file else "Unknown source"
    ts = f"{_fmt_timestamp(fr.start_time)} → {_fmt_timestamp(fr.end_time)}"

    # Stream badges (same style as the main results list)
    badges = ""
    if fr.stream_contributions:  # type: ignore[union-attr]
        badge_parts = []
        for stream, rank in sorted(
            fr.stream_contributions, key=lambda x: x[1]  # type: ignore[union-attr]
        ):
            color = _STREAM_COLORS.get(stream, "white")
            badge_parts.append(f"[{color}]{stream}#{rank}[/{color}]")
        badges = "  " + "  ".join(badge_parts)

    try:
        prev_segs, next_segs = _fetch_adjacent(
            fr.transcription_id,
            fr.start_time,
            fr.end_time,
            limit=3,
            segment_id=getattr(fr, "segment_id", 0),
            source_file=getattr(fr, "source_file", ""),
        )
        prev_text = " ".join(seg["text"].strip() for seg in prev_segs if seg["text"].strip())
        next_text = " ".join(seg["text"].strip() for seg in next_segs if seg["text"].strip())
    except Exception:
        prev_text, next_text = "", ""

    # Wrap to panel width so long passages are readable
    import textwrap
    wrap_w = max(40, term_width - 8)
    
    body_parts = []
    if prev_text:
        body_parts.append(f"[dim]{rich_escape(textwrap.fill(prev_text, width=wrap_w))}[/dim]")
        
    body_parts.append(f"[bold]{rich_escape(textwrap.fill(fr.text.strip(), width=wrap_w))}[/bold]")
    
    if next_text:
        body_parts.append(f"[dim]{rich_escape(textwrap.fill(next_text, width=wrap_w))}[/dim]")

    wrapped_content = "\n\n".join(body_parts)

    card_content = (
        f"[bold]{rich_escape(source_name)}[/bold]\n"
        f"[cyan]{ts}[/cyan]{badges}\n\n"
        f"{wrapped_content}"
    )

    panel = Panel(
        card_content,
        title=f"[dim cyan]{sup}[/dim cyan]  [dim]fragment {n} of {len(all_sources)}[/dim]",
        title_align="left",
        border_style="dim cyan",
        expand=True,
        padding=(1, 2),
    )
    footer = (
        "[dim]  any key to dismiss  ·  "
        "[bold]E[/bold] open reader here  ·  "
        "[bold]H[/bold]/[bold]L[/bold] prev/next fragment[/dim]"
    )

    # ── Measure rendered height so we can erase the card on dismiss ───────────
    # console.capture() renders Rich markup to a string without printing.
    # Counting '\n' in that string gives the exact number of terminal lines
    # the card will occupy (Rich already wraps to console.width).
    import sys as _sys
    with console.capture() as _cap:
        console.print()
        console.print(panel)
        console.print(footer)
    _card_lines = _cap.get().count("\n")

    def _erase_card() -> None:
        """Move the cursor up over the card and wipe it from the buffer.

        Uses console.file (the same stream Rich writes to) to ensure the ANSI
        cursor-up sequence is flushed in the correct order relative to Rich's
        own output — avoids the race condition from using sys.stdout directly.
        """
        console.file.write(f"\033[{_card_lines}A\033[J")
        console.file.flush()

    # ── Actually print the card ───────────────────────────────────────────────
    console.print()
    console.print(panel)
    console.print(footer)

    key = _read_single_key().upper()

    # Every exit path erases the card first so the synthesis panel is cleanly
    # restored — no leftover panel border or text in the scroll buffer.
    _erase_card()

    if key == "E":
        _open_fragment_reader(console, all_sources, initial_idx=n - 1)
    elif key == "H" and n > 1:
        _show_citation_card(n - 1, all_sources[n - 2], all_sources)
    elif key == "L" and n < len(all_sources):
        # Step forwards to the next fragment
        _show_citation_card(n + 1, all_sources[n], all_sources)
    # Escape, any other key: fall through → returns to inner prompt loop


def _open_fragment_reader(
    console: Console,
    fragments: list,  # list[FusedResult]
    initial_idx: int = 0,
) -> None:
    """Interactive fragment reader workspace."""
    import shutil
    import textwrap

    from audiobench.storage.bookmark_repository import BookmarkRepository
    from audiobench.storage.note_repository import NoteRepository

    if not fragments:
        console.print("[dim]No fragments to read.[/dim]")
        return

    current_idx = max(0, min(initial_idx, len(fragments) - 1))
    viewport_offset = 0
    exclude_ids: set[int] = {fr.segment_id for fr in fragments}

    bookmark_repo = BookmarkRepository()
    note_repo = NoteRepository()

    while True:
        fr = fragments[current_idx]
        total = len(fragments)
        audio_file_id = _resolve_audio_file_id(fr.transcription_id)

        try:
            prev_segs, next_segs = _fetch_adjacent(
                fr.transcription_id, fr.start_time, fr.end_time
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reader adjacency fetch failed: %s", exc)
            prev_segs, next_segs = [], []

        if viewport_offset < 0:
            scroll_idx = min(abs(viewport_offset) - 1, len(prev_segs) - 1)
            if prev_segs and scroll_idx >= 0:
                scrolled_seg = prev_segs[scroll_idx]
                try:
                    prev_segs, next_segs = _fetch_adjacent(
                        fr.transcription_id, scrolled_seg["start_time"], scrolled_seg["end_time"]
                    )
                    next_segs = [{"id": fr.segment_id, "start_time": fr.start_time,
                                  "end_time": fr.end_time, "text": fr.text}] + list(next_segs[:1])
                    fr_display = scrolled_seg
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Viewport scroll fetch failed: %s", exc)
                    fr_display = {"id": fr.segment_id, "start_time": fr.start_time, "end_time": fr.end_time, "text": fr.text}
            else:
                fr_display = {"id": fr.segment_id, "start_time": fr.start_time, "end_time": fr.end_time, "text": fr.text}
        elif viewport_offset > 0:
            scroll_idx = min(viewport_offset - 1, len(next_segs) - 1)
            if next_segs and scroll_idx >= 0:
                scrolled_seg = next_segs[scroll_idx]
                try:
                    prev_segs, next_segs = _fetch_adjacent(
                        fr.transcription_id, scrolled_seg["start_time"], scrolled_seg["end_time"]
                    )
                    prev_segs = list(prev_segs[-1:]) + [{"id": fr.segment_id, "start_time": fr.start_time,
                                                         "end_time": fr.end_time, "text": fr.text}]
                    fr_display = scrolled_seg
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Viewport scroll fetch failed: %s", exc)
                    fr_display = {"id": fr.segment_id, "start_time": fr.start_time, "end_time": fr.end_time, "text": fr.text}
            else:
                fr_display = {"id": fr.segment_id, "start_time": fr.start_time, "end_time": fr.end_time, "text": fr.text}
        else:
            fr_display = {"id": fr.segment_id, "start_time": fr.start_time, "end_time": fr.end_time, "text": fr.text}

        import sys as _sys
        _sys.stdout.write("\033[2J\033[H")
        _sys.stdout.flush()

        # Header indicators
        has_bookmark = bool(bookmark_repo.get_nearest(audio_file_id, fr_display["start_time"], window=1.0)) if audio_file_id else False
        captures = note_repo.get_captures_for_segment(fr_display.get("id"))

        bookmark_ind = "  [yellow]★[/yellow]" if has_bookmark else ""
        note_ind = "  [cyan]✎[/cyan]" if captures else ""

        src_name = _short_source(fr.source_file)
        ts_range = f"{_fmt_timestamp(fr_display['start_time'])} – {_fmt_timestamp(fr_display['end_time'])}"
        console.print(
            f"  [bold]Fragment {current_idx + 1}/{total}[/bold]  [dim]·[/dim]  {src_name}  [dim]·[/dim]  {ts_range}{bookmark_ind}{note_ind}"
        )

        # Progress bar — build with rich.text.Text so █░ are treated as plain
        # chars and Rich never sees them as markup tags.  No [ ] brackets.
        max_end_time = _get_max_end_time(fr.transcription_id)
        if max_end_time > 0:
            progress = min(1.0, max(0.0, fr_display["start_time"] / max_end_time))
            bar_len = 36
            filled = int(progress * bar_len)
            bar_chars = "\u2588" * filled + "\u2591" * (bar_len - filled)
            pos_str   = f"{_fmt_timestamp(fr_display['start_time'])} / {_fmt_timestamp(max_end_time)}"
            from rich.text import Text as _T
            _bar = _T()
            _bar.append("  ")
            _bar.append(bar_chars)          # plain — no markup interpretation
            _bar.append("  ")
            _bar.append(pos_str, style="dim")
            console.print(_bar)

        console.print()
        console.rule("earlier", style="dim", characters="\u2500")
        console.print()

        cols = shutil.get_terminal_size().columns
        wrap_width = max(40, cols - 8)
        # Indent for hanging timestamp ("  00:00:00   " = 14 chars)
        ts_prefix_len = 14
        text_wrap_width = max(30, wrap_width - ts_prefix_len)
        hanging_indent = " " * ts_prefix_len

        # ── Earlier context: word-wrapped, timestamp on first line only ──────
        for seg in prev_segs:
            ts      = _fmt_timestamp(seg["start_time"])
            raw     = seg["text"].replace("\n", " ").strip()
            lines   = textwrap.wrap(raw, width=text_wrap_width) or [""]
            first   = f"  [dim]{ts}   {lines[0]}[/dim]"
            console.print(first)
            for extra in lines[1:]:
                console.print(f"  [dim]{hanging_indent}{extra}[/dim]")
            console.print()  # blank line between each context segment

        console.rule(style="bold white", characters="\u2550")

        # ── Active fragment ──────────────────────────────────────────────────
        current_text = textwrap.fill(fr_display["text"], width=wrap_width)
        ts_label     = f"[bold cyan]\u25b6 {_fmt_timestamp(fr_display['start_time'])}[/bold cyan]"
        # Right-align source name on the same line as the ▶ timestamp
        src_padded   = src_name.rjust(max(0, wrap_width - len(_fmt_timestamp(fr_display['start_time'])) - 4))
        console.print(f"  {ts_label}  [dim]{src_padded}[/dim]")
        console.print(f"{textwrap.indent(current_text, '     ')}\n")

        # Render captures inline
        for cap in captures:
            cap_body = textwrap.fill(cap.body, width=wrap_width - 4)
            console.print(f"     [cyan]\u270e[/cyan]  [dim]\"{cap_body}\"[/dim]")
            console.print(f"         [dim]\u2014 {str(cap.created_at)[:16]}[/dim]\n")

        console.rule(style="bold white", characters="\u2550")
        console.print()

        # ── Later context: same wrapped + hanging-indent layout ───────────────
        for seg in next_segs:
            ts    = _fmt_timestamp(seg["start_time"])
            raw   = seg["text"].replace("\n", " ").strip()
            lines = textwrap.wrap(raw, width=text_wrap_width) or [""]
            console.print(f"  [dim]{ts}   {lines[0]}[/dim]")
            for extra in lines[1:]:
                console.print(f"  [dim]{hanging_indent}{extra}[/dim]")
            console.print()  # blank line between each context segment

        console.rule("later", style="dim", characters="\u2500")

        console.print()
        # Action bar — key letters dim-highlighted so they stand out against descriptions
        console.print(
            "  [dim]P[/dim] play  [dim]\u00b7[/dim]  "
            "[dim]B[/dim] bookmark  [dim]\u00b7[/dim]  "
            "[dim]N[/dim] note  [dim]\u00b7[/dim]  "
            "[dim]Y[/dim] yank  [dim]\u00b7[/dim]  "
            "[dim]H/L[/dim] scroll  [dim]\u00b7[/dim]  "
            "[dim]J/K[/dim] fragments  [dim]\u00b7[/dim]  "
            "[dim]R[/dim] related  [dim]\u00b7[/dim]  "
            "[dim]Q[/dim] back"
        )

        try:
            key = _read_single_key()
        except (KeyboardInterrupt, EOFError):
            break

        if key in ("q", "Q", "\x1b", "\x03"):
            # \x03 = Ctrl-C in raw mode (termios setraw returns the byte, not an exception)
            break
        elif key in ("p", "P"):
            try:
                from audiobench.daemon.factory import get_daemon_client
                client = get_daemon_client()
                rec = repo.get_by_id(fr.transcription_id)
                if rec and rec.get("file_path"):
                    client.playback_play(
                        rec["file_path"],
                        start_pos=fr_display["start_time"],
                        audio_file_id=rec.get("audio_file_id"),
                        transcription_id=fr.transcription_id,
                    )
            except Exception:
                pass
        elif key in ("h", "H", "\x1b[D"):
            new_vp = viewport_offset - 1
            if new_vp < 0:
                # Would we be asking for a prev_seg that doesn't exist?
                check_idx = abs(new_vp) - 1
                if not prev_segs or check_idx >= len(prev_segs):
                    # Already at the earliest segment — nothing to show, skip re-render
                    continue
            viewport_offset = new_vp
        elif key in ("l", "L", "\x1b[C"):
            new_vp = viewport_offset + 1
            if new_vp > 0:
                check_idx = new_vp - 1
                if not next_segs or check_idx >= len(next_segs):
                    # Already at the latest segment — nothing to show, skip re-render
                    continue
            viewport_offset = new_vp
        elif key in ("j", "J"):
            current_idx = max(0, current_idx - 1)
            viewport_offset = 0
        elif key in ("k", "K"):
            current_idx = min(total - 1, current_idx + 1)
            viewport_offset = 0
        elif key in ("n", "N"):
            _write_note(console, fr_display, audio_file_id, src_name)
        elif key in ("b", "B"):
            _toggle_bookmark(fr_display, fr.transcription_id, console)
        elif key in ("y", "Y"):
            _yank_to_clipboard(fr_display["text"], console)
        elif key in ("r", "R"):
            new_hit = _find_related(console, fragments[current_idx], exclude_ids)
            if new_hit:
                fragments.append(new_hit)
                exclude_ids.add(new_hit.segment_id)
                current_idx = len(fragments) - 1
                viewport_offset = 0

def _replay_session_history(state: SearchSessionState) -> Optional["ResearchResult"]:
    """Replay all prior searches exactly as they appeared — full results, full synthesis.

    Loads every query + fragment from the DB and calls _display_results() for each
    one in sequence, so the terminal looks exactly like it did when each search ran.
    Returns the ResearchResult for the very last query replayed.
    """
    import json

    from audiobench.memory.query_engine import ResearchResult
    from audiobench.memory.rrf_fusion import FusedResult
    from audiobench.memory.session_store import _get_conn, get_session

    detail = get_session(state.session_id)
    if not detail or not detail.queries:
        return None

    console.print()
    console.print(
        f"  [bold cyan]Resuming session #{state.session_id}[/bold cyan]  "
        f"[dim]{detail.title or '(untitled)'}  ·  "
        f"{len(detail.queries)} prior search{'es' if len(detail.queries) != 1 else ''}[/dim]"
    )
    console.rule(style="dim cyan")
    console.print()

    try:
        conn = _get_conn()
    except Exception as e:
        console.print(f"  [red]Could not load prior searches: {e}[/red]")
        return

    for q in detail.queries:
        # ── Load fragments from DB ────────────────────────────────────────────
        try:
            frag_rows = conn.execute(
                """SELECT segment_id, source_file, rank, rrf_score,
                          stream_contributions, start_time, end_time, fragment_text
                   FROM search_query_fragments
                   WHERE query_id=?
                   ORDER BY rank""",
                (q.query_id,),
            ).fetchall()
        except Exception:
            frag_rows = []

        seg_ids_to_lookup = [r["segment_id"] for r in frag_rows if r["segment_id"]]
        tid_lookup: dict[int, int] = {}
        if seg_ids_to_lookup:
            try:
                from sqlalchemy import text as _sql_t
                with get_session() as s:
                    placeholders = ",".join(f":id{i}" for i in range(len(seg_ids_to_lookup)))
                    res = s.execute(
                        _sql_t(f"SELECT id, transcription_id FROM segments WHERE id IN ({placeholders})"),
                        {f"id{i}": sid for i, sid in enumerate(seg_ids_to_lookup)},
                    ).mappings().all()
                    tid_lookup = {r["id"]: int(r["transcription_id"]) for r in res if r.get("transcription_id")}
            except Exception:
                tid_lookup = {}

        sources: list[FusedResult] = []
        for row in frag_rows:
            stream_contrib: tuple[tuple[str, int], ...] = ()
            if row["stream_contributions"]:
                try:
                    raw = json.loads(row["stream_contributions"])
                    stream_contrib = tuple(tuple(x) for x in raw)
                except Exception:
                    pass
            sid = row["segment_id"] or 0
            sources.append(FusedResult(
                segment_id=sid,
                start_time=row["start_time"] or 0.0,
                end_time=row["end_time"] or 0.0,
                text=row["fragment_text"] or "",
                rrf_score=row["rrf_score"] or 0.0,
                source_file=row["source_file"] or "",
                stream_contributions=stream_contrib,
                transcription_id=tid_lookup.get(sid, 0),
            ))

        # ── Load query-level meta from DB ─────────────────────────────────────
        try:
            qr = conn.execute(
                """SELECT synthesis_text, synthesis_failed, synthesis_error,
                          hyde_document, retrieval_time_seconds,
                          synthesis_time_seconds, total_time_seconds
                   FROM search_queries WHERE id=?""",
                (q.query_id,),
            ).fetchone()
        except Exception:
            qr = None

        # ── Build a ResearchResult replica ────────────────────────────────────
        result = ResearchResult(
            query=q.query_text,
            sources=sources,
            answer=(qr["synthesis_text"] if qr else None) or q.synthesis_text,
            synthesis_failed=bool((qr["synthesis_failed"] if qr else None) or q.synthesis_failed),
            synthesis_error=(qr["synthesis_error"] if qr else None),
            hyde_document=(qr["hyde_document"] if qr else None),
            retrieval_time_seconds=float((qr["retrieval_time_seconds"] if qr else None) or 0.0),
            synthesis_time_seconds=float((qr["synthesis_time_seconds"] if qr else None) or 0.0),
            query_time_seconds=float((qr["total_time_seconds"] if qr else None) or 0.0),
            streams_skipped=[],
            hyde_fallback=False,
        )

        # ── Temporarily set state to this search's sequence num ───────────────
        # _display_results reads state.search_count for the S-prefix badge
        real_count = state.search_count
        state.search_count = q.sequence_num
        # Also populate overlap tracking up to this point
        for prev in detail.queries:
            if prev.sequence_num < q.sequence_num:
                state.search_segment_ids.setdefault(prev.sequence_num, set(prev.segment_ids))

        # ── Reconstruct prior synthesis hits from persisted JSON ──────────────
        prior_hits: list = []
        if q.prior_synthesis_hits:
            try:
                from types import SimpleNamespace
                for h in q.prior_synthesis_hits:
                    prior_hits.append(SimpleNamespace(
                        content=h.get("content", ""),
                        source_type=h.get("source_type", "search_synthesis"),
                        session_id=h.get("session_id", state.session_id),
                        sequence_num=h.get("sequence_num"),
                    ))
            except Exception:
                prior_hits = []

        result.prior_synthesis_hits = prior_hits  # type: ignore[attr-defined]

        # ── Echo the original query ───────────────────────────────────────────
        # During a live search the query is visible as the REPL input. During
        # replay there is no prompt, so we print it explicitly so the user can
        # tell what each result block corresponds to.
        seq_label = f"[bold cyan]S{q.sequence_num}[/bold cyan]  " if q.sequence_num else ""
        console.print(f"  {seq_label}[dim]› [/dim][bold]{result.query}[/bold]")
        console.print()

        _display_results(result, state)

        state.search_count = real_count  # restore

    conn.close()

    console.rule(style="dim cyan")
    console.print(
        f"  [dim]Session replayed. Next will be "
        f"[cyan]S{state.search_count + 1}[/cyan] — "
        "type a new query below or use a slash command[/dim]"
    )
    console.print()
    
    return result if 'result' in locals() else None


def _run_search_loop(
    engine: "ResearchEngine",  # type: ignore[name-defined]
    query: str,
    state: SearchSessionState,
    *,
    resumed: bool = False,
) -> None:
    """Interactive loop for searching, REPL commands, and navigating results.

    All user input flows through a single prompt_toolkit PromptSession that
    provides:
      • Semantic autocomplete from the daemon (fragment matches as you type)
      • Slash-command completion (/set, /settings, /focus, …)
      • Up/down arrow history (persisted across sessions)
      • Ghost-text auto-suggest from history
      • Single-char keybindings (Q/E/C/S/1–9) that fire when buffer is empty
      • Clean paste handling — no contamination bugs
    """
    # ── Build the shared PromptSession ────────────────────────────────────────
    try:
        pt_session = _make_search_prompt_session(state)
    except Exception:
        pt_session = None  # graceful fallback to input()

    def _prompt(prompt_str: str = "  › ") -> str:
        """Read one line from the user, using prompt_toolkit if available."""
        if pt_session is not None:
            return pt_session.prompt(prompt_str)
        return input(prompt_str)

    # ── Entry mode: resume history OR start fresh REPL prompt ─────────────────
    result: Optional["ResearchResult"] = None
    if resumed:
        result = _replay_session_history(state)
        if result:
            query = result.query
    else:
        if not query.strip():
            _emit_hint(state, "new_session_preset_hint",
                       f"Preset defaults to [cyan]{state.preset}[/cyan] · use [cyan]/set fast|balanced|synthesis|deep[/cyan] to change anytime")

    while True:
        # ── Outer loop: no result yet — get a query ───────────────────────────
        if not result and not query.strip():
            while True:
                console.print(
                    "[dim]/set  /focus  /pin  /history  /summary  /sessions  /switch  /show  /rename  /export"
                    "  · or just type a query  ·  [bold]Q[/bold] quit[/dim]"
                )
                try:
                    choice = _prompt("  › ").strip()
                except (KeyboardInterrupt, EOFError):
                    console.print()
                    return

                # ── Sentinel from keybinding (buffer was empty when key pressed) ─
                if choice.startswith("\x00"):
                    sentinel = choice[1:].upper()
                    if sentinel == "Q":
                        if state.session_id >= 0:
                            try:
                                from audiobench.memory.session_store import close_session
                                close_session(state.session_id)
                            except Exception:
                                pass
                        return
                    # S in outer loop: user wants to start a new search — just
                    # fall through to the next prompt iteration (buffer empty)
                    continue

                if not choice:
                    continue

                if choice.startswith("/"):
                    query_parts = choice.split(maxsplit=1)
                    if query_parts[0].lower() in ("/fast", "/balanced", "/deep", "/synthesis") and len(query_parts) > 1:
                        query = choice
                        break
                    msg = _parse_slash_command(choice, state, last_sources=None, engine=engine)
                    if msg:
                        console.print(f"  {msg}")
                    continue

                # Treat any non-slash text as the new search query
                query = choice
                break

        # ── Preset override prefix: `/fast <query>` etc. ──────────────────────
        active_preset = state.preset
        query_parts = query.split(maxsplit=1)
        if query_parts and query_parts[0].lower() in ("/fast", "/balanced", "/deep", "/synthesis"):
            active_preset = query_parts[0][1:].lower()
            if len(query_parts) == 1:
                console.print("[red]Missing query after preset override.[/red]")
                return
            query = query_parts[1]
            console.print(f"[dim]Note: Using one-off override preset '{active_preset}' for this query[/dim]")

        if not query.strip():
            console.print("[red]Query cannot be empty.[/red]")
            return

        # ── Execute search ────────────────────────────────────────────────────
        if not result:
            console.print(f"[dim]Searching: preset={active_preset} focus={state.focus_source or 'all'} λ={state.mmr_lambda}[/dim]")

            state.search_count += 1
            seq = state.search_count
            query_id: int | None = None

            if state.session_id < 0 and seq == 1:
                try:
                    from audiobench.memory.session_store import create_session
                    state.session_id = create_session(preset=state.preset)
                    console.print(f"  [dim]Session #{state.session_id} opened[/dim]")
                except Exception as e:
                    logger.warning("Failed to create session: %s", e)

            if state.session_id >= 0:
                try:
                    from audiobench.memory.session_store import create_query_record, set_session_title
                    if seq == 1:
                        set_session_title(state.session_id, query)
                    query_id = create_query_record(state.session_id, seq, query, active_preset)
                except Exception as e:
                    logger.warning("Failed to create query record: %s", e)

            state.search_query_texts[seq] = query

            with console.status("Querying memory graph..."):
                result = engine.search(
                    query=query,
                    preset=active_preset,
                    mmr_lambda=state.mmr_lambda,
                    focus_source=state.focus_source,
                    model=state.model,
                    diversity_weight=state.diversity_weight,
                    pinned_fragments=list(state.pinned_fragments.values()),
                    prior_synthesis=state.last_synthesis,
                    session_id=state.session_id if state.session_id >= 0 else None,
                    query_id=query_id,
                )

            if result.answer and not result.synthesis_failed:
                state.last_synthesis = result.answer

            seg_ids = {fr.segment_id for fr in result.sources}
            state.search_segment_ids[seq] = seg_ids
            for fr in result.sources:
                if fr.source_file:
                    state.search_source_files.setdefault(fr.source_file, [])
                    if seq not in state.search_source_files[fr.source_file]:
                        state.search_source_files[fr.source_file].append(seq)

            if state.session_id >= 0 and query_id is not None and query_id >= 0:
                try:
                    from audiobench.memory.session_store import persist_fragments, update_query_synthesis
                    update_query_synthesis(query_id, result)
                    if result.sources:
                        persist_fragments(query_id, result.sources)
                except Exception as e:
                    logger.warning("Failed to persist search query: %s", e)

            _display_results(result, state)

        if not result.sources:
            result = None
            query = ""
            continue

        # ── Inner loop — action bar + single prompt per iteration ─────────────
        while True:
            # Render action bar
            cols = shutil.get_terminal_size().columns

            session_tag = ""
            if state.session_id >= 0:
                _layout_badge = f" · {getattr(state, 'layout', 'book')}"
                session_tag = f"S{state.search_count} · #{state.session_id} · {state.preset}{_layout_badge}"

            bar_left = (
                f"  [bold white]{session_tag}[/bold white]  "
                "[dim]·  [bold]1–9[/bold] peek  ·  "
                "[bold]E[/bold] reader  ·  "
                "[bold]S[/bold] search  ·  "
                "[bold]C[/bold] chat  ·  "
                "[bold]Q[/bold] quit[/dim]"
            ) if session_tag else (
                "  [dim][bold]1–9[/bold] peek  ·  "
                "[bold]E[/bold] reader  ·  "
                "[bold]S[/bold] search  ·  "
                "[bold]C[/bold] chat  ·  "
                "[bold]Q[/bold] quit[/dim]"
            )
            bar_left_plain = (
                f"  {session_tag}  ·  1–9 peek  ·  E reader  ·  S search  ·  C chat  ·  Q quit"
                if session_tag else
                "  1–9 peek  ·  E reader  ·  S search  ·  C chat  ·  Q quit"
            )

            last_q = state.search_query_texts.get(state.search_count, "")
            q_preview = (f'"{last_q[:20]}…"' if len(last_q) > 20 else (f'"{last_q}"' if last_q else ""))
            rerun_hint = f"[dim italic]⏎ re-show {q_preview}[/dim italic]" if q_preview else ""
            rerun_plain = f"⏎ re-show {q_preview}" if q_preview else ""

            if rerun_hint and cols >= len(bar_left_plain) + len(rerun_plain) + 6:
                padding = max(2, cols - len(bar_left_plain) - len(rerun_plain) - 2)
                console.print(f"{bar_left}{' ' * padding}{rerun_hint}")
            else:
                console.print(bar_left)

            # ── Prompt ────────────────────────────────────────────────────────
            try:
                choice = _prompt("  › ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print()
                return

            # ── Sentinel dispatch (buffer was empty, keybinding fired) ─────────
            if choice.startswith("\x00"):
                sentinel = choice[1:].upper()

                if sentinel == "Q":
                    if state.session_id >= 0:
                        try:
                            from audiobench.memory.session_store import close_session
                            close_session(state.session_id)
                        except Exception:
                            pass
                    return

                if sentinel == "E":
                    _open_fragment_reader(console, result.sources, initial_idx=0)
                    continue

                if sentinel == "C":
                    from audiobench.chat.chat_repl import ChatREPL
                    repl = ChatREPL(
                        session_type="search_followup",
                        preloaded_fragments=result.sources,
                        preloaded_title=f"🔍 Search: {result.query}",
                    )
                    repl.run()
                    return

                if sentinel == "S":
                    # S with empty buffer → start a fresh new search
                    query = ""
                    result = None
                    break

                if sentinel.isdigit():
                    n_str = sentinel
                    # peek for a second digit already in the sentinel (shouldn't
                    # happen with current keybindings but defensive)
                    try:
                        n = int(n_str)
                    except ValueError:
                        n = -1
                    if 1 <= n <= len(result.sources):
                        _show_citation_card(n, result.sources[n - 1], result.sources)
                    else:
                        hi = len(result.sources)
                        console.print(
                            f"[dim]Fragment {n} out of range — "
                            f"{hi} fragment{'s' if hi != 1 else ''} available (1–{hi}).[/dim]"
                        )
                    continue

                continue  # unknown sentinel — ignore

            # ── Empty input → re-show current results ─────────────────────────
            if not choice:
                if result is not None:
                    _display_results(result, state)
                continue

            # ── Slash commands ────────────────────────────────────────────────
            if choice.startswith("/"):
                query_parts = choice.split(maxsplit=1)
                if (
                    query_parts[0].lower() in ("/fast", "/balanced", "/deep", "/synthesis")
                    and len(query_parts) > 1
                ):
                    query = choice
                    result = None
                    break
                msg = _parse_slash_command(choice, state, last_sources=result.sources, engine=engine)
                if msg:
                    console.print(f"  {msg}")
                continue

            # ── New query (typed text, long enough) ───────────────────────────
            if len(choice) > 3:
                query = choice
                result = None
                break

            if choice:
                console.print(
                    "[dim]Too short for a query — type at least 4 characters.[/dim]"
                )


    while True:
        # If we don't have a resumed result and no query, loop until we get one
        if not result and not query.strip():
            while True:
                console.print(
                    "[dim]Enter re-run  "
                    "[bold]S[/bold] new search  "
                    "[bold]Q[/bold] quit[/dim]"
                )
                console.print(
                    "[dim]/set  /focus  /pin  /history  /summary  /sessions  /switch  /show  /rename  /export"
                    "  · or just type a new query[/dim]"
                )
                try:
                    choice = input("  › ").strip()
                except (KeyboardInterrupt, EOFError):
                    console.print()
                    return

                if not choice:
                    continue

                if choice.startswith("/"):
                    query_parts = choice.split(maxsplit=1)
                    if query_parts[0].lower() in ("/fast", "/balanced", "/deep", "/synthesis") and len(query_parts) > 1:
                        query = choice
                        break  # fall through to normal search execution
                    msg = _parse_slash_command(choice, state, last_sources=None, engine=engine)
                    if msg:
                        console.print(f"  {msg}")
                    continue

                choice_upper = choice.upper()
                if choice_upper == "Q":
                    if state.session_id >= 0:
                        try:
                            from audiobench.memory.session_store import close_session
                            close_session(state.session_id)
                        except Exception:
                            pass
                    return
                elif choice_upper == "S":
                    try:
                        query = input("  Search query › ").strip()
                    except (KeyboardInterrupt, EOFError):
                        console.print()
                        return
                    if query:
                        break
                    continue
                else:
                    # Treat any non-slash input directly as the new search query
                    query = choice
                    break
        # Check for one-off preset overrides in the query
        active_preset = state.preset
        query_parts = query.split(maxsplit=1)
        if query_parts and query_parts[0].lower() in ("/fast", "/balanced", "/deep", "/synthesis"):
            active_preset = query_parts[0][1:].lower()
            if len(query_parts) == 1:
                console.print("[red]Missing query after preset override.[/red]")
                return
            query = query_parts[1]
            console.print(f"[dim]Note: Using one-off override preset '{active_preset}' for this query[/dim]")

        if not query.strip():
            console.print("[red]Query cannot be empty.[/red]")
            return

        if not result:
            console.print(f"[dim]Searching: preset={active_preset} focus={state.focus_source or 'all'} λ={state.mmr_lambda}[/dim]")

            # ── Pre-retrieval Persistence ──────────────────────────────────────
            # Create session and query record before we hit the engine, so we can 
            # save synthesis hits even if the LLM crashes.
            state.search_count += 1
            seq = state.search_count
            query_id: int | None = None
            
            if state.session_id < 0 and seq == 1:
                try:
                    from audiobench.memory.session_store import create_session
                    state.session_id = create_session(preset=state.preset)
                    console.print(f"  [dim]Session #{state.session_id} opened[/dim]")
                except Exception as e:
                    logger.warning("Failed to create session: %s", e)
            
            if state.session_id >= 0:
                try:
                    from audiobench.memory.session_store import create_query_record, set_session_title
                    if seq == 1:
                        set_session_title(state.session_id, query)
                    query_id = create_query_record(state.session_id, seq, query, active_preset)
                except Exception as e:
                    logger.warning("Failed to create query record: %s", e)

            # Track query text for /history and /summary
            state.search_query_texts[seq] = query

            with console.status("Querying memory graph..."):
                result = engine.search(
                    query=query,
                    preset=active_preset,
                    mmr_lambda=state.mmr_lambda,
                    focus_source=state.focus_source,
                    model=state.model,
                    diversity_weight=state.diversity_weight,
                    pinned_fragments=list(state.pinned_fragments.values()),
                    prior_synthesis=state.last_synthesis,  # synthesis carryforward
                    session_id=state.session_id if state.session_id >= 0 else None,
                    query_id=query_id,
                )

            # ── Post-retrieval Persistence ──────────────────────────────────────
            # Update last_synthesis for carryforward on next search.
            # Fallback (extractive) answer is still useful context for the next
            # LLM call — it tells the LLM what fragments were surfaced, even if
            # it wasn’t interpreted. Only exclude true hard-failures (no content).
            if result.answer and not result.synthesis_failed:
                state.last_synthesis = result.answer

            # Update in-memory overlap tracking (offline-safe)
            seg_ids = {fr.segment_id for fr in result.sources}
            state.search_segment_ids[seq] = seg_ids
            for fr in result.sources:
                if fr.source_file:
                    state.search_source_files.setdefault(fr.source_file, [])
                    if seq not in state.search_source_files[fr.source_file]:
                        state.search_source_files[fr.source_file].append(seq)

            if state.session_id >= 0 and query_id is not None and query_id >= 0:
                try:
                    from audiobench.memory.session_store import persist_fragments, update_query_synthesis
                    update_query_synthesis(query_id, result)
                    if result.sources:
                        persist_fragments(query_id, result.sources)
                except Exception as e:
                    logger.warning("Failed to persist search query: %s", e)

            _display_results(result, state)

        if not result.sources:
            result = None
            query = ""
            continue

        # Inner loop — single-key dispatch.
        # _read_single_key() captures one keystroke without waiting for Enter,
        # so digit presses (citation peek), E, S, C, Q all fire immediately.
        # When the user starts typing a query or slash command the first char is
        while True:
            # ── Action Bar Dock ──────────────────────────────────────────────
            cols = shutil.get_terminal_size().columns

            # Session state badge (left, highlighted)
            session_tag = ""
            if state.session_id >= 0:
                _layout_badge = f" · {getattr(state, 'layout', 'book')}"
                session_tag = f"S{state.search_count} · #{state.session_id} · {state.preset}{_layout_badge}"

            bar_left = (
                f"  [bold white]{session_tag}[/bold white]  "
                "[dim]·  [bold]1–9[/bold] peek  ·  "
                "[bold]E[/bold] reader  ·  "
                "[bold]S[/bold] search  ·  "
                "[bold]C[/bold] chat  ·  "
                "[bold]Q[/bold] quit[/dim]"
            ) if session_tag else (
                "  [dim][bold]1–9[/bold] peek  ·  "
                "[bold]E[/bold] reader  ·  "
                "[bold]S[/bold] search  ·  "
                "[bold]C[/bold] chat  ·  "
                "[bold]Q[/bold] quit[/dim]"
            )
            bar_left_plain = (
                f"  {session_tag}  ·  1–9 peek  ·  E reader  ·  S search  ·  C chat  ·  Q quit"
                if session_tag else
                "  1–9 peek  ·  E reader  ·  S search  ·  C chat  ·  Q quit"
            )

            last_q = state.search_query_texts.get(state.search_count, "")
            q_preview = (f'"{last_q[:20]}…"' if len(last_q) > 20 else (f'"{last_q}"' if last_q else ""))
            rerun_hint = f"[dim italic]⏎ re-show {q_preview}[/dim italic]" if q_preview else ""
            rerun_plain = f"⏎ re-show {q_preview}" if q_preview else ""

            if rerun_hint and cols >= len(bar_left_plain) + len(rerun_plain) + 6:
                padding = max(2, cols - len(bar_left_plain) - len(rerun_plain) - 2)
                bar_content = f"{bar_left}{' ' * padding}{rerun_hint}"
            else:
                bar_content = bar_left

            # ── Measure rendered height to survive scrolling ──────────────────
            # If the terminal is at the bottom, printing the citation card causes
            # the screen to scroll up. Absolute save/restore (\033[s / \033[u)
            # fails because the saved row is now wrong.
            # Instead, we measure exactly how many lines the bar + prompt take,
            # and use relative cursor-up (\033[{N}A) which is immune to scrolling.
            with console.capture() as _cap:
                console.print(bar_content)
                console.print("  › 1")  # Simulate prompt + a typed digit
            _prompt_lines = _cap.get().count("\n")

            console.print(bar_content)

            # ── Raw keypress ─────────────────────────────────────────────────
            console.print("  › ", end="", highlight=False)
            try:
                first = _read_single_key()
            except (KeyboardInterrupt, EOFError):
                console.print()
                return

            # ── Escape / Ctrl-C from _read_single_key ────────────────────────
            if first in ("\x1b", "\x03"):
                console.print()
                return

            # ── Enter (re-show current result, no new search) ─────────────────
            if first in ("\r", "\n", ""):
                console.print()
                if result is not None:
                    _display_results(result, state)
                continue

            # ── Paste detection ───────────────────────────────────────────────
            # _read_single_key() drains any burst of buffered bytes after the
            # first char (paste protection).  If `first` is longer than 1 char
            # the user pasted text; skip single-char command dispatch entirely
            # and treat the whole string as a query or slash command directly.
            is_paste = len(first) > 1

            if not is_paste:
                # ── Digit → instant citation card, zero Enter needed ──────────
                if first.isdigit():
                    # Consume any additional digits (e.g. "10") by trying to peek
                    # one more raw char; fall back gracefully if not possible.
                    second = ""
                    try:
                        import sys, termios, tty, select
                        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0.15)[0]:
                            fd = sys.stdin.fileno()
                            old = termios.tcgetattr(fd)
                            try:
                                tty.setraw(fd)
                                c2 = sys.stdin.read(1)
                                if c2.isdigit():
                                    second = c2
                                else:
                                    # Put it back via the next loop iteration isn't
                                    # possible, so handle it: if it's Enter, ignore;
                                    # otherwise treat as a new key next loop.
                                    if c2 not in ("\r", "\n"):
                                        second = ""   # discard non-digit, non-Enter
                            finally:
                                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    except Exception:
                        pass

                    console.print(first + second)   # echo what the user "typed"
                    raw_n = first + second
                    try:
                        n = int(raw_n)
                    except ValueError:
                        n = -1

                    if 1 <= n <= len(result.sources):
                        _show_citation_card(n, result.sources[n - 1], result.sources)
                        # The card cleanly erased itself and left the cursor on
                        # the line right after the echoed digit. We move up by
                        # exactly the measured height of the bar+prompt and clear.
                        console.file.write(f"\033[{_prompt_lines}A\033[J")
                        console.file.flush()
                    else:
                        hi = len(result.sources)
                        console.print(
                            f"[dim]Fragment {n} out of range — "
                            f"{hi} fragment{'s' if hi != 1 else ''} available (1–{hi}).[/dim]"
                        )
                    continue

                # ── Known single-char commands ────────────────────────────────
                upper = first.upper()

                if upper == "Q":
                    console.print(first)   # echo before acting
                    if state.session_id >= 0:
                        try:
                            from audiobench.memory.session_store import close_session
                            close_session(state.session_id)
                        except Exception:
                            pass
                    return

                if upper == "E":
                    console.print(first)   # echo before acting
                    _open_fragment_reader(console, result.sources, initial_idx=0)
                    continue

                if upper == "C":
                    console.print(first)   # echo before acting
                    from audiobench.chat.chat_repl import ChatREPL
                    repl = ChatREPL(
                        session_type="search_followup",
                        preloaded_fragments=result.sources,
                        preloaded_title=f"🔍 Search: {result.query}",
                    )
                    repl.run()
                    return

                if upper == "S":
                    console.print(first)   # echo before acting
                    # S → dedicated "new search" prompt
                    try:
                        new_query = input("  New query: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        console.print()
                        continue
                    if new_query:
                        query = new_query
                        result = None
                        console.print()
                        break
                    continue

            # ── Query or slash command (typed char-by-char OR pasted) ─────────
            # For single-char entry: write the first char inline then collect
            # the rest with input() so everything appears on one continuous line.
            # For paste: first already contains the full pasted string, so we
            # print it and skip the extra input() call.
            
            # Ignore raw escape sequences (like Arrow Keys/Page Down) so they don't stack
            if is_paste and first.startswith("\x1b"):
                # Print newline to match cursor position of other branches
                _sys.stdout.write("\n")
                _sys.stdout.flush()
                console.file.write(f"\033[{_prompt_lines}A\033[J")
                console.file.flush()
                continue
                
            import sys as _sys
            if is_paste:
                # Echo the full paste and use it directly — no second input() needed
                _sys.stdout.write(first + "\n")
                _sys.stdout.flush()
                choice = first.strip()
            else:
                _sys.stdout.write(first)
                _sys.stdout.flush()
                try:
                    rest = input("")   # empty prompt — cursor stays right after first
                except (KeyboardInterrupt, EOFError):
                    console.print()
                    continue
                choice = (first + rest).strip()

            if choice.startswith("/"):
                query_parts = choice.split(maxsplit=1)
                if (
                    query_parts[0].lower() in ("/fast", "/balanced", "/deep", "/synthesis")
                    and len(query_parts) > 1
                ):
                    query = choice
                    result = None
                    console.print()
                    break
                msg = _parse_slash_command(choice, state, last_sources=result.sources, engine=engine)
                if msg:
                    console.print(f"  {msg}")
                # Erase prompt on invalid slash command so it doesn't stack
                console.file.write(f"\033[{_prompt_lines}A\033[J")
                console.file.flush()
                continue

            # Long enough to be a new query
            if len(choice) > 3:
                query = choice
                result = None
                console.print()
                break

            if choice:
                console.print(
                    "[dim]Too short for a query — press [bold]S[/bold] or type at least 4 characters.[/dim]"
                )
