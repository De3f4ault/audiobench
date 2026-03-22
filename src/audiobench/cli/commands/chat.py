"""Chat + Ask (AI interactive) commands."""

from __future__ import annotations

import click
from rich.live import Live
from rich.text import Text

from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    BOLD,
    CHAT_CODE_THEME,
    DIM,
    PROMPT,
    SUCCESS,
    chat_console,
    console,
    error_panel,
)
from audiobench.core.settings import get_settings

# ── Chat Help Text ──────────────────────────────────────────

CHAT_HELP_TEXT = (
    "  [bold]Slash Commands[/]\n"
    "  ─────────────────────────────────────\n"
    "  /help              Show this help\n"
    "  /context           Show loaded transcripts\n"
    "  /load <ID>         Add a transcript to context\n"
    "  /remove <ID>       Remove a transcript from context\n"
    "  /clear             Clear history and all context\n"
    "  /model <name>      Switch model mid-chat\n"
    "  /think             Toggle thinking display\n"
    "  /retry             Regenerate last response\n"
    "  /export [file]     Export chat to markdown\n"
    "  /history           List past chat sessions\n"
    "  /save              Force-save conversation\n"
    "  /exit              Exit chat (also Ctrl+D)\n"
    "\n"
    "  [bold]Multi-line Input[/]\n"
    "  ─────────────────────────────────────\n"
    '  Type [bold]triple-quotes (\\"\\"\\")'
    "[/] to start/end a multi-line block.\n"
)


# ── Slash Command Handler ───────────────────────────────────


def _handle_slash_command(
    cmd: str,
    session,
    tx_repo,
    chat_repo,
    settings,
) -> bool:
    """Handle a slash command. Returns True if the REPL should exit."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/exit", "/quit", "/q"):
        return True

    elif command == "/help":
        console.print()
        console.print(CHAT_HELP_TEXT)

    elif command == "/context":
        console.print()
        for line in session.get_context_summary():
            console.print(f"    {line}")
        console.print()

    elif command == "/load":
        if not arg or not arg.strip().isdigit():
            console.print(f"  [{DIM}]Usage: /load <transcript_id>[/]")
            return False
        tid = int(arg.strip())
        record = tx_repo.get_by_id(tid)
        if not record:
            console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
            return False
        session.load_transcripts([record])
        console.print(
            f"  [{SUCCESS}]✓ Loaded #{tid} "
            f"{record['file_name']} "
            f"({record['word_count']:,} words)[/]"
        )

    elif command == "/clear":
        session.clear_history()
        console.print(
            f"  [{SUCCESS}]✓ Conversation cleared (new session #{session.conversation_id})[/]"
        )
        console.print(f"  [{DIM}]Context reset — use /load <ID> to add transcripts[/]")

    elif command == "/remove":
        if not arg or not arg.strip().isdigit():
            console.print(f"  [{DIM}]Usage: /remove <transcript_id>[/]")
            return False
        tid = int(arg.strip())
        if session.remove_transcript(tid):
            console.print(f"  [{SUCCESS}]✓ Removed transcript #{tid} from context[/]")
        else:
            console.print(f"  [{DIM}]Transcript #{tid} not in context[/]")

    elif command == "/model":
        if not arg:
            console.print(f"  [{DIM}]Current model: {session.model}[/]")
            console.print(f"  [{DIM}]Usage: /model <name>[/]")
            return False
        session.switch_model(arg.strip())
        console.print(f"  [{SUCCESS}]✓ Switched to {arg.strip()}[/]")

    elif command == "/think":
        session.show_thinking = not session.show_thinking
        state = "on" if session.show_thinking else "off"
        console.print(f"  [{SUCCESS}]✓ Thinking display: {state}[/]")

    elif command == "/history":
        convs = chat_repo.list_conversations(limit=10)
        if not convs:
            console.print(f"  [{DIM}]No past conversations[/]")
            return False
        console.print()
        for c in convs:
            tid_list = c.get("transcript_ids", [])
            ctx = f" (transcripts: {tid_list})" if tid_list else ""
            console.print(
                f"    [{ACCENT}]#{c['id']}[/] "
                f"{c['title']} "
                f"[{DIM}]({c['message_count']} msgs, "
                f"{c['model']}){ctx}[/]"
            )
        console.print()

    elif command == "/save":
        console.print(f"  [{SUCCESS}]✓ Conversation #{session.conversation_id} saved[/]")

    elif command == "/export":
        import time as _time
        from pathlib import Path

        if not session.messages:
            console.print(f"  [{DIM}]Nothing to export yet[/]")
            return False
        fname = arg.strip() if arg.strip() else None
        if not fname:
            slug = f"chat_{session.conversation_id or 'new'}_{int(_time.time())}"
            fname = f"{slug}.md"
        path = Path(fname).expanduser()
        lines = [f"# Chat #{session.conversation_id or 'new'}\n"]
        lines.append(f"Model: {session.model}  \n")
        lines.append("---\n")
        for msg in session.messages:
            if msg["role"] == "user":
                lines.append(f"**You:** {msg['content']}\n")
            elif msg["role"] == "assistant":
                lines.append(f"**AI:**\n\n{msg['content']}\n")
            lines.append("---\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"  [{SUCCESS}]✓ Exported to {path}[/]")

    elif command == "/retry":
        # Signal to the REPL that we want a retry
        # We store a flag on the session object
        session._retry_requested = True  # noqa: SLF001
        return False  # handled in the REPL loop

    else:
        console.print(f"  [{DIM}]Unknown command: {command} (type /help for commands)[/]")

    return False


# ── Ask Command ─────────────────────────────────────────────


@click.command()
@click.argument("transcript_id", type=int)
@click.argument("question")
@click.option("--model", default=None, help="Ollama model (default: from settings)")
def ask(transcript_id: int, question: str, model: str | None) -> None:
    """Ask a question about a transcript using AI.

    \b
    Examples:
      audiobench ask 3 "What decisions were made?"
      audiobench ask 3 "Who is responsible for the API?"
      audiobench ask 3 "List all mentioned dates" --model deepseek-v3.2
    """
    from audiobench.chat.context_builder import TRANSCRIPT_SYSTEM, qa
    from audiobench.chat.providers.ollama_provider import AIError, OllamaClient
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    settings = get_settings()
    model_name = model or settings.ollama_model

    # Fetch transcript
    init_db()
    repo = TranscriptionRepository()
    record = repo.get_by_id(transcript_id)
    if not record:
        console.print(error_panel("Not found", f"Transcript #{transcript_id} not found"))
        return

    console.print()
    console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Q&A")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Source:   [{ACCENT}]#{transcript_id} {record['file_name']}[/]")
    console.print(f"    Question: {question}")
    console.print(f"    Model:    {model_name}")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()

    prompt = qa(record["full_text"], question)

    try:
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=model_name,
        )

        if not client.is_available():
            console.print(
                error_panel(
                    "Ollama not running",
                    "Start with: ollama serve",
                )
            )
            return

        for token in client.stream(prompt, system_prompt=TRANSCRIPT_SYSTEM):
            console.print(token, end="")

        console.print()
        console.print()

    except AIError as e:
        console.print(error_panel("AI Error", str(e)))


# ── Chat Command ────────────────────────────────────────────


@click.command()
@click.argument("transcript_ids", nargs=-1, type=int)
@click.option(
    "--model",
    default=None,
    help="Ollama model (default: from settings)",
)
@click.option(
    "--temperature",
    default=0.3,
    type=float,
    help="Creativity level (0.0-1.0)",
)
@click.option(
    "--search",
    "search_query",
    default=None,
    help="Load transcripts matching this search",
)
@click.option(
    "--recent",
    default=None,
    type=int,
    help="Load N most recent transcripts as context",
)
@click.option(
    "--resume",
    "resume_id",
    default=None,
    type=int,
    help="Resume a previous conversation by ID",
)
@click.option(
    "--list",
    "list_chats",
    is_flag=True,
    help="List past chat conversations",
)
@click.option(
    "--delete",
    "delete_id",
    default=None,
    type=int,
    help="Delete a chat conversation by ID",
)
@click.option(
    "--think/--no-think",
    default=True,
    help="Show/hide model chain-of-thought",
)
def chat(
    transcript_ids: tuple[int, ...],
    model: str | None,
    temperature: float,
    search_query: str | None,
    recent: int | None,
    resume_id: int | None,
    list_chats: bool,
    delete_id: int | None,
    think: bool,
) -> None:
    """Interactive AI chat with transcript context.

    \b
    Examples:
      audiobench chat                           Chat freely
      audiobench chat 3                         Chat about transcript #3
      audiobench chat 3 5 7                     Chat with multiple transcripts
      audiobench chat --search "meeting"        Load matching transcripts
      audiobench chat --recent 5                Load 5 most recent
      audiobench chat --resume 2                Resume conversation #2
      audiobench chat --list                    List past conversations
      audiobench chat --delete 2                Delete conversation #2
      audiobench chat --model deepseek-v3.1:671b-cloud
    """
    from audiobench.chat.chat_session import ChatSession
    from audiobench.chat.chat_store import ChatRepository
    from audiobench.chat.providers.ollama_provider import AIError, OllamaClient
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    settings = get_settings()
    model_name = model or settings.ollama_model
    init_db()

    chat_repo = ChatRepository()
    tx_repo = TranscriptionRepository()

    # ── Handle --list ──
    if list_chats:
        convs = chat_repo.list_conversations(limit=20)
        if not convs:
            console.print(f"  [{DIM}]No chat conversations yet[/]")
            return
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Chat History")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        for c in convs:
            tid_list = c.get("transcript_ids", [])
            ctx = f" ctx:{tid_list}" if tid_list else ""
            console.print(
                f"    [{ACCENT}]#{c['id']}[/] "
                f"{c['title']} "
                f"[{DIM}]({c['message_count']} msgs"
                f"{ctx})[/]"
            )
        console.print()
        console.print(f"  [{DIM}]Resume with: audiobench chat --resume <ID>[/]")
        console.print()
        return

    # ── Handle --delete ──
    if delete_id is not None:
        if chat_repo.delete_conversation(delete_id):
            console.print(f"  [{SUCCESS}]✓ Deleted conversation #{delete_id}[/]")
        else:
            console.print(
                error_panel(
                    "Not found",
                    f"Conversation #{delete_id} not found",
                )
            )
        return

    # ── Check Ollama ──
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=model_name,
    )
    if not client.is_available():
        console.print(
            error_panel(
                "Ollama not running",
                f"Start with: ollama serve\nPull model: ollama pull {model_name}",
            )
        )
        return

    # ── Create or resume session ──
    session = ChatSession(
        client=client,
        chat_repo=chat_repo,
        model=model_name,
        temperature=temperature,
        conversation_id=resume_id,
        show_thinking=think,
    )

    # Resume existing conversation
    if resume_id is not None and not session.restore_from_db():
        console.print(
            error_panel(
                "Not found",
                f"Conversation #{resume_id} not found",
            )
        )
        return

    # ── Load transcript context ──
    transcripts_to_load = []

    # By explicit IDs
    for tid in transcript_ids:
        record = tx_repo.get_by_id(tid)
        if record:
            transcripts_to_load.append(record)
        else:
            console.print(f"  [{DIM}]Transcript #{tid} not found, skipping[/]")

    # By search
    if search_query:
        results = tx_repo.search(search_query, limit=5)
        for r in results:
            full = tx_repo.get_by_id(r["id"])
            if full:
                transcripts_to_load.append(full)
        if not results:
            console.print(f"  [{DIM}]No transcripts matching '{search_query}'[/]")

    # By recent
    if recent:
        history_items = tx_repo.get_history(limit=recent)
        for h in history_items:
            full = tx_repo.get_by_id(h["id"])
            if full:
                transcripts_to_load.append(full)

    if transcripts_to_load:
        session.load_transcripts(transcripts_to_load)

    # ── Header ──
    console.print()
    conv_label = f" [#{resume_id}]" if resume_id else ""
    console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Chat{conv_label}")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Model:    {model_name}")
    ctx_lines = session.get_context_summary()
    console.print(f"    Context:  {ctx_lines[0]}")
    for line in ctx_lines[1:]:
        console.print(f"              {line}")
    think_label = "on" if think else "off"
    console.print(f"    Thinking: {think_label}")
    if resume_id and session.turn_count > 0:
        console.print(f"    Resumed:  {session.turn_count} previous turn(s)")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()

    # ── Render past messages on resume ──
    import contextlib
    import readline as _readline
    import time as _time
    from pathlib import Path as _Path

    from rich.console import Group
    from rich.markdown import Markdown as RichMarkdown
    from rich.padding import Padding

    # ── Readline history setup ──
    _history_file = _Path.home() / ".cache" / "audiobench_chat_history"
    _history_file.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        _readline.read_history_file(str(_history_file))
    _readline.set_history_length(500)

    def _save_readline_history() -> None:
        with contextlib.suppress(OSError):
            _readline.write_history_file(str(_history_file))

    # ── Render past messages on resume ──
    if resume_id and session.messages:
        console.print(f"  [{DIM}]─── Previous Messages ───[/]")
        console.print()
        for msg in session.messages:
            if msg["role"] == "user":
                console.print(f"  [{PROMPT}]>>> {msg['content']}[/]")
                console.print()
            elif msg["role"] == "assistant" and msg["content"].strip():
                md = RichMarkdown(
                    msg["content"],
                    code_theme=CHAT_CODE_THEME,
                )
                chat_console.print(Padding(md, (0, 2, 1, 2)))
                console.print()
        console.print(f"  [{DIM}]─── End of History ───[/]")
        console.print()

    # ── Helper: stream a message and render ──
    def _stream_and_render(user_text: str) -> None:
        """Send user input and render the streamed response.

        Uses a compact tail-preview in Rich Live during streaming to
        avoid the scrollback-duplication problem that occurs when Live
        content exceeds the terminal viewport height.  The full formatted
        markdown is printed once after streaming completes.
        """
        console.print()
        try:
            thinking_parts: list[str] = []
            content_parts: list[str] = []
            token_count = 0
            t_start = _time.monotonic()

            with Live(
                console=chat_console,
                refresh_per_second=8,
                transient=True,
            ) as live:
                for chunk in session.send(user_text):
                    thinking = chunk.get("thinking", "")
                    content = chunk.get("content", "")

                    if thinking:
                        thinking_parts.append(thinking)

                    if content:
                        content_parts.append(content)

                    if content:
                        token_count += 1

                    # ── Compact streaming preview ──
                    # Only show the *tail* of thinking/content so the Live
                    # viewport never exceeds the terminal height.  This
                    # prevents Rich from pushing lines into the permanent
                    # scrollback buffer where they can't be erased.
                    display_parts = []

                    if thinking_parts and session.show_thinking:
                        think_text = "".join(thinking_parts)
                        think_lines = think_text.splitlines()
                        if len(think_lines) > 5:
                            think_text = "…\n" + "\n".join(think_lines[-5:])
                        display_parts.append(
                            Text(f"💭 {think_text}", style="dim italic"),
                        )

                    if content_parts:
                        full_text = "".join(content_parts)
                        preview_lines = full_text.splitlines()
                        if len(preview_lines) > 8:
                            preview = "\n".join(preview_lines[-8:])
                            display_parts.append(
                                Text("  ⋮\n", style="dim"),
                            )
                        else:
                            preview = full_text
                        display_parts.append(Text(preview))
                        # Token counter only during content streaming
                        elapsed_so_far = _time.monotonic() - t_start
                        tps_so_far = token_count / elapsed_so_far if elapsed_so_far > 0 else 0
                        display_parts.append(
                            Text(
                                f"\n  ▍ {token_count} tokens · {tps_so_far:.0f} tok/s",
                                style="dim",
                            ),
                        )

                    if display_parts:
                        live.update(Group(*display_parts))

            # ── Print full formatted content ──
            # Live was transient so its viewport is erased; we now
            # print the complete markdown-rendered response once.
            if content_parts:
                final_md = "".join(content_parts)
                chat_console.print(
                    Padding(
                        RichMarkdown(final_md, code_theme=CHAT_CODE_THEME),
                        (0, 0, 0, 0),
                    )
                )
            elif thinking_parts:
                # Some models (e.g. deepseek) return the entire response
                # in the "thinking" field with empty "content". Display
                # the thinking text as the response in that case.
                final_md = "".join(thinking_parts)
                chat_console.print(
                    Padding(
                        RichMarkdown(final_md, code_theme=CHAT_CODE_THEME),
                        (0, 0, 0, 0),
                    )
                )

            # Persist response + background title gen (non-blocking)
            session.finalize_response()

            # Token stats
            elapsed = _time.monotonic() - t_start
            if token_count > 0 and elapsed > 0:
                tps = token_count / elapsed
                console.print(
                    f"  [{DIM}]{token_count} tokens · {tps:.1f} tok/s · {elapsed:.1f}s[/]"
                )
            console.print()

        except KeyboardInterrupt:
            # Save partial response if anything was generated
            if content_parts:
                session.finalize_response()
            console.print()
            console.print(f"  [{DIM}]Generation interrupted[/]")
            console.print()

        except AIError as e:
            console.print(error_panel("AI Error", str(e)))
            console.print()

    # ── Multi-line input helper ──
    def _read_multiline() -> str:
        """Read lines until closing triple-quotes."""
        lines: list[str] = []
        console.print(f'  [{DIM}]Multi-line mode (type """ to end):[/]')
        while True:
            try:
                line = input("... ")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == '"""':
                break
            lines.append(line)
        return "\n".join(lines)

    # ── Interactive REPL ──
    last_user_input: str | None = None
    session._retry_requested = False  # noqa: SLF001

    while True:
        try:
            user_input = input("\001\033[38;5;48m\002>>> ").strip()
            print("\033[0m", end="", flush=True)
        except (EOFError, KeyboardInterrupt):
            console.print()
            _save_readline_history()
            if session.conversation_id:
                console.print(
                    f"  [{SUCCESS}]✓ Conversation "
                    f"#{session.conversation_id} saved "
                    f"({session.turn_count * 2} messages)[/]"
                )
            console.print(f"  [{DIM}]Goodbye![/]")
            console.print()
            break

        if not user_input:
            continue

        # Multi-line input
        if user_input == '"""':
            user_input = _read_multiline()
            if not user_input.strip():
                continue

        # Slash commands (accept both / and \)
        if user_input.startswith("\\"):
            user_input = "/" + user_input[1:]
        if user_input.startswith("/"):
            should_exit = _handle_slash_command(
                user_input,
                session,
                tx_repo,
                chat_repo,
                settings,
            )

            # Handle /retry
            if getattr(session, "_retry_requested", False):
                session._retry_requested = False  # noqa: SLF001
                if last_user_input and session.messages:
                    # Remove last assistant + user message
                    session._messages = [m for m in session._messages if m != session._messages[-1]]
                    if session._messages and session._messages[-1]["role"] == "user":
                        session._messages.pop()
                    console.print(f"  [{DIM}]Regenerating...[/]")
                    _stream_and_render(last_user_input)
                else:
                    console.print(f"  [{DIM}]Nothing to retry[/]")
                continue

            if should_exit:
                _save_readline_history()
                if session.conversation_id:
                    console.print(
                        f"  [{SUCCESS}]✓ Conversation "
                        f"#{session.conversation_id} saved "
                        f"({session.turn_count * 2} messages)"
                        f"[/]"
                    )
                console.print(f"  [{DIM}]Goodbye![/]")
                console.print()
                break
            continue

        last_user_input = user_input
        _stream_and_render(user_input)
