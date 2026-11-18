import json
from pathlib import Path

import click
from rich.console import Console

from audiobench.cli.display.theme import ACCENT, APP_NAME, BOLD, DIM
from audiobench.storage.repository import TranscriptionRepository

console = Console()


def _spawn_janitor():
    """Spawn a background thread to silently rename any files tagged with 'pending_auto_rename'."""
    import threading

    def _janitor_sweep():
        import json
        import re
        import time

        from audiobench.core.db_session import get_session
        from audiobench.core.logger_factory import get_logger
        from audiobench.core.settings import get_settings
        from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

        logger = get_logger("core.janitor")

        # Wait a moment so we don't delay TUI startup
        time.sleep(1.0)

        settings = get_settings()

        with get_session() as session:
            all_records = session.query(AudioFileRecord).all()
            pending_records = []
            for record in all_records:
                try:
                    data = json.loads(record.tags) if record.tags else []
                    labels = data.get("_labels", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    if "pending_auto_rename" in labels:
                        pending_records.append(record)
                except Exception:
                    pass

            if not pending_records:
                return

            logger.info("Janitor found %d files pending auto-rename", len(pending_records))

            gemini_client = None
            from google import genai

            from audiobench.chat.providers.ollama_provider import OllamaClient

            if settings.gemini_api_key:
                gemini_client = genai.Client(api_key=settings.gemini_api_key)

            ollama = OllamaClient(base_url=settings.ollama_base_url, model=settings.clean_model)

            for audio_record in pending_records:
                try:
                    tx_record = (
                        session.query(TranscriptionRecord)
                        .filter_by(audio_file_id=audio_record.id)
                        .order_by(TranscriptionRecord.id.desc())
                        .first()
                    )
                    if not tx_record or not tx_record.segments:
                        continue

                    segments = sorted(tx_record.segments, key=lambda s: s.start_time)
                    intro_segments = [s for s in segments if s.end_time <= 300][:50]
                    intro_text = " ".join([s.text for s in intro_segments])

                    if not intro_text.strip():
                        continue

                    prompt = (
                        "Based on the following transcript excerpt, generate a concise, highly semantic 3-5 word title for this audio file.\n"
                        "Do NOT use quotes, special characters, or prefixes like 'Title:'. Just the raw title text.\n\n"
                        f"{intro_text}"
                    )

                    new_title = ""
                    try:
                        if gemini_client:
                            response = gemini_client.models.generate_content(
                                model="gemini-2.5-pro", contents=prompt
                            )
                            new_title = response.text.strip()
                        else:
                            raise ValueError("No Gemini API key")
                    except Exception:
                        if ollama.is_available():
                            response = ollama.chat(
                                [{"role": "user", "content": prompt}], think=False
                            )
                            new_title = response.get("content", "").strip()

                    if not new_title:
                        continue  # Still offline, keep the tag for next time

                    new_title = re.sub(r"[^\w\s-]", " ", new_title).strip()
                    new_title = re.sub(r"\s+", " ", new_title)

                    if not new_title or len(new_title.split()) > 10:
                        continue

                    old_path = Path(audio_record.file_path)
                    if not old_path.exists():
                        continue

                    new_filename = f"{new_title}{old_path.suffix}"
                    new_path = old_path.parent / new_filename

                    counter = 1
                    while new_path.exists() and new_path != old_path:
                        new_path = old_path.parent / f"{new_title}_{counter}{old_path.suffix}"
                        new_filename = new_path.name
                        counter += 1

                    if new_path != old_path:
                        old_path.rename(new_path)
                        audio_record.file_name = new_filename
                        audio_record.file_path = str(new_path)

                        try:
                            data = json.loads(audio_record.tags) if audio_record.tags else []
                        except Exception:
                            data = []
                        if isinstance(data, dict):
                            labels = data.get("_labels", [])
                            if "pending_auto_rename" in labels:
                                labels.remove("pending_auto_rename")
                                audio_record.tags = json.dumps(data)
                        elif isinstance(data, list):
                            if "pending_auto_rename" in data:
                                data.remove("pending_auto_rename")
                                audio_record.tags = json.dumps(data)

                        session.commit()
                        logger.info("Janitor auto-renamed file to %s", new_filename)

                except Exception as e:
                    logger.warning("Janitor failed on record %d: %s", audio_record.id, e)

    threading.Thread(target=_janitor_sweep, name="janitor", daemon=True).start()


from audiobench.cli.plugins.custom_group import DefaultGroup


@click.group(name="library", cls=DefaultGroup, default_command="center", invoke_without_command=True)
@click.pass_context
def library_group(ctx):
    """Launch the interactive Library Command Center or manage library upgrades."""
    if ctx.invoked_subcommand is None or ctx.invoked_subcommand == "center":
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Library Command Center")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        from audiobench.cli.repl.session import ReplSession
        session = ReplSession(ctx.parent.command if ctx.parent else ctx.command)
        run_library(session)


@library_group.command(name="center", hidden=True)
@click.pass_context
def library_center(ctx):
    """Internal default subcommand to launch the library command center."""
    console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Library Command Center")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    from audiobench.cli.repl.session import ReplSession
    session = ReplSession(ctx.parent.command if ctx.parent else ctx.command)
    run_library(session)


# Backward-compatible alias
library_cmd = library_group

def run_library(session, restore_state=None):
    _spawn_janitor()

    from audiobench.cli.tui.library_tui import launch_library_tui

    result = launch_library_tui(restore_state)

    action = result.get("action")
    selected_ids = result.get("selected_ids", [])
    tab = result.get("tab")
    state_export = result.get("state", {})

    if action == "switch_to_import":
        from audiobench.cli.repl.dispatch import dispatch_command
        from audiobench.cli.repl.session import NavigationFrame
        session.push_frame(NavigationFrame(
            context="library",
            state=state_export,
            intent="importing OS files"
        ))
        dispatch_command(session, ["import"])
        return

    if not action or (not selected_ids and action != "switch_to_import"):
        console.print(f"\n  [{DIM}]Exited library command center.[/]")
        return

    repo = TranscriptionRepository()

    if action == "delete":
        # Placeholder for delete logic
        console.print(f"  [{DIM}]Deleted {len(selected_ids)} items.[/]")
        return

    if action == "transcribe" and tab == "audio":
        import questionary

        batch_engine = questionary.select(
            f"Choose transcription engine for {len(selected_ids)} file(s):",
            choices=["auto (use pre-assigned)", "gemini", "whisper", "cancel"],
            default="auto (use pre-assigned)",
        ).ask()

        if not batch_engine or batch_engine == "cancel":
            console.print(f"\n  [{DIM}]Transcription cancelled.[/]")
            return

        console.print(
            f"\n  [{BOLD} {ACCENT}]Preparing to transcribe {len(selected_ids)} audio files![/]"
        )

        # Invoke the transcribe command via subprocess for each file.
        for audio_id in selected_ids:
            audio_record = repo.get_audio_file(audio_id)
            if audio_record:
                # Try to parse the engine and strategy preference from tags if it exists
                tags_str = audio_record.get("tags", "[]")
                engine = "gemini"  # default
                strategy = "batch" # default
                try:
                    tags = json.loads(tags_str)
                    for tag in tags:
                        if tag.startswith("engine_preference:"):
                            engine = tag.split(":")[1].strip()
                        elif tag.startswith("strategy_preference:"):
                            strategy = tag.split(":")[1].strip()
                except Exception:
                    pass

                if batch_engine != "auto (use pre-assigned)":
                    engine = batch_engine

                # Use subprocess to invoke the CLI command safely and avoid Click internal crashes
                import subprocess

                try:
                    console.print(
                        f"  [{DIM}]Transcribing {audio_record['file_name']} with {engine} ({strategy} mode)...[/]"
                    )
                    # Flush console so output is in order
                    import sys

                    sys.stdout.flush()

                    cmd = [
                        "audiobench",
                        "transcribe",
                        audio_record["file_path"],
                        "--engine",
                        engine,
                        "--strategy",
                        strategy,
                    ]
                    subprocess.run(cmd)
                except Exception as e:
                    console.print(f"  [red]Error transcribing {audio_record['file_name']}: {e}[/]")

        console.print(f"  [{BOLD} {ACCENT}]Transcription batch complete![/]")
