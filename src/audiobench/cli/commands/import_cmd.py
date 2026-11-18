from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TaskID, TextColumn

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, error_panel

console = Console()


def extract_metadata(file_path: Path) -> tuple[str | None, str | None]:
    title, author = None, None
    try:
        import mutagen
        m = mutagen.File(file_path, easy=True)
        if m:
            title = m.get("title", [None])[0]
            if not title:
                title = m.get("album", [None])[0]
            author = m.get("artist", [None])[0]
            if not author:
                author = m.get("albumartist", [None])[0]
    except ImportError:
        pass
    except Exception:
        pass

    if not title:
        name = file_path.stem
        if " - " in name:
            parts = name.split(" - ", 1)
            if len(parts) >= 2:
                author = parts[0].strip()
                title = parts[1].strip()
    return title, author


def normalize_stem(stem: str) -> str:
    """Normalize file stem for fuzzy pairing."""
    import re

    return re.sub(r"[\W_]+", " ", stem).strip().lower()


def token_similarity(a: str, b: str) -> float:
    """Compute token set overlap ratio between two strings."""
    import re

    words_a = set(re.sub(r"[\W_]+", " ", a).lower().split())
    words_b = set(re.sub(r"[\W_]+", " ", b).lower().split())
    if not words_a or not words_b:
        return 0.0
    common = words_a & words_b
    return len(common) / min(len(words_a), len(words_b))


def find_companion_transcript(audio_file: Path) -> tuple[Path | None, float]:
    """Find companion transcript file (.json, .srt, .vtt) by exact name, subfolder, or fuzzy token match."""
    # 1. Exact stem matches in parent directory
    for ext in [".json", ".srt", ".vtt"]:
        candidate = audio_file.with_suffix(ext)
        if candidate.exists() and candidate.is_file():
            return candidate, 1.0
        candidate2 = audio_file.parent / (audio_file.name + ext)
        if candidate2.exists() and candidate2.is_file():
            return candidate2, 1.0

    # 2. Search common subfolders (transcripts, subs, etc.)
    search_dirs = [audio_file.parent]
    for sub in ["transcripts", "transcriptions", "subs", "subtitles"]:
        sub_dir = audio_file.parent / sub
        if sub_dir.exists() and sub_dir.is_dir():
            search_dirs.append(sub_dir)

    best_match: Path | None = None
    best_score: float = 0.0
    stem_norm = normalize_stem(audio_file.stem)

    for d in search_dirs:
        for ext in [".json", ".srt", ".vtt"]:
            for f in d.glob(f"*{ext}"):
                if not f.is_file():
                    continue
                f_norm = normalize_stem(f.stem)
                if f_norm == stem_norm:
                    return f, 1.0
                score = token_similarity(audio_file.stem, f.stem)
                if score > best_score and score >= 0.70:
                    best_score = score
                    best_match = f

    return best_match, best_score


def run_post_import_launchpad(
    imported_ids: list[int],
    results: list[dict],
    interactive: bool = True,
) -> None:
    """Actionable post-import launchpad: summarize imported files and prompt next steps."""
    if not imported_ids:
        return

    from audiobench.cli.display.theme import SUCCESS, format_size, make_table
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

    table = make_table(
        "Imported Files",
        [
            ("ID", {"justify": "right", "style": "bold"}),
            ("Filename", {"style": ACCENT}),
            ("Size", {"justify": "right"}),
            ("Engine", {"style": "cyan"}),
            ("Transcript", {"style": "green"}),
        ],
    )

    tx_map = {}
    records = []
    with get_session() as session:
        records = session.query(AudioFileRecord).filter(AudioFileRecord.id.in_(imported_ids)).all()
        rec_map = {r.id: r for r in records}

        # Check existing transcriptions for these files
        tx_records = session.query(TranscriptionRecord).filter(
            TranscriptionRecord.audio_file_id.in_(imported_ids)
        ).all()
        tx_map = {t.audio_file_id: t for t in tx_records}

        for aid in imported_ids:
            rec = rec_map.get(aid)
            if not rec:
                continue
            eng_str = "gemini"
            if rec.tags:
                try:
                    import json
                    tags = json.loads(rec.tags) if isinstance(rec.tags, str) and rec.tags.startswith("[") else []
                    for t in tags:
                        if t.startswith("engine_preference: "):
                            eng_str = t.split("engine_preference: ")[1]
                except Exception:
                    pass

            tx_rec = tx_map.get(aid)
            if tx_rec:
                status_str = f"✓ Ready (#{tx_rec.id})"
            else:
                status_str = f"[{DIM}]Pending[/]"

            table.add_row(
                str(rec.id),
                rec.file_name,
                format_size(rec.file_size_bytes or 0),
                eng_str,
                status_str,
            )

    console.print()
    console.print(table)

    if not interactive:
        console.print(f"  [{DIM}]Tip: Run 'audiobench transcribe' or 'audiobench info <id>' to proceed.[/]\n")
        return

    try:
        import questionary
    except ImportError:
        return

    pending_ids = [aid for aid in imported_ids if aid not in tx_map]

    choices = []
    if pending_ids:
        choices.append(
            questionary.Choice(
                title=f"🚀 Transcribe Now (Enqueue {len(pending_ids)} pending file(s))",
                value="transcribe",
            )
        )
    choices.extend([
        questionary.Choice(
            title=f"🛒 Add to Staging Cart ({len(imported_ids)} file(s))",
            value="cart",
        ),
        questionary.Choice(
            title="🔍 Inspect Dossier (View 360° relations & metadata)",
            value="dossier",
        ),
        questionary.Choice(
            title="✓ Done (Return to shell)",
            value="done",
        ),
    ])

    ans = questionary.select(
        "What would you like to do next?",
        choices=choices,
    ).ask()

    if ans == "transcribe" and pending_ids:
        from datetime import UTC, datetime
        from audiobench.jobs.scheduler import enqueue

        batch_id = f"import_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        queued = 0
        with get_session() as s:
            for idx, aid in enumerate(pending_ids):
                rec = s.query(AudioFileRecord).get(aid)
                if not rec:
                    continue
                args = ["transcribe", "--audio-id", str(aid)]
                eng = "gemini"
                if rec.tags:
                    try:
                        import json
                        tags = json.loads(rec.tags) if isinstance(rec.tags, str) and rec.tags.startswith("[") else []
                        for t in tags:
                            if t.startswith("engine_preference: "):
                                eng = t.split("engine_preference: ")[1]
                                if not eng.startswith("companion"):
                                    args.extend(["--engine", eng])
                            elif t.startswith("model_preference: "):
                                mod = t.split("model_preference: ")[1]
                                args.extend(["--model", mod])
                    except Exception:
                        pass
                enqueue(
                    job_type="transcribe",
                    args=args,
                    slot="transcription",
                    file_label=rec.file_name,
                    batch_id=batch_id,
                    batch_label="Import Launchpad",
                    batch_index=idx + 1,
                    batch_total=len(pending_ids),
                )
                queued += 1
        console.print(f"\n  [{BOLD}{SUCCESS}]✓ Enqueued {queued} transcription job(s) in background![/]")
        console.print(f"  [{DIM}]Monitor progress with: audiobench jobs (or audiobench daemon status)[/]\n")

    elif ans == "cart":
        from audiobench.storage.models import StagingCartItem
        added = 0
        with get_session() as s:
            for aid in imported_ids:
                if not s.query(StagingCartItem).filter_by(audio_file_id=aid).first():
                    rec = s.query(AudioFileRecord).get(aid)
                    eng = "gemini"
                    mod = "large-v3-turbo"
                    if rec and rec.tags:
                        try:
                            import json
                            tags = json.loads(rec.tags) if isinstance(rec.tags, str) and rec.tags.startswith("[") else []
                            for t in tags:
                                if t.startswith("engine_preference: "):
                                    val = t.split("engine_preference: ")[1]
                                    if not val.startswith("companion"):
                                        eng = val
                                elif t.startswith("model_preference: "):
                                    mod = t.split("model_preference: ")[1]
                        except Exception:
                            pass
                    s.add(StagingCartItem(audio_file_id=aid, engine=eng, model_name=mod))
                    added += 1
            s.commit()
        console.print(f"\n  [{BOLD}{SUCCESS}]✓ Added {added} file(s) to Staging Cart![/]")
        console.print(f"  [{DIM}]Review cart with: audiobench cart (or audiobench transcribe --cart)[/]\n")

    elif ans == "dossier":
        from audiobench.cli.commands.info_cmd import render_file_dossier
        target_id = imported_ids[0]
        if len(imported_ids) > 1 and records:
            item_choices = [
                questionary.Choice(
                    title=f"#{r.id} - {r.file_name}",
                    value=r.id,
                )
                for r in records
            ]
            picked = questionary.select("Select file to inspect:", choices=item_choices).ask()
            if picked:
                target_id = picked
        console.print()
        render_file_dossier(str(target_id), console, interactive=False)


def run_import_flow(
    session=None,
    restore_state=None,
    initial_mode="MODE_AUDIO",
    override_title=None,
    override_author=None,
    target_files: list[Path] | None = None,
    default_engine: str | None = None,
    default_model: str | None = None,
    interactive: bool = True,
) -> list[int]:
    """Run the import flow and return a list of newly imported or updated audio_file IDs."""
    try:
        import questionary
    except ImportError:
        console.print(
            error_panel("Dependencies missing", "Please install questionary and prompt_toolkit")
        )
        return []

    from audiobench.cli.tui.import_tui import launch_file_manager
    from audiobench.cli.tui.reimport_tui import ReimportTUI
    from audiobench.cli.wizard import prompt_menu

    current_mode = initial_mode
    shared_state = restore_state or {}

    # Check if we should show the main menu
    if current_mode == "main_menu":
        options = [
            ("Audio", "Import raw audio files", "MODE_AUDIO"),
            ("Auto-Detect", "Intelligently scan and pair transcripts/audio", "MODE_AUTO"),
            ("Manual Pair", "Manually link a single transcript to an audio file", "MODE_SINGLE_TX"),
            ("Batch Folders", "Map a folder of transcripts to a folder of audio", "MODE_BATCH_TX"),
            ("Exit", "Cancel and exit", "exit")
        ]
        current_mode = prompt_menu("Select Import Mode", options, default_idx=0)
        if not current_mode or current_mode == "exit":
            return []

    # Routing Loop
    while current_mode:
        if current_mode == "MODE_AUDIO":
            if target_files:
                selected_files = list(target_files)
            else:
                selected_files, new_state = launch_file_manager(start_state=shared_state)
                shared_state.update(new_state)

                if isinstance(selected_files, str) and selected_files.startswith("MODE_"):
                    current_mode = selected_files
                    continue

                if not selected_files:
                    console.print(f"\n  [{DIM}]Import cancelled or no files selected.[/]")
                    return []

            # Break out of loop to process audio files below
            break

        elif current_mode in ("MODE_AUTO", "MODE_SINGLE_TX", "MODE_BATCH_TX"):
            is_auto = (current_mode == "MODE_AUTO")
            is_batch = (current_mode == "MODE_BATCH_TX")

            tui = ReimportTUI(batch=is_batch, auto_detect=is_auto, shared_state=shared_state)
            result = tui.run()
            shared_state.update(tui.state_export())

            if isinstance(result, str) and result.startswith("MODE_"):
                current_mode = result
                continue

            if not result:
                console.print(f"\n  [{DIM}]Transcript import cancelled.[/]")
            return []

        else:
            break

    if session and getattr(session, "navigation_stack", None) and session.navigation_stack[-1].context == "import":
        session.navigation_stack[-1].state = shared_state
        session._persist_stack()

    # Sort files alphabetically for the allocation wizard
    selected_files.sort(key=lambda x: x.name.lower())

    # 3. Engine Allocation Wizard
    total_files = len(selected_files)
    console.print(f"\n  [{DIM}]Scanning companion transcripts for {total_files} file(s)...[/]")

    file_companions: dict[Path, tuple[Path, Any, Any] | None] = {}
    for file in selected_files:
        companion_cand, sim_score = find_companion_transcript(file)
        if companion_cand:
            try:
                from audiobench.transcribe.quality_evaluator import evaluate_transcript_quality
                from audiobench.transcribe.transcript_parser import parse_transcript_file

                parsed_tx, detected_fmt = parse_transcript_file(companion_cand)
                if parsed_tx and (parsed_tx.segments or parsed_tx.text):
                    tqs = evaluate_transcript_quality(
                        segments=parsed_tx.segments,
                        total_duration=parsed_tx.duration_seconds,
                        engine=parsed_tx.engine,
                        full_text=parsed_tx.text,
                    )
                    file_companions[file] = (companion_cand, parsed_tx, tqs)
                else:
                    file_companions[file] = None
            except Exception:
                file_companions[file] = None
        else:
            file_companions[file] = None

    companion_count = sum(1 for v in file_companions.values() if v is not None)

    # Determine allocation strategy
    chosen_strategy = None
    if default_engine:
        chosen_strategy = default_engine.lower()
    elif not interactive:
        chosen_strategy = "hybrid"
    elif total_files > 1:
        # 1-Click Batch Allocation Strategy Menu
        console.print(f"\n  [{BOLD}{ACCENT}]Batch Engine Allocation ({total_files} files selected):[/]")
        if companion_count > 0:
            console.print(f"  [{DIM}]🔍 Found companion transcripts for {companion_count}/{total_files} files.[/]")

        batch_choices = [
            questionary.Choice(
                title=f"🚀 Hybrid (Use companions for {companion_count}/{total_files} files, transcribe rest with Gemini)",
                value="hybrid_gemini",
            ),
            questionary.Choice(
                title=f"⚡ Hybrid (Use companions for {companion_count}/{total_files} files, transcribe rest with Whisper)",
                value="hybrid_whisper",
            ),
            questionary.Choice(
                title=f"🌐 Transcribe All with Gemini ({total_files} files)",
                value="all_gemini",
            ),
            questionary.Choice(
                title=f"💻 Transcribe All with Whisper ({total_files} files)",
                value="all_whisper",
            ),
        ]
        if companion_count > 0:
            batch_choices.append(
                questionary.Choice(
                    title=f"📄 Companion Transcripts Only ({companion_count} files, skip untranscribed)",
                    value="companion_only",
                )
            )
        batch_choices.append(
            questionary.Choice(
                title="⚙️ Custom / Per-File Allocation (Configure each file individually)",
                value="custom",
            )
        )

        chosen_strategy = questionary.select(
            "Select allocation strategy for this batch:",
            choices=batch_choices,
        ).ask()

        if not chosen_strategy:
            console.print(f"  [{DIM}]Import cancelled.[/]")
            return []

    allocations = []
    if chosen_strategy and chosen_strategy != "custom":
        for file in selected_files:
            c_info = file_companions.get(file)
            alloc_dict = {"file": file}
            if default_model:
                alloc_dict["model"] = default_model

            if chosen_strategy in ("hybrid", "auto", "hybrid_gemini"):
                if c_info:
                    c_path, c_parsed, c_tqs = c_info
                    alloc_dict.update({
                        "engine": f"companion:{c_parsed.engine or 'json'}",
                        "companion_file": c_path,
                        "tqs_score": f"{c_tqs.total_score:.1f}",
                    })
                else:
                    alloc_dict["engine"] = "gemini"
                allocations.append(alloc_dict)

            elif chosen_strategy == "hybrid_whisper":
                if c_info:
                    c_path, c_parsed, c_tqs = c_info
                    alloc_dict.update({
                        "engine": f"companion:{c_parsed.engine or 'json'}",
                        "companion_file": c_path,
                        "tqs_score": f"{c_tqs.total_score:.1f}",
                    })
                else:
                    alloc_dict["engine"] = "whisper"
                allocations.append(alloc_dict)

            elif chosen_strategy in ("all_gemini", "gemini"):
                alloc_dict["engine"] = "gemini"
                allocations.append(alloc_dict)

            elif chosen_strategy in ("all_whisper", "whisper"):
                alloc_dict["engine"] = "whisper"
                allocations.append(alloc_dict)

            elif chosen_strategy == "companion_only":
                if c_info:
                    c_path, c_parsed, c_tqs = c_info
                    alloc_dict.update({
                        "engine": f"companion:{c_parsed.engine or 'json'}",
                        "companion_file": c_path,
                        "tqs_score": f"{c_tqs.total_score:.1f}",
                    })
                    allocations.append(alloc_dict)
    else:
        # Per-file interactive allocation (single file or chosen "custom")
        for file in selected_files:
            c_info = file_companions.get(file)
            alloc_dict = {"file": file}
            if default_model:
                alloc_dict["model"] = default_model

            if c_info:
                c_path, c_parsed, c_tqs = c_info
                eng_tag = (c_parsed.engine or "json").upper()
                choices = [
                    questionary.Choice(
                        title=f"Use companion '{c_path.name}' ({eng_tag} • TQS {c_tqs.total_score:.0f} • {c_tqs.grade}) [skip transcription]",
                        value="companion",
                    ),
                    questionary.Choice(title="Transcribe with Gemini", value="gemini"),
                    questionary.Choice(title="Transcribe with Whisper", value="whisper"),
                    questionary.Choice(title="Skip this file", value="skip"),
                ]
                ans = questionary.select(
                    f"Action for '{file.name}':",
                    choices=choices,
                ).ask()

                if ans == "companion":
                    alloc_dict.update({
                        "engine": f"companion:{c_parsed.engine or 'json'}",
                        "companion_file": c_path,
                        "tqs_score": f"{c_tqs.total_score:.1f}",
                    })
                    allocations.append(alloc_dict)
                elif ans and ans != "skip":
                    alloc_dict["engine"] = ans
                    allocations.append(alloc_dict)
            else:
                engine = questionary.select(
                    f"Engine for '{file.name}':", choices=["gemini", "whisper", "skip"]
                ).ask()

                if engine and engine != "skip":
                    alloc_dict["engine"] = engine
                    allocations.append(alloc_dict)

    if not allocations:
        console.print(f"  [{DIM}]All files skipped. Import cancelled.[/]")
        return []

    # 4. Trigger Background Importer
    console.print(f"\n  [{BOLD} {ACCENT}]Ready to import {len(allocations)} files![/]")

    from audiobench.storage.importer import BackgroundImporter

    importer = BackgroundImporter(max_workers=4)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        transient=False,
    )

    job_tasks: dict[str, TaskID] = {}

    def on_progress(total_bytes: int, copied_bytes: int, filename: str) -> None:
        if filename not in job_tasks:
            task = progress.add_task(f"[cyan]{filename}", total=total_bytes)
            job_tasks[filename] = task
        else:
            progress.update(job_tasks[filename], completed=copied_bytes)

    def on_file_done(source: Path, dest: Path, engine: str) -> None:
        pass

    importer.set_callbacks(progress=on_progress, file_done=on_file_done)

    with progress:
        results = importer.run_import_jobs(allocations)

    reflink_count = sum(1 for r in results if r.get("copy_method") == "reflink")
    stream_count = sum(1 for r in results if r.get("copy_method") == "stream_copy")
    dup_count = sum(1 for r in results if r.get("is_duplicate"))

    console.print(f"\n  [{BOLD} {ACCENT}]✓ Successfully imported {len(results)} files![/]")
    if reflink_count > 0:
        console.print(f"  [{DIM}]⚡ Reflink CoW clones: {reflink_count} (instantaneous, 0 duplicated disk space)[/]")
    if stream_count > 0:
        console.print(f"  [{DIM}]📦 Stream copied: {stream_count}[/]")
    if dup_count > 0:
        console.print(f"  [{DIM}]🔁 Deduplicated: {dup_count} (existing library files reused)[/]")

    # 5. Register to Database
    import json
    from audiobench.core.db_session import get_session
    from audiobench.observatory.context import log_event
    from audiobench.storage.models import AudioFileRecord, WorkRecord

    imported_ids = []
    with get_session() as session:
        for res in results:
            dest: Path = res["destination"]
            engine: str = res["engine"]
            is_dup = res["is_duplicate"]
            file_hash = res["file_hash"]

            tag_list = [f"engine_preference: {engine}"]
            if res.get("model"):
                tag_list.append(f"model_preference: {res['model']}")
            tag_json = json.dumps(tag_list)

            if is_dup:
                existing_id = res["existing_id"]
                audio_record = session.query(AudioFileRecord).get(existing_id)
                if audio_record:
                    audio_record.tags = tag_json
                    imported_ids.append(audio_record.id)
                    console.print(
                        f"  [{DIM}]Skipped duplicate data for '{dest.name}', but updated engine to {engine}.[/]"
                    )
            else:
                title, author = extract_metadata(dest)
                if override_title: title = override_title
                if override_author: author = override_author

                if not author or not title:
                    from audiobench.daemon.intelligence.work_identity import resolve_work_identity
                    res_title, res_author, _ = resolve_work_identity(file_path=str(dest), file_name=dest.name)
                    if not title and res_title:
                        title = res_title
                    if not author and res_author and res_author.lower() != "unknown author":
                        author = res_author

                work_id = None
                if title:
                    work_record = session.query(WorkRecord).filter_by(title=title, author=author).first()
                    if not work_record:
                        work_record = WorkRecord(title=title, author=author)
                        session.add(work_record)
                        session.flush()
                    work_id = work_record.id

                audio_record = AudioFileRecord(
                    file_path=str(dest),
                    file_name=dest.name,
                    file_size_bytes=dest.stat().st_size,
                    format=dest.suffix.lstrip("."),
                    file_hash=file_hash,
                    tags=tag_json,
                    work_id=work_id,
                )
                session.add(audio_record)
                session.flush()

                if not work_id:
                    log_event(
                        "import",
                        "work_unassigned",
                        f"Work could not be assigned for {dest.name}",
                        entity_type="audio_file",
                        entity_id=audio_record.id
                    )

                imported_ids.append(audio_record.id)
        session.commit()

    # 5.5 Auto-detect chapters for newly imported audio files
    for res in results:
        if not res.get("is_duplicate") and res.get("destination"):
            dest_path = res["destination"]
            try:
                from audiobench.chapters.detector import ChapterDetector
                from audiobench.storage.chapter_repository import get_chapter_repo
                from audiobench.storage.models import AudioFileRecord
                with get_session() as s:
                    af = s.query(AudioFileRecord).filter_by(file_path=str(dest_path)).first()
                    if af:
                        chaps = ChapterDetector().detect(dest_path)
                        if chaps:
                            get_chapter_repo().save_chapters(af.id, chaps)
            except Exception:
                pass

    # 6. Process Companion Transcripts (if any)
    companion_imports = [r for r in results if r.get("companion_file")]
    if companion_imports:
        from audiobench.core.db_engine import init_db
        from audiobench.storage.models import TranscriptionRecord
        from audiobench.storage.repository import TranscriptionRepository
        from audiobench.transcribe.transcript_parser import build_audio_metadata, parse_transcript_file

        init_db()
        tx_repo = TranscriptionRepository()

        for res in companion_imports:
            dest = res["destination"]
            companion_path = Path(res["companion_file"])
            dest_sidecar = dest.with_suffix(companion_path.suffix.lower())
            try:
                if companion_path.exists() and companion_path.resolve() != dest_sidecar.resolve():
                    import shutil
                    shutil.copy2(str(companion_path), str(dest_sidecar))

                parsed_tx, raw_meta = parse_transcript_file(dest_sidecar)
                try:
                    audio_meta = build_audio_metadata(dest, raw_meta)
                except Exception:
                    audio_meta = build_audio_metadata(dest, None)

                tx_id = tx_repo.save_transcription(parsed_tx, audio_meta, overwrite=True)

                with get_session() as s:
                    tx_rec = s.query(TranscriptionRecord).filter_by(id=tx_id).first()
                    if tx_rec:
                        tx_rec.source = "companion_import"
                        s.commit()

                tqs_info = f", TQS: {res.get('tqs_score')}" if res.get("tqs_score") else ""
                console.print(
                    f"  [{BOLD}{ACCENT}]✓ Imported companion transcript for '{dest.name}' (Transcription #{tx_id}{tqs_info})[/]"
                )
            except Exception as ex:
                console.print(f"  [{DIM}]Note: companion transcript import for '{dest.name}' had error: {ex}[/]")

    # Fire plugin hooks for each newly imported file
    try:
        from audiobench.events import get_bus
        _bus = get_bus()
        for res in results:
            if not res.get("is_duplicate"):
                _bus.emit(
                    "import.complete",
                    audio_file_id=res.get("audio_file_id") or imported_ids[results.index(res)],
                    file_path=str(res["destination"]),
                )
    except Exception:
        pass

    # 7. Actionable Post-Import Launchpad
    run_post_import_launchpad(imported_ids, results, interactive=interactive)

    return imported_ids


@click.command(name="import")
@click.argument("paths", nargs=-1, type=str)
@click.option("-e", "--engine", type=click.Choice(["gemini", "whisper", "auto", "hybrid", "companion_only"], case_sensitive=False), default=None, help="Default transcription engine or batch allocation strategy")
@click.option("-m", "--model", type=str, default=None, help="Transcription model to use")
@click.option("-y", "--yes", is_flag=True, help="Non-interactive mode: accept defaults without confirmation prompts")
@click.option("-T", "--transcript", is_flag=True, help="Reverse import: select transcript file first")
@click.option("--batch", is_flag=True, help="Batch mode for --transcript (folder-to-folder mapping)")
@click.option("--auto-detect", is_flag=True, help="Auto-detect mode: recursively scan and pair audio and transcript files")
@click.option("--title", default=None, help="Force assign a work title to imported files")
@click.option("--author", default=None, help="Force assign a work author to imported files")
def import_cmd(
    paths: tuple[str, ...],
    engine: str | None,
    model: str | None,
    yes: bool,
    transcript: bool,
    batch: bool,
    auto_detect: bool,
    title: str | None,
    author: str | None,
):
    """Import audio files into the internal library.

    \b
    Examples:
      audiobench import
      audiobench import ./audio/
      audiobench import speech.mp3
      audiobench import *.mp3 -e whisper -y
      audiobench import transcript.json
      audiobench import https://youtu.be/...
      audiobench import --transcript --batch
      audiobench import --auto-detect
    """
    initial_mode = "main_menu"
    target_files = None
    shared_state = {}

    if not paths:
        if auto_detect:
            initial_mode = "MODE_AUTO"
        elif transcript and batch:
            initial_mode = "MODE_BATCH_TX"
        elif transcript:
            initial_mode = "MODE_SINGLE_TX"
        else:
            initial_mode = "main_menu"
    else:
        url_imported_ids = []
        regular_paths = []

        for p_str in paths:
            if p_str.startswith("http://") or p_str.startswith("https://"):
                if "youtube.com" in p_str or "youtu.be" in p_str:
                    try:
                        from audiobench.youtube.fetcher import fetch_and_register
                        console.print(f"  [{DIM}]Fetching YouTube audio from {p_str}...[/]")
                        aid = fetch_and_register(p_str)
                        console.print(f"  [{SUCCESS}]✓ Registered YouTube audio (ID #{aid})[/]")
                        url_imported_ids.append(aid)
                    except Exception as ex:
                        console.print(error_panel("YouTube Import Failed", str(ex)))
                else:
                    import tempfile
                    import urllib.parse
                    import urllib.request
                    parsed_u = urllib.parse.urlparse(p_str)
                    name = Path(parsed_u.path).name or "web_audio.mp3"
                    if not any(name.lower().endswith(f".{ext}") for ext in ["mp3", "m4a", "wav", "flac", "ogg", "opus", "webm"]):
                        name += ".mp3"
                    tmp_target = Path(tempfile.gettempdir()) / f"audiobench_{name}"
                    console.print(f"  [{DIM}]Downloading remote audio from {p_str}...[/]")
                    try:
                        req = urllib.request.Request(p_str, headers={"User-Agent": "Audiobench/1.0"})
                        with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_target, "wb") as f_out:
                            f_out.write(resp.read())
                        regular_paths.append(str(tmp_target))
                    except Exception as ex:
                        console.print(error_panel("URL Download Failed", str(ex)))
            else:
                regular_paths.append(p_str)

        if len(regular_paths) == 1:
            resolved_p = Path(regular_paths[0]).resolve()
            if resolved_p.is_dir():
                shared_state["current_path"] = resolved_p
                if transcript and batch:
                    initial_mode = "MODE_BATCH_TX"
                elif transcript:
                    initial_mode = "MODE_SINGLE_TX"
                elif engine or yes:
                    from audiobench.cli.io.file_collector import collect_files
                    collected = collect_files(tuple(regular_paths), recursive=True)
                    if collected:
                        target_files = collected
                        initial_mode = "MODE_AUDIO"
                    else:
                        initial_mode = "MODE_AUTO"
                else:
                    initial_mode = "MODE_AUTO"
            elif resolved_p.is_file():
                if resolved_p.suffix.lower() == ".json":
                    initial_mode = "MODE_SINGLE_TX"
                    shared_state["selected_tx_file"] = resolved_p
                    shared_state["current_path"] = resolved_p.parent
                else:
                    initial_mode = "MODE_AUDIO"
                    target_files = [resolved_p]
            else:
                from audiobench.cli.io.file_collector import collect_files
                collected = collect_files(tuple(regular_paths), recursive=True)
                if collected:
                    target_files = collected
                    initial_mode = "MODE_AUDIO"
                else:
                    console.print(error_panel("File Not Found", f"Cannot find audio matching: {regular_paths[0]}"))
                    return
        elif len(regular_paths) > 1:
            from audiobench.cli.io.file_collector import collect_files
            collected = collect_files(tuple(regular_paths), recursive=True)
            if collected:
                target_files = collected
                initial_mode = "MODE_AUDIO"
            else:
                console.print(error_panel("Files Not Found", f"No valid audio files found matching: {', '.join(regular_paths)}"))
                return

        if not regular_paths and url_imported_ids:
            if not yes:
                run_post_import_launchpad(url_imported_ids, [], interactive=True)
            return

    run_import_flow(
        restore_state=shared_state,
        initial_mode=initial_mode,
        override_title=title,
        override_author=author,
        target_files=target_files,
        default_engine=engine.lower() if engine else None,
        default_model=model,
        interactive=not yes,
    )
