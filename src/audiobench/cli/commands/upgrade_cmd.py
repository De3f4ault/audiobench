"""Advisory CLI Upgrade and Rollback commands for AudioBench.

Provides `audiobench library upgrade` and `audiobench library rollback`,
enforcing daemon liveness (Gates 1 & 2), advisory lockfiles, quality evaluation,
fidelity omission verification, and post-hoc stratified audit sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import glob
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from audiobench.cli.commands.library_cmd import library_group
from audiobench.cli.display.theme import ACCENT, APP_NAME, BOLD, DIM, SUCCESS, WARNING
from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.transcribe.matcher import MatchResult, TranscriptMatcher
from audiobench.transcribe.quality_evaluator import (
    ComparisonResult,
    compare_transcripts,
)
from audiobench.transcribe.upgrader import (
    TranscriptUpgrader,
    create_pre_upgrade_backup,
)

logger = get_logger("cli.upgrade")
console = Console()


class UpgradeLock:
    """Advisory process lock preventing concurrent batch upgrades."""

    def __init__(self, lock_path: Path | str = "data/upgrade.lock", max_age_hours: float = 6.0) -> None:
        self.lock_path = Path(lock_path).resolve()
        self.max_age_hours = max_age_hours

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            try:
                data = json.loads(self.lock_path.read_text(encoding="utf-8"))
                pid = data.get("pid")
                ts = data.get("timestamp", 0)
                is_stale = False

                # Process liveness check
                if pid:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        # Process dead
                        is_stale = True

                # Age check (> 6h)
                if time.time() - ts > self.max_age_hours * 3600:
                    is_stale = True

                if is_stale:
                    logger.warning("Clearing stale upgrade lockfile (PID %s).", pid)
                    self.lock_path.unlink(missing_ok=True)
                else:
                    raise click.ClickException(
                        f"Another upgrade process is running (PID: {pid}).\n"
                        f"If this is a dead process, remove {self.lock_path} manually."
                    )
            except (json.JSONDecodeError, KeyError):
                logger.warning("Clearing corrupt lockfile.")
                self.lock_path.unlink(missing_ok=True)

        payload = {"pid": os.getpid(), "timestamp": time.time()}
        self.lock_path.write_text(json.dumps(payload), encoding="utf-8")

    def release(self) -> None:
        try:
            if self.lock_path.exists():
                data = json.loads(self.lock_path.read_text(encoding="utf-8"))
                if data.get("pid") == os.getpid():
                    self.lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    def __enter__(self) -> UpgradeLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def check_daemon_gate1() -> None:
    """Gate 1: Hard check before any DB write. Exits with code 1 if daemon is down."""
    try:
        from audiobench.daemon.client import DaemonClient

        client = DaemonClient()
        if not client.daemon_is_healthy():
            console.print(
                "\n[bold red]✗ Daemon not reachable.[/bold red] The upgrade command requires a running daemon to keep\n"
                "  the HNSW vector index consistent.\n\n"
                "  Start it first:   [bold cyan]audiobench daemon start[/bold cyan]\n"
                "  Then retry:       [bold cyan]audiobench library upgrade <path> [flags][/bold cyan]\n\n"
                "  If you must run offline, upgrade manually then reindex:\n"
                "    [bold yellow]audiobench daemon reindex[/bold yellow]\n"
            )
            raise SystemExit(1)
    except Exception as e:
        if isinstance(e, SystemExit):
            raise
        console.print(f"[bold red]✗ Daemon check error:[/bold red] {e}")
        raise SystemExit(1)


def check_daemon_gate2(batch_id: str, upgraded_count: int, flag_path: Path | None = None) -> None:
    """Gate 2: Post-batch check. Surfaces warning and writes durable flag file if daemon died mid-run."""
    try:
        from audiobench.daemon.client import DaemonClient

        client = DaemonClient()
        if not client.daemon_is_healthy():
            if flag_path is None:
                flag_path = Path("data/needs_reindex")
            flag_path.parent.mkdir(parents=True, exist_ok=True)
            flag_data = {
                "batch_id": batch_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "upgraded_count": upgraded_count,
                "reason": "Daemon unreachable after batch upgrade commit",
            }
            try:
                flag_path.write_text(json.dumps(flag_data, indent=2), encoding="utf-8")
            except Exception as ex:
                logger.warning("Could not write durable needs_reindex flag: %s", ex)

            console.print("\n" + "=" * 80)
            console.print("[bold red]⚠ Daemon became unreachable during the batch (detected after commit).[/bold red]")
            console.print(f"  All {upgraded_count} records were written successfully to DB, but HNSW may be stale.\n")
            console.print("  [bold]Run immediately:[/bold]  [cyan]audiobench daemon start && audiobench daemon reindex[/cyan]")
            console.print(f"  [bold]Batch ID:[/bold]         {batch_id}")
            console.print(f"  [bold]Durable flag set:[/bold] {flag_path}")
            console.print("=" * 80 + "\n")
    except Exception as e:
        logger.warning("Error running Gate 2 daemon check: %s", e)


@dataclass
class EvaluatedCandidate:
    """Evaluated external transcript candidate paired with DB records."""

    json_path: Path
    data: dict[str, Any]
    match: MatchResult
    comparison: ComparisonResult


def scan_and_evaluate(
    target_path: Path,
    session: Any,
) -> tuple[list[EvaluatedCandidate], list[EvaluatedCandidate], list[EvaluatedCandidate]]:
    """Scan and evaluate transcript files using TranscriptMatcher and compare_transcripts.

    Returns:
      (clean_candidates, quarantined_candidates, skipped_candidates)
    """
    if target_path.is_file():
        files = [target_path]
    elif target_path.is_dir():
        files = sorted(target_path.glob("*.json"))
    else:
        raise click.BadParameter(f"Path does not exist: {target_path}")

    # Exclude registry files
    files = [f for f in files if "registry" not in f.name.lower()]

    matcher = TranscriptMatcher(session)
    clean: list[EvaluatedCandidate] = []
    quarantined: list[EvaluatedCandidate] = []
    skipped: list[EvaluatedCandidate] = []

    from audiobench.storage.models import TranscriptionRecord

    for f in files:
        match = matcher.match_file(f)
        if not match or not match.existing_transcription_id:
            continue

        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue

        existing_record = (
            session.query(TranscriptionRecord)
            .filter_by(id=match.existing_transcription_id)
            .first()
        )
        if not existing_record:
            continue

        comp = compare_transcripts(
            existing_record=existing_record,
            incoming_transcript=data,
            audio_duration=match.audio_duration,
        )

        cand = EvaluatedCandidate(json_path=f, data=data, match=match, comparison=comp)

        # Check fidelity and recommendation
        is_faithful = comp.fidelity.is_faithful if comp.fidelity else True
        if comp.recommendation == "UPGRADE_RECOMMENDED":
            if is_faithful:
                clean.append(cand)
            else:
                quarantined.append(cand)
        elif not is_faithful or comp.recommendation == "POTENTIAL_OMISSION_DETECTED":
            quarantined.append(cand)
        else:
            skipped.append(cand)

    return clean, quarantined, skipped


def display_quarantine(quarantined: list[EvaluatedCandidate]) -> None:
    """Display quarantined candidates with detected defects and omission flags."""
    if not quarantined:
        console.print("[green]No candidates currently quarantined.[/green]")
        return

    table = Table(title=f"Quarantined Transcript Candidates ({len(quarantined)})", border_style="red")
    table.add_column("Audio File", style="bold", max_width=35)
    table.add_column("Tier", justify="center")
    table.add_column("Existing TQS", justify="right")
    table.add_column("Incoming TQS", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Flag / Reason", style="yellow")

    for cand in quarantined:
        c = cand.comparison
        delta_str = f"{c.delta_tqs:+.1f}" if c.delta_tqs is not None else "N/A"
        flag = "Omission / Defect Detected"
        if c.fidelity and not c.fidelity.is_faithful:
            flag = f"Omission ({c.fidelity.flagged_windows_count} window(s))"
        elif c.recommendation == "POTENTIAL_OMISSION_DETECTED":
            flag = "Potential Omission"
        elif c.recommendation == "DEFECTIVE_REJECTED":
            flag = "Defective Quality"
        elif c.recommendation == "SIMILAR_QUALITY":
            flag = "Similar Quality (Not Upgrade)"

        ex_tqs_str = f"{c.existing_tqs.total_score:.1f}" if c.existing_tqs else "N/A"
        in_tqs_str = f"{c.incoming_tqs.total_score:.1f}" if c.incoming_tqs else "N/A"

        table.add_row(
            cand.match.audio_file_name,
            cand.match.match_tier,
            ex_tqs_str,
            in_tqs_str,
            delta_str,
            flag,
        )

    console.print(table)


def run_upgrade_pipeline(
    path: str,
    apply_all_clean: bool = False,
    review_quarantine: bool = False,
    batch_id: str | None = None,
    no_daemon_check: bool = False,
) -> None:
    """Execute the core upgrade workflow."""
    target_path = Path(path).resolve()
    if not target_path.exists():
        raise click.BadParameter(f"Path not found: {target_path}")

    if not no_daemon_check:
        check_daemon_gate1()

    if not batch_id:
        batch_id = f"batch_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    with UpgradeLock():
        with get_session() as session:
            console.print(f"\n[{BOLD} {ACCENT}]{APP_NAME}[/] — Transcript Upgrade Evaluation")
            console.print(f"[{DIM}]Scanning and scoring against library database...[/DIM]")

            clean, quarantined, skipped = scan_and_evaluate(target_path, session)

            console.print(
                f"Scan Results: [bold green]{len(clean)} clean upgrade candidate(s)[/bold green], "
                f"[bold red]{len(quarantined)} quarantined[/bold red], "
                f"[dim]{len(skipped)} skipped (similar/worse)[/dim]\n"
            )

            if review_quarantine:
                display_quarantine(quarantined)
                return

            if not clean:
                console.print("[yellow]No clean upgrade candidates available to apply.[/yellow]")
                if quarantined:
                    console.print(f"[dim]Run with --review-quarantine to inspect {len(quarantined)} flagged file(s).[/dim]")
                return

            upgrader = TranscriptUpgrader(session)

            if apply_all_clean:
                # Enforce typed confirmation phrase
                confirm_phrase = f"CONFIRM UPGRADE {len(clean)} RECORDS"
                console.print(
                    Panel(
                        f"[bold yellow]GUARDED BULK UPGRADE ACTION[/bold yellow]\n\n"
                        f"You are about to upgrade [bold green]{len(clean)}[/bold green] production transcripts in place.\n"
                        f"All old transcripts will be cloned to [cyan]archived_transcriptions[/cyan] for granular rollback.\n"
                        f"A full safety snapshot of the database will be created before any changes.\n\n"
                        f"Type [bold red]'{confirm_phrase}'[/bold red] to proceed:",
                        border_style="yellow",
                    )
                )

                typed = click.prompt("Confirmation", default="", show_default=False)
                if typed.strip() != confirm_phrase:
                    console.print("[bold red]Aborted.[/bold red] Confirmation phrase did not match. No changes made.")
                    return

                # Pre-flight backup snapshot
                bak_path = create_pre_upgrade_backup(suffix=batch_id[:16])
                console.print(f"[{SUCCESS}]✓ Safety snapshot created:[/] {bak_path.name}")

                upgraded_audio_ids: list[int] = []

                with console.status(f"Upgrading {len(clean)} transcripts..."):
                    for cand in clean:
                        res = upgrader.upgrade_record(
                            audio_file_id=cand.match.audio_file_id,
                            incoming_data=cand.data,
                            upgrade_batch_id=batch_id,
                            keep_chapters=True,
                        )
                        upgraded_audio_ids.append(res.audio_file_id)

                    session.commit()

                console.print(
                    f"\n[{SUCCESS}]✓ Successfully upgraded {len(upgraded_audio_ids)} records under batch ID:[/] [cyan]{batch_id}[/cyan]"
                )

                # Gate 2 daemon check
                if not no_daemon_check:
                    check_daemon_gate2(batch_id, len(upgraded_audio_ids))

                # Post-Hoc Audit Sampling
                console.print(f"\n[{BOLD} {ACCENT}]Post-Hoc Human Verification Audit (Stratified Sample)[/]")
                samples = upgrader.select_post_hoc_audit_samples(upgraded_audio_ids, sample_size=15)
                if samples:
                    tbl = Table(title="15 Stratified Post-Hoc Audit Candidates (Duration & Author Diverse)", border_style="cyan")
                    tbl.add_column("ID", justify="right")
                    tbl.add_column("Author / Work", style="bold")
                    tbl.add_column("File Name")
                    tbl.add_column("Stratum", justify="center")
                    tbl.add_column("Duration", justify="right")

                    for s in samples:
                        dur_m = (s["duration_seconds"] or 0) / 60.0
                        author_work = f"{s.get('author', 'Unknown')} - {s.get('work_title', 'Unknown')}"
                        tbl.add_row(
                            str(s["audio_file_id"]),
                            author_work[:35],
                            s["file_name"][:35],
                            s["stratum"],
                            f"{dur_m:.1f}m",
                        )
                    console.print(tbl)
                    console.print(
                        f"[{DIM}]Inspect playback and HUD sync for these samples using:\n"
                        f"  audiobench listen <ID>  or  audiobench inspect <ID>[/DIM]\n"
                    )
                return

            # Interactive Paged Review Mode
            console.print(f"Entering interactive review for {len(clean)} candidates. [y/n/d/q]\n")
            upgraded_count = 0
            bak_created = False

            for idx, cand in enumerate(clean, 1):
                c = cand.comparison
                ex_score = c.existing_tqs.total_score if c.existing_tqs else 0.0
                in_score = c.incoming_tqs.total_score if c.incoming_tqs else 0.0
                delta = c.delta_tqs

                console.print(f"[{BOLD}]Candidate [{idx}/{len(clean)}]:[/] {cand.match.audio_file_name}")
                console.print(
                    f"  TQS: [dim]{ex_score:.1f}[/dim] → [bold green]{in_score:.1f}[/bold green] ([cyan]{delta:+.1f}[/cyan]) · "
                    f"Segments: {len(cand.data.get('segments', []))} · Engine: {cand.data.get('engine', 'whisper')}"
                )

                while True:
                    choice = click.prompt(
                        "  Apply upgrade? [y]es, [n]o, [d]iff, [q]uit",
                        type=click.Choice(["y", "n", "d", "q"], case_sensitive=False),
                        default="y",
                    )
                    if choice == "d":
                        ex_sample = (cand.match.existing_full_text or "")[:150]
                        in_sample = (cand.data.get("text") or cand.data.get("full_text") or "")[:150]
                        console.print(f"    [dim]Existing excerpt:[/dim] {ex_sample}...")
                        console.print(f"    [cyan]Incoming excerpt:[/cyan] {in_sample}...\n")
                        continue
                    elif choice == "y":
                        if not bak_created:
                            create_pre_upgrade_backup(suffix=batch_id[:16])
                            bak_created = True

                        upgrader.upgrade_record(
                            audio_file_id=cand.match.audio_file_id,
                            incoming_data=cand.data,
                            upgrade_batch_id=batch_id,
                            keep_chapters=True,
                        )
                        session.commit()
                        upgraded_count += 1
                        console.print(f"    [{SUCCESS}]✓ Upgraded #{cand.match.audio_file_id}[/]\n")
                        break
                    elif choice == "n":
                        console.print("    [dim]Skipped.[/dim]\n")
                        break
                    elif choice == "q":
                        console.print("[dim]Exited review.[/dim]")
                        if not no_daemon_check and upgraded_count > 0:
                            check_daemon_gate2(batch_id, upgraded_count)
                        return

            if not no_daemon_check and upgraded_count > 0:
                check_daemon_gate2(batch_id, upgraded_count)


def run_rollback_pipeline(
    identifier: str | None = None,
    batch_id: str | None = None,
    no_daemon_check: bool = False,
) -> None:
    """Execute the rollback workflow for single records or batch IDs."""
    if not identifier and not batch_id:
        raise click.UsageError("Specify an audio ID/title to rollback, or pass --batch <batch_id>.")

    if not no_daemon_check:
        check_daemon_gate1()

    with UpgradeLock():
        with get_session() as session:
            upgrader = TranscriptUpgrader(session)

            if batch_id:
                console.print(f"[{BOLD}]Rolling back batch ID:[/] [cyan]{batch_id}[/cyan]...")
                results = upgrader.rollback_batch(batch_id)
                session.commit()
                console.print(
                    f"[{SUCCESS}]✓ Batch rollback complete:[/] restored {len(results)} transcriptions and their segments."
                )
            elif identifier:
                console.print(f"[{BOLD}]Rolling back transcription for:[/] {identifier}...")
                res = upgrader.rollback_record(identifier)
                session.commit()
                console.print(
                    f"[{SUCCESS}]✓ Rollback successful:[/] restored original transcription #{res.restored_transcription_id} "
                    f"({res.segments_restored_count} segments) for audio file #{res.audio_file_id}."
                )

            if not no_daemon_check:
                check_daemon_gate2(batch_id or "single_rollback", 1)


# ── Attach subcommands to `library_group` ───────────────────────────────────


@library_group.command(name="upgrade")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--apply-all-clean",
    is_flag=True,
    help="Guarded bulk upgrade: apply all clean, faithful candidates after typed confirmation.",
)
@click.option(
    "--review-quarantine",
    is_flag=True,
    help="Display quarantined candidates with detected omissions or quality defects.",
)
@click.option(
    "--batch-id",
    type=str,
    default=None,
    help="Custom upgrade batch identifier for tracking and rollback.",
)
@click.option(
    "--no-daemon-check",
    is_flag=True,
    hidden=True,
    help="Bypass daemon reachability checks (testing only).",
)
def library_upgrade_cmd(
    path: str,
    apply_all_clean: bool,
    review_quarantine: bool,
    batch_id: str | None,
    no_daemon_check: bool,
) -> None:
    """Evaluate and safely upgrade transcriptions from an external JSON directory or file."""
    run_upgrade_pipeline(
        path=path,
        apply_all_clean=apply_all_clean,
        review_quarantine=review_quarantine,
        batch_id=batch_id,
        no_daemon_check=no_daemon_check,
    )


@library_group.command(name="rollback")
@click.argument("identifier", required=False, default=None)
@click.option(
    "--batch",
    "batch_id",
    type=str,
    default=None,
    help="Roll back all records upgraded under this batch ID.",
)
@click.option(
    "--no-daemon-check",
    is_flag=True,
    hidden=True,
    help="Bypass daemon reachability checks (testing only).",
)
def library_rollback_cmd(
    identifier: str | None,
    batch_id: str | None,
    no_daemon_check: bool,
) -> None:
    """Roll back upgraded transcriptions using in-database archival records."""
    run_rollback_pipeline(
        identifier=identifier,
        batch_id=batch_id,
        no_daemon_check=no_daemon_check,
    )


# ── Top-level aliases (`audiobench upgrade` & `audiobench rollback`) ─────────


@click.command(name="upgrade")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--apply-all-clean",
    is_flag=True,
    help="Guarded bulk upgrade: apply all clean, faithful candidates after typed confirmation.",
)
@click.option(
    "--review-quarantine",
    is_flag=True,
    help="Display quarantined candidates with detected omissions or quality defects.",
)
@click.option(
    "--batch-id",
    type=str,
    default=None,
    help="Custom upgrade batch identifier for tracking and rollback.",
)
@click.option(
    "--no-daemon-check",
    is_flag=True,
    hidden=True,
    help="Bypass daemon reachability checks (testing only).",
)
def upgrade_cmd(
    path: str,
    apply_all_clean: bool,
    review_quarantine: bool,
    batch_id: str | None,
    no_daemon_check: bool,
) -> None:
    """Evaluate and safely upgrade transcriptions (alias for 'audiobench library upgrade')."""
    run_upgrade_pipeline(
        path=path,
        apply_all_clean=apply_all_clean,
        review_quarantine=review_quarantine,
        batch_id=batch_id,
        no_daemon_check=no_daemon_check,
    )


@click.command(name="rollback")
@click.argument("identifier", required=False, default=None)
@click.option(
    "--batch",
    "batch_id",
    type=str,
    default=None,
    help="Roll back all records upgraded under this batch ID.",
)
@click.option(
    "--no-daemon-check",
    is_flag=True,
    hidden=True,
    help="Bypass daemon reachability checks (testing only).",
)
def rollback_cmd(
    identifier: str | None,
    batch_id: str | None,
    no_daemon_check: bool,
) -> None:
    """Roll back upgraded transcriptions (alias for 'audiobench library rollback')."""
    run_rollback_pipeline(
        identifier=identifier,
        batch_id=batch_id,
        no_daemon_check=no_daemon_check,
    )
