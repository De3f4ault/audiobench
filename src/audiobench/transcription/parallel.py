import multiprocessing
import os
import subprocess
import time
from pathlib import Path

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, SUCCESS, console, error_panel


def detect_gpus() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        return len([l for l in result.stdout.strip().split('\n') if l.strip()])
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return 0

def warm_model_cache(model_name: str):
    console.print(f"  [{DIM}]Warming model cache for '{model_name}'...[/]")
    try:
        from faster_whisper import WhisperModel
        # Download is triggered upon instantiation
        _ = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=None)
    except Exception as e:
        console.print(f"  [{DIM}]Could not warm cache: {e}[/]")

def _worker_process(gpu_id: int, queue: multiprocessing.Queue, total_files: int, opts: dict):
    # Set GPU isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Set DB isolation
    db_dir = os.path.join(opts.get("output_path", "/tmp"), f".audiobench_gpu{gpu_id}")
    os.makedirs(db_dir, exist_ok=True)
    os.environ["AUDIOBENCH_DATA_DIR"] = db_dir
    os.environ["AUDIOBENCH_DATABASE_URL"] = f"sqlite:///{db_dir}/transcriptions.db"

    from audiobench.cli.io.output_resolver import resolve_output
    from audiobench.transcribe.transcriber import TranscriptionPipeline

    pipeline = TranscriptionPipeline()

    while True:
        try:
            file_idx, file_path = queue.get_nowait()
        except multiprocessing.queues.Empty:
            break
        except Exception:
            break

        input_p = Path(file_path)
        print(f"[GPU {gpu_id}] [{file_idx}/{total_files}] Processing {input_p.name} ...", flush=True)

        start_time = time.perf_counter()

        try:
            resolved_output, resolved_format = resolve_output(
                str(file_path),
                opts.get("output_path"),
                opts.get("primary_format"),
                opts.get("settings_output_format"),
                input_base_dir=opts.get("input_base_dir"),
                collision=opts.get("collision"),
            )

            if resolved_output is None and opts.get("collision") == "skip":
                print(f"[GPU {gpu_id}] [{file_idx}/{total_files}] Skipped {input_p.name} (exists)", flush=True)
                continue

            # Need a stub PhaseTracker for quiet
            class StubTracker:
                def update(self, *args, **kwargs): pass
                def on_segment(self, *args, **kwargs): pass
            tracker = StubTracker()

            transcript = pipeline.transcribe_file(
                file_path=str(file_path),
                language=opts.get("language"),
                output_format=resolved_format,
                output_path=resolved_output,
                word_timestamps=opts.get("word_timestamps", True),
                skip_cache=opts.get("skip_cache", False),
                speed_preset=opts.get("speed_preset", "balanced"),
                initial_prompt=opts.get("initial_prompt"),
                translate=opts.get("translate", False),
                enable_diarization=opts.get("enable_diarization", False),
                diarize_mode=opts.get("diarize_mode", "fast"),
                diarize_threshold=opts.get("diarize_threshold", 0.65),
                map_speakers=opts.get("map_speakers"),
                auto_name=opts.get("auto_name", False),
                filters=opts.get("filters"),
                engine_name=opts.get("engine_name", "whisper"),
                target_chapters=opts.get("target_chapters"),
                resume=opts.get("resume", False),
                parallel=opts.get("parallel_chapters", 1),
                skip_ghost=opts.get("skip_ghost", True),
                on_phase=tracker.update,
                on_segment=tracker.on_segment,
            )

            elapsed = time.perf_counter() - start_time
            if transcript:
                speed_ratio = transcript.duration_seconds / elapsed if elapsed > 0 else 0

                # output extra formats if any
                extra_formats = opts.get("extra_formats", [])
                if extra_formats:
                    from audiobench.output.base import get_formatter as get_fmt
                    for extra_fmt in extra_formats:
                        extra_out, _ = resolve_output(
                            str(file_path),
                            opts.get("output_path"),
                            extra_fmt,
                            extra_fmt,
                            input_base_dir=opts.get("input_base_dir"),
                            collision=opts.get("collision"),
                        )
                        if extra_out:
                            fmt_obj = get_fmt(extra_fmt)
                            content = fmt_obj.format(transcript)
                            Path(extra_out).parent.mkdir(parents=True, exist_ok=True)
                            Path(extra_out).write_text(content, encoding="utf-8")

                print(f"[GPU {gpu_id}] [{file_idx}/{total_files}] [OK] Done {input_p.name} ({elapsed:.1f}s, {speed_ratio:.1f}x)", flush=True)
            else:
                print(f"[GPU {gpu_id}] [{file_idx}/{total_files}] [OK] Done {input_p.name} ({elapsed:.1f}s)", flush=True)

        except Exception as e:
            print(f"[GPU {gpu_id}] [{file_idx}/{total_files}] [ERR] ERROR {input_p.name}: {e}", flush=True)

class ParallelTranscriber:
    def run(self, files: list, opts: dict):
        gpu_count = detect_gpus()

        workers = opts.get("workers", 0)
        if workers <= 0:
            workers = gpu_count

        if workers < 1:
            console.print(error_panel("Parallel Transcription", "No GPUs detected and workers=0. Falling back to 1 worker."))
            workers = 1

        console.print(f"  [{ACCENT}]Detected {gpu_count} GPU(s) — spawning {workers} worker(s)[/]")

        if opts.get("model"):
            warm_model_cache(opts.get("model"))

        manager = multiprocessing.Manager()
        queue = manager.Queue()

        from audiobench.cli.io.output_resolver import resolve_output
        pending_files = []
        for f in files:
            resolved_output, _ = resolve_output(
                str(f),
                opts.get("output_path"),
                opts.get("primary_format"),
                opts.get("settings_output_format"),
                input_base_dir=opts.get("input_base_dir"),
                collision=opts.get("collision"),
            )
            if resolved_output is None and opts.get("collision") == "skip":
                continue
            pending_files.append(f)

        total_files = len(pending_files)
        console.print(f"  [{BOLD}][{total_files} files pending → dynamic queue][/]\n")

        if total_files == 0:
            return

        for i, f in enumerate(pending_files, 1):
            queue.put((i, str(f)))

        processes = []
        # Ensure multiprocessing context is spawn to avoid CUDA initialization issues
        ctx = multiprocessing.get_context('spawn')
        for i in range(workers):
            gpu_id = i % gpu_count if gpu_count > 0 else 0
            p = ctx.Process(target=_worker_process, args=(gpu_id, queue, total_files, opts))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        console.print(f"\n  [{SUCCESS}]All parallel workers finished![/]")
