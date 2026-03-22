"""Output resolver — determines output path and format from CLI args.

Handles collision strategies (overwrite/skip/rename), format detection,
multi-format parsing, directory mirroring, and auto-naming.

Usage:
    from audiobench.cli.io.output_resolver import resolve_output, parse_formats

    path, fmt = resolve_output("input.mp3", None, "srt", "txt")
    formats = parse_formats("srt,json")
"""

from __future__ import annotations

from pathlib import Path

from audiobench.cli.display.theme import FORMAT_TO_EXT, detect_format_from_path


def resolve_collision(path: str, strategy: str) -> str | None:
    """Apply collision strategy when output file already exists.

    Args:
        path: The target output path.
        strategy: One of 'overwrite', 'skip', 'rename'.

    Returns:
        Final path to write to, or None if 'skip'.
    """
    p = Path(path)
    if not p.exists():
        return path

    if strategy == "overwrite":
        return path
    elif strategy == "skip":
        return None  # Caller should skip this file
    elif strategy == "rename":
        # Auto-increment: file.srt → file_1.srt → file_2.srt
        stem = p.stem
        suffix = p.suffix
        parent = p.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return str(candidate)
            counter += 1
    return path  # fallback


def resolve_output(
    input_path: str,
    output_path: str | None,
    output_format: str | None,
    default_format: str,
    *,
    input_base_dir: str | None = None,
    collision: str = "overwrite",
) -> tuple[str | None, str]:
    """Resolve output path and format from CLI args.

    Rules:
        1. -o path.srt              → auto-detect format from extension
        2. -f srt (no -o)           → <stem>.srt in same dir as input
        3. -o dir/ (existing dir)   → dir/<stem>.<fmt>
        3b. Mirror mode: if input_base_dir is set, preserves relative
            structure (e.g., base/sub/file.m4a → out/sub/file.srt)
        4. Neither -o nor -f        → None (print to stdout)

    Args:
        input_path: Path to the input audio file.
        output_path: User-provided output path (-o flag).
        output_format: User-provided format (-f flag).
        default_format: Fallback format from settings.
        input_base_dir: If set, enables mirror mode — preserves relative
            directory structure under the output directory.
        collision: Strategy for existing files: 'overwrite', 'skip', 'rename'.

    Returns:
        Tuple of (resolved_output_path or None, format_string).
    """
    input_p = Path(input_path)
    stem = input_p.stem

    # Rule 1: -o with extension → auto-detect format
    if output_path:
        out_p = Path(output_path)

        # Rule 3: output is a directory
        if out_p.is_dir() or output_path.endswith("/"):
            fmt = output_format or default_format
            ext = FORMAT_TO_EXT.get(fmt, f".{fmt}")
            out_p.mkdir(parents=True, exist_ok=True)

            # Mirror mode (C1): preserve relative directory structure
            if input_base_dir:
                try:
                    rel = Path(input_path).resolve().relative_to(Path(input_base_dir).resolve())
                    # Replace the file extension, keep the dir structure
                    mirrored = out_p / rel.with_suffix(ext)
                    mirrored.parent.mkdir(parents=True, exist_ok=True)
                    final = str(mirrored)
                except ValueError:
                    # input_path is not relative to input_base_dir, fallback
                    final = str(out_p / f"{stem}{ext}")
            else:
                final = str(out_p / f"{stem}{ext}")

            # C2: collision handling
            final = resolve_collision(final, collision)
            return final, fmt

        # Rule 1: detect format from output extension
        detected = detect_format_from_path(output_path)
        fmt = output_format or detected or default_format
        final = resolve_collision(output_path, collision)
        return final, fmt

    # Rule 2: -f specified but no -o → auto-name
    if output_format:
        ext = FORMAT_TO_EXT.get(output_format, f".{output_format}")
        auto_path = str(input_p.with_suffix(ext))
        final = resolve_collision(auto_path, collision)
        return final, output_format

    # Rule 4: neither → stdout
    return None, default_format


def parse_formats(format_str: str | None) -> list[str]:
    """Parse multi-format string like 'srt,json' or 'all'.

    Returns a list of format strings. If None, returns empty list
    (caller should use default_format).
    """
    valid_formats = {"txt", "srt", "vtt", "json"}

    if not format_str:
        return []

    if format_str.strip().lower() == "all":
        return sorted(valid_formats)

    formats = [f.strip().lower() for f in format_str.split(",")]
    invalid = [f for f in formats if f not in valid_formats]
    if invalid:
        from audiobench.cli.display.theme import console, error_panel

        console.print(
            error_panel(
                "Invalid format",
                f"Unknown format(s): {', '.join(invalid)}. "
                f"Valid: {', '.join(sorted(valid_formats))}",
            )
        )
        return []  # Return empty to signal error

    return formats
