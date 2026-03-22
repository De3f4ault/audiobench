"""File collector — resolves CLI input paths into a flat file list.

Handles individual files, directories, globs, recursive walking,
extension filtering, manifest files, stdin piping, exclude patterns,
and deduplication via resolved paths.

Usage:
    from audiobench.cli.io.file_collector import collect_files

    files = collect_files(("audio/", "extra.mp3"), recursive=True)
"""

from __future__ import annotations

import sys
from pathlib import Path


def collect_files(
    paths: tuple[str, ...],
    *,
    recursive: bool = False,
    extensions: str | None = None,
    from_file: str | None = None,
    exclude: str | None = None,
) -> list[Path]:
    """Resolve CLI input paths into a flat, deduplicated list of files.

    Handles:
    - Individual files passed directly
    - Directories (walks for supported audio/video files)
    - Recursive walking (-R / --recursive)
    - Extension filtering (--ext mp3,m4a)
    - Manifest file (--from-file list.txt)
    - Stdin piping (path "-" reads file paths from stdin)
    - Exclude patterns (--exclude "*_draft*")
    - Deduplication via resolved paths

    Args:
        paths: Tuple of file/directory paths from Click argument.
        recursive: Walk directories recursively.
        extensions: Comma-separated list of extensions to include (e.g. "mp3,m4a").
        from_file: Path to a manifest file containing one path per line.
        exclude: Comma-separated glob patterns to exclude.

    Returns:
        Sorted, deduplicated list of Path objects.
    """
    from audiobench.transcribe.audio_converter import ALL_SUPPORTED_FORMATS

    # Build the allowed extension set
    if extensions:
        allowed_exts = {e.strip().lstrip(".").lower() for e in extensions.split(",")}
    else:
        allowed_exts = ALL_SUPPORTED_FORMATS

    # Build exclude patterns
    exclude_patterns = []
    if exclude:
        exclude_patterns = [p.strip() for p in exclude.split(",")]

    def _is_excluded(p: Path) -> bool:
        """Check if a path matches any exclude pattern."""
        if not exclude_patterns:
            return False
        import fnmatch

        name = p.name
        return any(fnmatch.fnmatch(name, pat) for pat in exclude_patterns)

    def _is_supported(p: Path) -> bool:
        """Check if a file has a supported extension."""
        ext = p.suffix.lstrip(".").lower()
        return ext in allowed_exts

    def _walk_directory(dir_path: Path) -> list[Path]:
        """Collect supported files from a directory."""
        if recursive:
            return sorted(
                f
                for f in dir_path.rglob("*")
                if f.is_file() and _is_supported(f) and not _is_excluded(f)
            )
        else:
            return sorted(
                f
                for f in dir_path.iterdir()
                if f.is_file() and _is_supported(f) and not _is_excluded(f)
            )

    collected: list[Path] = []

    # Collect from manifest file
    if from_file:
        manifest = Path(from_file)
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    p = Path(line).expanduser()
                    if p.exists():
                        collected.append(p)

    # Collect from stdin ("-")
    for path_str in paths:
        if path_str == "-":
            for line in sys.stdin:
                line = line.strip()
                if line:
                    p = Path(line).expanduser()
                    if p.exists():
                        collected.append(p)
            continue

        p = Path(path_str)

        if p.is_dir():
            collected.extend(_walk_directory(p))
        elif p.is_file() and not _is_excluded(p):
            collected.append(p)
        # else: Click's exists=True already validated, but be safe

    # Deduplicate via resolved paths, preserving order
    seen: set[Path] = set()
    deduped: list[Path] = []
    for f in collected:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(f)

    return deduped
