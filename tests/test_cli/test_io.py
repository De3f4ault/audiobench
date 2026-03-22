"""Tests for file_collector — input path resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestCollectFiles:
    """Test collect_files() with various input scenarios."""

    def _collect(self, paths, **kwargs):
        """Helper — import inside to avoid heavy top-level imports."""
        # Patch ALL_SUPPORTED_FORMATS at the source module (audio_converter)
        with patch(
            "audiobench.transcribe.audio_converter.ALL_SUPPORTED_FORMATS",
            {"mp3", "m4a", "wav", "flac", "ogg", "mp4", "mkv", "avi"},
        ):
            from audiobench.cli.io.file_collector import collect_files

            return collect_files(paths, **kwargs)

    def test_single_file(self, sample_audio_dir):
        result = self._collect((str(sample_audio_dir / "meeting.mp3"),))
        assert len(result) == 1
        assert result[0].name == "meeting.mp3"

    def test_directory_flat(self, sample_audio_dir):
        result = self._collect((str(sample_audio_dir),))
        names = {f.name for f in result}
        assert "meeting.mp3" in names
        assert "podcast.m4a" in names
        assert "draft_take.mp3" in names
        # Should NOT include non-audio
        assert "notes.txt" not in names
        # Should NOT include subdirectory files (non-recursive)
        assert "interview.wav" not in names

    def test_directory_recursive(self, sample_audio_dir):
        result = self._collect((str(sample_audio_dir),), recursive=True)
        names = {f.name for f in result}
        assert "meeting.mp3" in names
        assert "interview.wav" in names  # From subdirectory
        assert "backup.mp3" in names

    def test_extension_filter(self, sample_audio_dir):
        result = self._collect(
            (str(sample_audio_dir),),
            extensions="m4a",
        )
        names = {f.name for f in result}
        assert names == {"podcast.m4a"}

    def test_exclude_pattern(self, sample_audio_dir):
        result = self._collect(
            (str(sample_audio_dir),),
            exclude="*draft*",
        )
        names = {f.name for f in result}
        assert "meeting.mp3" in names
        assert "draft_take.mp3" not in names  # Excluded

    def test_deduplication(self, sample_audio_dir):
        """Same file given twice should appear once."""
        mp3 = str(sample_audio_dir / "meeting.mp3")
        result = self._collect((mp3, mp3))
        assert len(result) == 1

    def test_manifest_file(self, sample_audio_dir, tmp_path):
        """--from-file reads paths from a text file."""
        manifest = tmp_path / "files.txt"
        manifest.write_text(
            f"{sample_audio_dir / 'meeting.mp3'}\n"
            f"# comment line\n"
            f"{sample_audio_dir / 'podcast.m4a'}\n"
        )
        result = self._collect((), from_file=str(manifest))
        names = {f.name for f in result}
        assert names == {"meeting.mp3", "podcast.m4a"}

    def test_empty_input(self):
        result = self._collect(())
        assert result == []


class TestOutputResolver:
    """Test resolve_output() and parse_formats()."""

    def test_no_output_no_format_returns_none(self):
        from audiobench.cli.io.output_resolver import resolve_output

        path, fmt = resolve_output("input.mp3", None, None, "txt")
        assert path is None
        assert fmt == "txt"

    def test_format_flag_auto_names(self, tmp_path):
        from audiobench.cli.io.output_resolver import resolve_output

        inp = str(tmp_path / "input.mp3")
        Path(inp).write_text("fake")

        path, fmt = resolve_output(inp, None, "srt", "txt")
        assert fmt == "srt"
        assert path.endswith(".srt")

    def test_output_path_detects_format(self, tmp_path):
        from audiobench.cli.io.output_resolver import resolve_output

        inp = str(tmp_path / "input.mp3")
        out = str(tmp_path / "output.vtt")

        path, fmt = resolve_output(inp, out, None, "txt")
        assert fmt == "vtt"
        assert path == out

    def test_output_dir_creates_file(self, tmp_path):
        from audiobench.cli.io.output_resolver import resolve_output

        inp = str(tmp_path / "input.mp3")
        out_dir = str(tmp_path / "out") + "/"
        (tmp_path / "out").mkdir()

        path, fmt = resolve_output(inp, out_dir, "srt", "txt")
        assert fmt == "srt"
        assert "input.srt" in path

    def test_collision_skip(self, tmp_path):
        from audiobench.cli.io.output_resolver import resolve_collision

        existing = tmp_path / "output.srt"
        existing.write_text("existing")

        result = resolve_collision(str(existing), "skip")
        assert result is None

    def test_collision_rename(self, tmp_path):
        from audiobench.cli.io.output_resolver import resolve_collision

        existing = tmp_path / "output.srt"
        existing.write_text("existing")

        result = resolve_collision(str(existing), "rename")
        assert result is not None
        assert "_1" in result

    def test_collision_overwrite(self, tmp_path):
        from audiobench.cli.io.output_resolver import resolve_collision

        existing = tmp_path / "output.srt"
        existing.write_text("existing")

        result = resolve_collision(str(existing), "overwrite")
        assert result == str(existing)

    def test_parse_formats_single(self):
        from audiobench.cli.io.output_resolver import parse_formats

        assert parse_formats("srt") == ["srt"]

    def test_parse_formats_multi(self):
        from audiobench.cli.io.output_resolver import parse_formats

        result = parse_formats("srt,json")
        assert set(result) == {"srt", "json"}

    def test_parse_formats_all(self):
        from audiobench.cli.io.output_resolver import parse_formats

        result = parse_formats("all")
        assert set(result) == {"txt", "srt", "vtt", "json"}

    def test_parse_formats_none(self):
        from audiobench.cli.io.output_resolver import parse_formats

        assert parse_formats(None) == []
