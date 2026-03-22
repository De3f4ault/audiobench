"""Base output formatter and formatter registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from audiobench.core.error_types import OutputFormatError
from audiobench.transcribe.transcription_result import Transcript

_FORMATTERS: dict[str, type[OutputFormatter]] = {}


class OutputFormatter(ABC):
    """Abstract base for output formatters."""

    @abstractmethod
    def format(self, transcript: Transcript) -> str:
        """Format a transcript into a string output."""

    @staticmethod
    @abstractmethod
    def extension() -> str:
        """File extension for this format (without dot)."""


def register_formatter(name: str, cls: type[OutputFormatter]) -> None:
    _FORMATTERS[name] = cls


def get_formatter(name: str) -> OutputFormatter:
    """Get a formatter instance by name."""
    _ensure_registered()
    if name not in _FORMATTERS:
        raise OutputFormatError(name, f"Unknown format. Available: {', '.join(_FORMATTERS.keys())}")
    return _FORMATTERS[name]()


def _ensure_registered() -> None:
    if _FORMATTERS:
        return
    from audiobench.output.json_fmt import JsonFormatter
    from audiobench.output.srt import SrtFormatter
    from audiobench.output.text import TextFormatter
    from audiobench.output.vtt import VttFormatter

    register_formatter("txt", TextFormatter)
    register_formatter("srt", SrtFormatter)
    register_formatter("vtt", VttFormatter)
    register_formatter("json", JsonFormatter)
