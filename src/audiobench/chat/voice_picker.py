"""Interactive Voice Picker with Audio Previews for Piper TTS."""

from __future__ import annotations

import re
from typing import Any

from audiobench.cli.display.theme import (
    ACCENT,
    DIM,
    SUCCESS,
    WARNING,
    console as default_console,
    error_panel,
)
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings, invalidate_settings_cache
from audiobench.tts.engine import PiperTTSEngine, TTSError

logger = get_logger("chat.voice_picker")

# Curated list of high-quality Piper voices
CURATED_VOICES: list[dict[str, str]] = [
    {
        "name": "en_US-amy-medium",
        "language": "en_US",
        "description": "Natural, clear American female (Default, ~60MB)",
    },
    {
        "name": "en_US-lessac-high",
        "language": "en_US",
        "description": "Studio audiobook American female (Top High Quality, ~109MB)",
    },
    {
        "name": "en_US-ryan-high",
        "language": "en_US",
        "description": "Studio narrative American male (Top High Quality, ~115MB)",
    },
    {
        "name": "en_US-ryan-medium",
        "language": "en_US",
        "description": "Calm, narrative American male (Balanced, ~60MB)",
    },
    {
        "name": "en_US-lessac-medium",
        "language": "en_US",
        "description": "Fast, balanced American female (~60MB)",
    },
    {
        "name": "en_GB-cori-high",
        "language": "en_GB",
        "description": "Studio British English female (Top High Quality, ~109MB)",
    },
    {
        "name": "en_GB-alan-medium",
        "language": "en_GB",
        "description": "British English male narrator (~60MB)",
    },
    {
        "name": "de_DE-thorsten-high",
        "language": "de_DE",
        "description": "Studio German male (Top High Quality, ~109MB)",
    },
    {
        "name": "de_DE-thorsten-medium",
        "language": "de_DE",
        "description": "Standard German male (~60MB)",
    },
    {
        "name": "en_US-danny-low",
        "language": "en_US",
        "description": "Casual, low-resource American male (~60MB)",
    },
    {
        "name": "fr_FR-siwis-medium",
        "language": "fr_FR",
        "description": "Clear French female (~60MB)",
    },
    {
        "name": "es_ES-carlfm-x_low",
        "language": "es_ES",
        "description": "Standard European Spanish male (~27MB)",
    },
]


def strip_markdown_for_speech(text: str) -> str:
    """Clean markdown, code blocks, and citation markers for natural TTS speech."""
    if not text:
        return ""
    # Remove code blocks
    cleaned = re.sub(r"```[\s\S]*?```", " (code snippet omitted) ", text)
    # Remove inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Remove markdown link markup [text](url) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # Remove citation brackets like [1], [2], [1, 2]
    cleaned = re.sub(r"\[\d+(?:[,\s]+\d+)*\]", "", cleaned)
    # Remove rich formatting tags like [bold], [/bold]
    cleaned = re.sub(r"\[/?[a-zA-Z0-9_\s#]+\]", "", cleaned)
    # Remove markdown headers and bullets
    cleaned = re.sub(r"^[#*>\-]+\s+", "", cleaned, flags=re.MULTILINE)
    # Remove bold / italic symbols
    cleaned = re.sub(r"[*_~]+", "", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def get_voice_download_size_str(voice_name: str) -> str:
    """Return an accurate string for voice download size, querying HEAD or fallback heuristics."""
    # Fast path: exact known sizes for curated voices
    KNOWN_VOICE_SIZES = {
        "en_US-amy-medium": "~60MB",
        "en_US-lessac-high": "~109MB",
        "en_US-ryan-high": "~115MB",
        "en_US-ryan-medium": "~60MB",
        "en_US-lessac-medium": "~60MB",
        "en_GB-cori-high": "~109MB",
        "en_GB-alan-medium": "~60MB",
        "de_DE-thorsten-high": "~109MB",
        "de_DE-thorsten-medium": "~60MB",
        "en_US-danny-low": "~60MB",
        "fr_FR-siwis-medium": "~60MB",
        "es_ES-carlfm-x_low": "~27MB",
    }
    if voice_name in KNOWN_VOICE_SIZES:
        return KNOWN_VOICE_SIZES[voice_name]

    parts = voice_name.split("-")
    if len(parts) >= 3:
        lang_code = parts[0]
        lang = lang_code.split("_")[0]
        speaker = parts[1]
        quality = parts[2]
        url = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{lang}/{lang_code}/{speaker}/{quality}/{voice_name}.onnx"
        try:
            import urllib.request
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                cl = resp.headers.get("Content-Length")
                if cl and cl.isdigit():
                    size_mb = round(int(cl) / (1024 * 1024))
                    return f"~{size_mb}MB"
        except Exception:
            pass

    # Heuristic fallback based on quality tier
    q = parts[-1].lower() if parts else ""
    if "high" in q:
        return "~110MB"
    elif "x_low" in q:
        return "~27MB"
    return "~60MB"


def ensure_voice_downloaded(
    voice_name: str,
    console: Any = default_console,
    prompt_user: bool = True,
) -> bool:
    """Ensure a Piper voice model is present locally. If missing, prompt and download.

    Returns:
        True if the voice is available (already present or downloaded), False otherwise.
    """
    settings = get_settings()
    engine = PiperTTSEngine(voices_dir=settings.voices_dir)
    model_path = engine._voices_dir / f"{voice_name}.onnx"
    config_path = engine._voices_dir / f"{voice_name}.onnx.json"

    if model_path.exists() and config_path.exists():
        return True

    from audiobench.chat.voice_session import is_internet_connected

    if not is_internet_connected():
        console.print(f"  [yellow][Voice] Voice model '{voice_name}' is not downloaded and cannot download while offline.[/yellow]")
        local_voices = engine.list_voices()
        if local_voices:
            names = ", ".join(v.name for v in local_voices)
            console.print(f"  [{DIM}]Available offline voices: {names}. Use /voice picker to switch.[/]")
        return False

    if prompt_user:
        size_str = get_voice_download_size_str(voice_name)
        try:
            ans = input(f"  > [Voice] Voice '{voice_name}' is not downloaded ({size_str}). Download now? [Y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print()
            return False

        if ans not in ("", "y", "yes"):
            console.print("  [dim]Download cancelled.[/dim]")
            return False

    console.print(f"  [{ACCENT}][Voice] Downloading voice model: {voice_name}...[/]")
    try:
        engine.download_voice(voice_name)
        console.print(f"  [{SUCCESS}][Voice] Downloaded {voice_name} successfully.[/]")
        return True
    except Exception as e:
        console.print(f"  [yellow]Failed to download voice: {e}[/yellow]")
        return False


def play_voice_audition(
    voice_name: str,
    console: Any = default_console,
    sample_text: str | None = None,
) -> bool:
    """Synthesize and play a short audition snippet for a given voice model."""
    settings = get_settings()
    engine = PiperTTSEngine(voices_dir=settings.voices_dir)

    # Check if voice model files exist
    model_path = engine._voices_dir / f"{voice_name}.onnx"
    if not model_path.exists():
        console.print(f"  [{WARNING}]Voice '{voice_name}' is not downloaded yet.[/]")
        return False

    short_name = voice_name.split("-")[1].capitalize() if "-" in voice_name else voice_name
    text_to_speak = sample_text or f"Hello! I am {short_name}, your offline voice assistant for Audiobench."

    console.print(f"  [cyan][Voice][/cyan] [dim]Auditioning voice: {voice_name}...[/dim]")
    try:
        engine.play(text_to_speak, voice=voice_name)
        return True
    except TTSError as err:
        console.print(error_panel("Audition Error", str(err)))
        return False
    except Exception as exc:
        console.print(f"  [red]Audition failed: {exc}[/red]")
        return False


def show_voice_picker(
    console: Any = default_console,
    current_voice: str | None = None,
) -> str | None:
    """Display an interactive voice picker menu with audio previews and downloads.

    Returns:
        The selected voice name if changed, or None if kept/cancelled.
    """
    settings = get_settings()
    current = current_voice or settings.tts_voice
    engine = PiperTTSEngine(voices_dir=settings.voices_dir)

    # Get local voices
    local_voices = {v.name: v for v in engine.list_voices()}

    # Merge curated and local
    voice_items: list[dict[str, Any]] = []
    seen = set()

    for cv in CURATED_VOICES:
        name = cv["name"]
        seen.add(name)
        is_cached = name in local_voices
        voice_items.append({
            "name": name,
            "language": cv["language"],
            "description": cv["description"],
            "downloaded": is_cached,
        })

    # Include any additional downloaded voices not in curated list
    for name, vinfo in local_voices.items():
        if name not in seen:
            seen.add(name)
            voice_items.append({
                "name": name,
                "language": vinfo.language,
                "description": f"Local {vinfo.language} {vinfo.quality} voice",
                "downloaded": True,
            })

    while True:
        console.print()
        console.print("  [bold]Available Text-to-Speech Voices (Piper Offline TTS):[/bold]")
        for i, item in enumerate(voice_items, 1):
            name = item["name"]
            status_tag = f"[{SUCCESS}][Downloaded][/]" if item["downloaded"] else f"[{DIM}][Not Downloaded][/]"
            current_tag = f"  [{SUCCESS}]<- current[/]" if name == current else ""
            desc = item["description"]
            console.print(f"    [[bold cyan]{i}[/bold cyan]] {name:<24} {status_tag:<22} [dim]{desc}[/dim]{current_tag}")

        console.print()
        console.print(f"    [[bold cyan]P <num>[/bold cyan]] Preview / Audition voice snippet")
        console.print(f"    [[bold cyan]0 / q[/bold cyan]] Keep current / Exit ([dim]{current}[/dim])")
        console.print()

        try:
            raw = input("    > Pick voice [0/q] or P <num> to audition: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            return None

        if not raw or raw.lower() in ("0", "q", "quit", "exit"):
            console.print(f"  [{DIM}]Kept current voice: {current}[/]")
            return None

        # Check for preview audition command: "P 2" or "p2" or "play 2"
        preview_match = re.match(r"^[pP](?:review|lay)?\s*(\d+)$", raw)
        if preview_match:
            idx = int(preview_match.group(1)) - 1
            if 0 <= idx < len(voice_items):
                target_item = voice_items[idx]
                target_name = target_item["name"]
                if not target_item["downloaded"]:
                    if ensure_voice_downloaded(target_name, console=console):
                        target_item["downloaded"] = True
                    else:
                        continue
                play_voice_audition(target_name, console)
            else:
                console.print("  [red]Invalid voice index for preview.[/red]")
            continue

        # Check for selection
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(voice_items):
                selected_item = voice_items[idx]
                selected_name = selected_item["name"]

                if not selected_item["downloaded"]:
                    if ensure_voice_downloaded(selected_name, console=console):
                        selected_item["downloaded"] = True
                    else:
                        console.print("  [dim]Selection cancelled.[/dim]")
                        continue

                # Confirm and set
                settings.tts_voice = selected_name
                try:
                    settings.save()
                    invalidate_settings_cache()
                    console.print(f"  [{SUCCESS}][DONE] Default TTS voice set to: [bold]{selected_name}[/bold][/]")
                except Exception as e:
                    console.print(f"  [yellow]Failed to persist settings: {e}[/yellow]")

                return selected_name
            else:
                console.print("  [red]Invalid choice.[/red]")
        else:
            console.print("  [red]Please enter a valid number or 'P <num>' to preview.[/red]")
