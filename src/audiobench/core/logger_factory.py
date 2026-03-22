"""Logger factory — creates configured loggers under the audiobench namespace.

Sets up rich-formatted console logging with configurable levels.
All audiobench modules log under the 'audiobench' namespace.

Usage:
    from audiobench.core.logger_factory import setup_logging, get_logger

    setup_logging("DEBUG")
    logger = get_logger("core.pipeline")
    logger.info("Processing audio...")
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the entire application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root audiobench logger
    logger = logging.getLogger("audiobench")
    logger.setLevel(log_level)

    # Don't propagate to root logger to avoid duplicate output
    logger.propagate = False

    # Clear existing handlers (allows re-configuration)
    logger.handlers.clear()

    # Rich console handler — pretty formatted output
    console_handler = RichHandler(
        level=log_level,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=level.upper() == "DEBUG",
    )
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(console_handler)

    # File handler — plain text for debugging
    try:

        from audiobench.core.settings import get_settings

        log_dir = get_settings().data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "audiobench.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)-40s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
    except OSError:
        # If we can't write to the log file, just use console
        logger.warning("Could not create log file, using console only")

    # Suppress noisy third-party loggers
    for noisy_logger in [
        "faster_whisper",
        "ctranslate2",
        "urllib3",
        "httpx",
        "torch",
        "pyannote",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logger.debug("Logging configured at %s level", level.upper())


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the audiobench namespace.

    Args:
        name: Module name (e.g., 'core.pipeline').

    Returns:
        Logger instance under 'audiobench.<name>'.
    """
    return logging.getLogger(f"audiobench.{name}")
