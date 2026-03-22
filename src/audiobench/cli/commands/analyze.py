"""Analytics commands — vocab, word frequency, and transcript analysis."""

from __future__ import annotations

import sys

import click

from audiobench.cli.display.theme import (
    ACCENT,
    DIM,
    console,
    error_panel,
    make_table,
    stdout,
)

# ── Vocab Command ───────────────────────────────────────────


@click.command()
@click.argument("transcription_id", type=int)
@click.option("--top", default=30, help="Number of top words to show")
@click.option(
    "--min-length",
    default=3,
    help="Minimum word length to include",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    show_default=True,
    help="Output format",
)
def vocab(
    transcription_id: int,
    top: int,
    min_length: int,
    output_format: str,
) -> None:
    """Analyze word frequency in a transcription.

    \b
    Examples:
      audiobench vocab 3                      Top 30 words
      audiobench vocab 3 --top 50             Top 50 words
      audiobench vocab 3 --min-length 5       Only words ≥5 chars
      audiobench vocab 3 --format json        JSON output
      audiobench vocab 3 --format csv         CSV for spreadsheets
    """
    import json as json_lib
    import re
    from collections import Counter

    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()
    data = repo.get_by_id(transcription_id)

    if not data:
        console.print(error_panel(f"Transcription #{transcription_id} not found"))
        sys.exit(1)

    # Extract all words from segments
    text = data.get("full_text", "")
    if not text:
        # Fallback: join segment texts
        text = " ".join(s.get("text", "") for s in data.get("segments", []))

    # Tokenise: lowercase, strip punctuation
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if len(w) >= min_length]

    if not words:
        console.print(f"  [{DIM}]No words found (try lowering --min-length)[/]")
        return

    counter = Counter(words)
    total = len(words)
    top_words = counter.most_common(top)

    if output_format == "json":
        result = [
            {"word": w, "count": c, "percent": round(c / total * 100, 2)} for w, c in top_words
        ]
        stdout.print(json_lib.dumps(result, indent=2), highlight=False)
        return

    if output_format == "csv":
        print("word,count,percent")
        for w, c in top_words:
            print(f"{w},{c},{c / total * 100:.2f}")
        return

    # Table format
    table = make_table(
        f"Word Frequency — #{transcription_id} ({data.get('file_name', '?')})",
        [
            ("#", {"style": DIM, "width": 4}),
            ("Word", {"style": ACCENT}),
            ("Count", {"justify": "right", "width": 7}),
            ("%", {"justify": "right", "width": 7}),
            ("Bar", {"width": 20}),
        ],
    )

    max_count = top_words[0][1] if top_words else 1
    for i, (word, count) in enumerate(top_words, 1):
        pct = count / total * 100
        bar_len = int(count / max_count * 18)
        bar = "█" * bar_len
        table.add_row(str(i), word, str(count), f"{pct:.1f}", f"[{ACCENT}]{bar}[/]")

    console.print(table)
    console.print(
        f"  [{DIM}]Unique words: {len(counter)} • "
        f"Total words: {total} • "
        f"Showing top {min(top, len(top_words))}[/]"
    )
