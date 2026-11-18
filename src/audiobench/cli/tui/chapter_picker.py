

def pick_chapters(audio_file_id: int) -> list[int]:
    """Interactively prompt the user to pick chapters from an audio file.

    Returns a list of chapter IDs selected by the user.
    """

    from audiobench.cli.display.theme import console
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import ChapterRecord

    with get_session() as session:
        chapters = session.query(ChapterRecord).filter_by(audio_file_id=audio_file_id).order_by(ChapterRecord.start_time).all()

    if not chapters:
        return []

    console.print("\n[bold]Select Chapters[/bold]")
    for i, ch in enumerate(chapters, 1):
        console.print(f"  {i}. Chapter {i} ({ch.transcript_length or 0} chars)")

    while True:
        try:
            choice = input("\nSelect chapters (e.g., '1', '1,3,5', '2-6', 'all'): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return []

        if not choice:
            continue

        if choice == "all":
            return [ch.id for ch in chapters]

        selected_indices = set()
        parts = [p.strip() for p in choice.split(',')]
        valid = True

        for part in parts:
            if '-' in part:
                try:
                    start_str, end_str = part.split('-', 1)
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                    if 1 <= start <= end <= len(chapters):
                        selected_indices.update(range(start, end + 1))
                    else:
                        valid = False
                except ValueError:
                    valid = False
            else:
                try:
                    idx = int(part)
                    if 1 <= idx <= len(chapters):
                        selected_indices.add(idx)
                    else:
                        valid = False
                except ValueError:
                    valid = False

        if valid and selected_indices:
            # return the actual chapter DB IDs
            return [chapters[i-1].id for i in sorted(list(selected_indices))]

        console.print("[red]Invalid selection. Please try again.[/red]")
