from datetime import date
from typing import TypedDict, Sequence
from rich.console import Console

class PickerSession(TypedDict):
    id: int
    title: str | None
    created_at: str | None
    detail: str  # e.g., "4 searches", "12 msgs"

def show_session_picker(
    console: Console,
    sessions: Sequence[PickerSession],
    noun: str = "session",
) -> int | None:
    """Show an interactive session picker and return the chosen session ID, or None for new.
    
    Args:
        console: Rich console for output.
        sessions: List of dicts with 'id', 'title', 'created_at', and 'detail'.
                  These should ALREADY BE FILTERED to exclude empty sessions and sliced to limit.
        noun: The word to use for prompt text ("session", "conversation").
        
    Returns:
        int: The selected session ID.
        None: If the user chose to start a new session.
        
    Raises:
        KeyboardInterrupt: If the user cancels the prompt.
    """
    if not sessions:
        return None

    term_width = console.width or 100
    # Reserve space for ID, detail string, and timestamp
    title_max = max(20, term_width - 40)

    console.print(f"  [dim]Recent {noun}s:[/dim]")
    for idx, r in enumerate(sessions, 1):
        title = r["title"] or "(untitled)"
        if len(title) > title_max:
            title = title[:title_max - 1] + "…"
            
        today = date.today().isoformat()
        ts = r["created_at"][:16].replace("T", " ") if r["created_at"] else ""
        if ts.startswith(today):
            ts = ts[11:]  # just the time for today's sessions

        console.print(
            f"  [cyan][{idx}][/cyan] [dim]#{r['id']}[/dim] {title} "
            f"[dim]· {r['detail']} · {ts}[/dim]"
        )

    console.print()
    prompt_range = f"1-{len(sessions)}" if len(sessions) > 1 else "1"
    console.print(f"  [dim]Resume {noun} [[{prompt_range}]] or type [cyan]'n'[/cyan] for a new {noun} (default: 1)?[/dim]")
    
    try:
        ans = input("  › ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        console.print("\n  [dim]Cancelled.[/dim]")
        raise KeyboardInterrupt()

    console.print()

    if ans in ("", "y", "yes"):
        ans = "1"

    if ans == "n":
        return None

    raw_ans = ans.lstrip("#")
    if raw_ans.isdigit():
        val = int(raw_ans)
        if ans.startswith("#"):
            # Direct ID lookup (within the provided list)
            match = next((s for s in sessions if s["id"] == val), None)
            return match["id"] if match else None
        else:
            # 1-indexed list position
            if 1 <= val <= len(sessions):
                return sessions[val - 1]["id"]
            else:
                # Numeric fallback: treat as raw ID
                match = next((s for s in sessions if s["id"] == val), None)
                return match["id"] if match else None

    return None
