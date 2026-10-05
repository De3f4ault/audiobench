"""Contextual prefix generator for hierarchical vector embeddings.

Enriches raw chunk text with structured identity metadata (Author, Work, Chapter)
prior to vector embedding. This pulls chunks into the semantic neighborhood of their
work and creator in the embedding space (e.g., distinguishing Homer from modern
commentary on Homer).

Rules:
  - Omits missing or None fields cleanly.
  - Strictly filters out dummy strings: 'Full Recording', 'Untitled', 'Unknown', 'Various'.
  - Returns empty string if no valid context metadata is available.
"""

from __future__ import annotations


_DUMMY_STRINGS = {
    "full recording",
    "untitled",
    "unknown",
    "various",
    "various artists",
    "none",
}


def build_context_prefix(
    *,
    author: str | None = None,
    work_title: str | None = None,
    chapter_title: str | None = None,
) -> str:
    """Build a contextual prefix to prepend to chunk content before embedding.

    Format:
        "[Author: {author} | Work: {work_title} | Chapter: {chapter_title}]\\n\\n"

    Args:
        author: Optional author/artist name.
        work_title: Optional work/book title.
        chapter_title: Optional chapter/track title.

    Returns:
        Formatted prefix string with trailing double newline, or "" if no
        valid metadata components exist.
    """
    parts = []

    if author and author.strip():
        clean_author = author.strip()
        if clean_author.lower() not in _DUMMY_STRINGS:
            parts.append(f"Author: {clean_author}")

    if work_title and work_title.strip():
        clean_work = work_title.strip()
        if clean_work.lower() not in _DUMMY_STRINGS:
            parts.append(f"Work: {clean_work}")

    if chapter_title and chapter_title.strip():
        clean_chapter = chapter_title.strip()
        if clean_chapter.lower() not in _DUMMY_STRINGS:
            parts.append(f"Chapter: {clean_chapter}")

    if not parts:
        return ""

    return f"[{' | '.join(parts)}]\n\n"
