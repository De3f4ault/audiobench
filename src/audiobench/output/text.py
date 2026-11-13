"""Plain text output formatter."""

from audiobench.output.base import OutputFormatter
from audiobench.transcribe.transcription_result import Transcript


class TextFormatter(OutputFormatter):
    """Format transcript as plain text, grouped by speaker if available.

    When word-level speaker labels are present, speaker changes *within*
    a segment are detected and the text is split at the boundary so that
    interruptions and overlapping turns are shown correctly.
    """

    def format(self, transcript: Transcript) -> str:
        lines: list[str] = []
        current_speaker: str | None = None
        current_chapter_id: int | None = None

        # Build chapter map
        chapter_map = (
            {c["id"]: c["title"] for c in transcript.chapters} if transcript.chapters else {}
        )

        # Insert Markdown TOC if chapters exist
        if chapter_map:
            lines.append("# Table of Contents\n")
            for c in transcript.chapters:
                # Basic slugification for markdown anchor links
                slug = c["title"].lower().replace(" ", "-").replace(":", "").replace("'", "")
                lines.append(f"- [{c['title']}](#{slug})")
            lines.append("\n---\n")

        for seg in transcript.segments:
            # ── Check for chapter change ──
            if seg.chapter_id != current_chapter_id and seg.chapter_id:
                if lines and lines[-1] != "":
                    lines.append("")
                if current_chapter_id:
                    lines.append("\n---\n")  # Separator between chapters

                title = chapter_map.get(seg.chapter_id, f"Chapter {seg.chapter_id}")
                lines.append(f"## {title}\n")
                current_chapter_id = seg.chapter_id

                # Reset speaker to force re-printing speaker badge after a chapter heading
                current_speaker = None

            # Check if any words carry individual speaker labels
            has_word_speakers = any(w.speaker for w in seg.words)

            if has_word_speakers:
                # ── Word-level: detect speaker changes within the segment ──
                current_chunk: list[str] = []
                chunk_speaker: str | None = None

                for word in seg.words:
                    spk = word.speaker or seg.speaker  # fall back to segment label

                    if spk != chunk_speaker:
                        # Flush the previous chunk
                        if current_chunk and chunk_speaker:
                            if chunk_speaker != current_speaker:
                                if lines:
                                    lines.append("")
                                display_speaker = transcript.speaker_map.get(
                                    chunk_speaker, chunk_speaker
                                )
                                lines.append(f"[{display_speaker}]")
                                current_speaker = chunk_speaker
                            lines.append(" ".join(current_chunk))
                            current_chunk = []
                        chunk_speaker = spk

                    current_chunk.append(word.word)

                # Flush the final chunk
                if current_chunk:
                    if chunk_speaker and chunk_speaker != current_speaker:
                        if lines:
                            lines.append("")
                        display_speaker = transcript.speaker_map.get(chunk_speaker, chunk_speaker)
                        lines.append(f"[{display_speaker}]")
                        current_speaker = chunk_speaker
                    lines.append(" ".join(current_chunk))
            else:
                # ── Segment-level: original behaviour ─────────────────────
                if seg.speaker and seg.speaker != current_speaker:
                    if lines:
                        lines.append("")
                    display_speaker = transcript.speaker_map.get(seg.speaker, seg.speaker)
                    lines.append(f"[{display_speaker}]")
                    current_speaker = seg.speaker
                lines.append(seg.text)

        return "\n".join(lines) + "\n"

    @staticmethod
    def extension() -> str:
        return "txt"
