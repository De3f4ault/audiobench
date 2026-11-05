"""Repository — CRUD operations for bookmark data.

Provides a clean interface for managing audio file bookmarks:
- Creating point bookmarks and region markers
- Listing, searching, and navigating between bookmarks
- Updating names, notes, and types
- Export/import in JSON format
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import asc, desc

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.storage.models import AudioFileRecord, BookmarkRecord

logger = get_logger("storage.bookmark_repository")

# ── Bookmark type badge mapping ────────────────────────────
BOOKMARK_TYPES = {
    "bookmark": "[BM]",
    "highlight": "[HL]",
    "todo": "[TODO]",
    "note": "[NOTE]",
    "edit": "[CUT]",
    "clip": "[CLIP]",
    "explain": "[EXPLAIN]",
    "resume": "[>]",
}

# Ordered list for cycling through types (manual UI cycle)
BOOKMARK_TYPE_CYCLE = ["bookmark", "highlight", "todo", "note", "edit"]


def _format_timestamp(secs: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    s = int(secs)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


class BookmarkRepository:
    """CRUD operations for audio file bookmarks."""

    # ── Create ──────────────────────────────────────────────

    def add(
        self,
        audio_file_id: int,
        timestamp: float,
        *,
        name: str = "Untitled",
        bookmark_type: str = "bookmark",
        notes: str | None = None,
        transcription_id: int | None = None,
        color: str | None = None,
    ) -> int:
        """Create a point bookmark.

        Returns:
            The bookmark record ID.
        """
        with get_session() as session:
            record = BookmarkRecord(
                audio_file_id=audio_file_id,
                transcription_id=transcription_id,
                timestamp=timestamp,
                end_timestamp=None,
                name=name[:512],
                notes=notes,
                bookmark_type=bookmark_type if bookmark_type in BOOKMARK_TYPES else "bookmark",
                color=color,
            )
            session.add(record)
            session.commit()
            logger.info(
                "Created bookmark #%d at %s for audio_file #%d",
                record.id,
                _format_timestamp(timestamp),
                audio_file_id,
            )
            return record.id

    def add_region(
        self,
        audio_file_id: int,
        start: float,
        end: float,
        *,
        name: str = "Untitled Region",
        bookmark_type: str = "bookmark",
        notes: str | None = None,
        transcription_id: int | None = None,
        color: str | None = None,
    ) -> int:
        """Create a region marker (bookmark with start + end).

        Returns:
            The bookmark record ID.
        """
        # Ensure start < end
        if end < start:
            start, end = end, start

        with get_session() as session:
            record = BookmarkRecord(
                audio_file_id=audio_file_id,
                transcription_id=transcription_id,
                timestamp=start,
                end_timestamp=end,
                name=name[:512],
                notes=notes,
                bookmark_type=bookmark_type if bookmark_type in BOOKMARK_TYPES else "bookmark",
                color=color,
            )
            session.add(record)
            session.commit()
            logger.info(
                "Created region #%d %s→%s for audio_file #%d",
                record.id,
                _format_timestamp(start),
                _format_timestamp(end),
                audio_file_id,
            )
            return record.id

    # ── Read ────────────────────────────────────────────────

    def get_by_id(self, bookmark_id: int) -> dict | None:
        """Get a single bookmark by ID."""
        with get_session() as session:
            rec = session.query(BookmarkRecord).filter_by(id=bookmark_id).first()
            if rec is None:
                return None
            return self._to_dict(rec)

    def list_for_file(
        self,
        audio_file_id: int,
        *,
        type_filter: str | None = None,
    ) -> list[dict]:
        """List all bookmarks for an audio file, sorted by timestamp.

        Args:
            audio_file_id: The audio file to query.
            type_filter: Optional — only return bookmarks of this type.

        Returns:
            List of bookmark dicts.
        """
        with get_session() as session:
            query = session.query(BookmarkRecord).filter_by(audio_file_id=audio_file_id)
            if type_filter and type_filter in BOOKMARK_TYPES:
                query = query.filter_by(bookmark_type=type_filter)
            records = query.order_by(asc(BookmarkRecord.timestamp)).all()
            return [self._to_dict(r) for r in records]

    def list_all(self, *, limit: int = 50) -> list[dict]:
        """List all bookmarks across all files."""
        with get_session() as session:
            records = (
                session.query(BookmarkRecord)
                .order_by(desc(BookmarkRecord.created_at))
                .limit(limit)
                .all()
            )
            return [self._to_dict(r) for r in records]

    def get_nearest(
        self,
        audio_file_id: int,
        current_time: float,
        direction: str = "next",
        window: float | None = None,
    ) -> dict | None:
        """Find the nearest bookmark in a given direction or within a window.

        Args:
            audio_file_id: The audio file.
            current_time: Current playback position in seconds.
            direction: "next" or "prev". Only used if window is None.
            window: Optional — if provided, finds the nearest bookmark within absolute distance `window`.

        Returns:
            The nearest bookmark dict, or None.
        """
        with get_session() as session:
            query = session.query(BookmarkRecord).filter_by(audio_file_id=audio_file_id)

            if window is not None:
                from sqlalchemy import func
                rec = (
                    query.filter(
                        BookmarkRecord.timestamp >= current_time - window,
                        BookmarkRecord.timestamp <= current_time + window
                    )
                    .order_by(func.abs(BookmarkRecord.timestamp - current_time))
                    .first()
                )
            else:
                if direction == "next":
                    rec = (
                        query.filter(BookmarkRecord.timestamp > current_time + 0.5)
                        .order_by(asc(BookmarkRecord.timestamp))
                        .first()
                    )
                else:  # prev
                    rec = (
                        query.filter(BookmarkRecord.timestamp < current_time - 0.5)
                        .order_by(desc(BookmarkRecord.timestamp))
                        .first()
                    )

            return self._to_dict(rec) if rec else None

    def search(
        self,
        query: str,
        *,
        audio_file_id: int | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search bookmarks by name or notes content.

        Args:
            query: Search string (case-insensitive).
            audio_file_id: Optional — restrict to one file.
            limit: Maximum results.
        """
        with get_session() as session:
            pattern = f"%{query}%"
            q = session.query(BookmarkRecord).filter(
                (BookmarkRecord.name.ilike(pattern)) | (BookmarkRecord.notes.ilike(pattern))
            )
            if audio_file_id is not None:
                q = q.filter_by(audio_file_id=audio_file_id)
            records = q.order_by(asc(BookmarkRecord.timestamp)).limit(limit).all()
            return [self._to_dict(r) for r in records]

    # ── Update ──────────────────────────────────────────────

    def update(
        self,
        bookmark_id: int,
        *,
        name: str | None = None,
        notes: str | None = None,
        bookmark_type: str | None = None,
        color: str | None = None,
    ) -> bool:
        """Update bookmark fields. Only provided fields are changed.

        Returns:
            True if found and updated, False if not found.
        """
        with get_session() as session:
            rec = session.query(BookmarkRecord).filter_by(id=bookmark_id).first()
            if rec is None:
                return False
            if name is not None:
                rec.name = name[:512]
            if notes is not None:
                rec.notes = notes
            if bookmark_type is not None and bookmark_type in BOOKMARK_TYPES:
                rec.bookmark_type = bookmark_type
            if color is not None:
                rec.color = color
            session.commit()
            logger.info("Updated bookmark #%d", bookmark_id)
            return True

    def cycle_type(self, bookmark_id: int) -> str | None:
        """Cycle a bookmark's type to the next in the rotation.

        Returns:
            The new type string, or None if not found.
        """
        with get_session() as session:
            rec = session.query(BookmarkRecord).filter_by(id=bookmark_id).first()
            if rec is None:
                return None
            current_idx = (
                BOOKMARK_TYPE_CYCLE.index(rec.bookmark_type)
                if rec.bookmark_type in BOOKMARK_TYPE_CYCLE
                else 0
            )
            new_type = BOOKMARK_TYPE_CYCLE[(current_idx + 1) % len(BOOKMARK_TYPE_CYCLE)]
            rec.bookmark_type = new_type
            session.commit()
            logger.info("Cycled bookmark #%d type → %s", bookmark_id, new_type)
            return new_type

    # ── Delete ──────────────────────────────────────────────

    def delete(self, bookmark_id: int) -> bool:
        """Delete a single bookmark.

        Returns:
            True if found and deleted, False if not found.
        """
        with get_session() as session:
            rec = session.query(BookmarkRecord).filter_by(id=bookmark_id).first()
            if rec is None:
                return False
            session.delete(rec)
            session.commit()
            logger.info("Deleted bookmark #%d", bookmark_id)
            return True

    def delete_for_file(self, audio_file_id: int) -> int:
        """Delete all bookmarks for an audio file.

        Returns:
            Number of bookmarks deleted.
        """
        with get_session() as session:
            count = session.query(BookmarkRecord).filter_by(audio_file_id=audio_file_id).delete()
            session.commit()
            logger.info(
                "Deleted %d bookmark(s) for audio_file #%d",
                count,
                audio_file_id,
            )
            return count

    # ── Export / Import ──────────────────────────────────────

    def export_json(self, audio_file_id: int) -> str:
        """Export bookmarks for a file as JSON.

        Returns:
            JSON string.
        """
        bookmarks = self.list_for_file(audio_file_id)
        export_data = [
            {
                "timestamp": b["timestamp"],
                "end_timestamp": b["end_timestamp"],
                "name": b["name"],
                "type": b["bookmark_type"],
                "notes": b["notes"],
            }
            for b in bookmarks
        ]
        return json.dumps(export_data, indent=2)

    def import_json(
        self,
        audio_file_id: int,
        data: str | list,
        *,
        transcription_id: int | None = None,
    ) -> int:
        """Import bookmarks from JSON data.

        Args:
            audio_file_id: Target audio file.
            data: JSON string or already-parsed list of bookmark dicts.
            transcription_id: Optional transcript to link.

        Returns:
            Number of bookmarks imported.
        """
        if isinstance(data, str):
            items = json.loads(data)
        else:
            items = data

        count = 0
        for item in items:
            ts = item.get("timestamp")
            if ts is None:
                continue

            end_ts = item.get("end_timestamp")
            name = item.get("name", "Imported")
            btype = item.get("type", "bookmark")
            notes = item.get("notes")

            if end_ts is not None:
                self.add_region(
                    audio_file_id,
                    ts,
                    end_ts,
                    name=name,
                    bookmark_type=btype,
                    notes=notes,
                    transcription_id=transcription_id,
                )
            else:
                self.add(
                    audio_file_id,
                    ts,
                    name=name,
                    bookmark_type=btype,
                    notes=notes,
                    transcription_id=transcription_id,
                )
            count += 1

        logger.info("Imported %d bookmark(s) for audio_file #%d", count, audio_file_id)
        return count

    # ── Audacity Label Format ─────────────────────────────────

    def export_audacity(self, audio_file_id: int) -> str:
        """Export bookmarks as Audacity label track (TSV).

        Format: ``start\\tend\\tname`` per line.
        Point labels use the same value for start and end.

        Returns:
            Tab-separated string.
        """
        bookmarks = self.list_for_file(audio_file_id)
        lines = []
        for b in bookmarks:
            start = f"{b['timestamp']:.6f}"
            end = f"{b['end_timestamp']:.6f}" if b["end_timestamp"] else start
            lines.append(f"{start}\t{end}\t{b['name']}")
        return "\n".join(lines)

    def import_audacity(
        self,
        audio_file_id: int,
        data: str,
        *,
        transcription_id: int | None = None,
    ) -> int:
        """Import Audacity label track (TSV).

        Args:
            audio_file_id: Target audio file.
            data: TSV string content.
            transcription_id: Optional transcript to link.

        Returns:
            Number of bookmarks imported.
        """
        count = 0
        for line in data.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                name = parts[2].strip()
            except ValueError:
                continue

            if abs(end - start) < 0.01:  # point label
                self.add(
                    audio_file_id,
                    start,
                    name=name or "Imported",
                    transcription_id=transcription_id,
                )
            else:
                self.add_region(
                    audio_file_id,
                    start,
                    end,
                    name=name or "Imported Region",
                    transcription_id=transcription_id,
                )
            count += 1

        logger.info(
            "Imported %d Audacity label(s) for audio_file #%d",
            count,
            audio_file_id,
        )
        return count

    # ── Stats ────────────────────────────────────────────────

    def count(self, audio_file_id: int | None = None) -> int:
        """Count bookmarks, optionally for a specific file."""
        with get_session() as session:
            q = session.query(BookmarkRecord)
            if audio_file_id is not None:
                q = q.filter_by(audio_file_id=audio_file_id)
            return q.count()

    # ── Helpers ──────────────────────────────────────────────

    def _get_audio_file_id_by_path(self, file_path: str) -> int | None:
        """Resolve a file path to an audio_file_id."""
        resolved = str(Path(file_path).resolve())
        with get_session() as session:
            rec = (
                session.query(AudioFileRecord).filter(AudioFileRecord.file_path == resolved).first()
            )
            return rec.id if rec else None

    @staticmethod
    def _to_dict(rec: BookmarkRecord) -> dict:
        """Convert a BookmarkRecord to a plain dict."""
        return {
            "id": rec.id,
            "audio_file_id": rec.audio_file_id,
            "transcription_id": rec.transcription_id,
            "timestamp": rec.timestamp,
            "end_timestamp": rec.end_timestamp,
            "name": rec.name,
            "notes": rec.notes,
            "bookmark_type": rec.bookmark_type,
            "color": rec.color,
            "is_region": rec.is_region,
            "created_at": rec.created_at.isoformat() if rec.created_at else "",
        }

    # ── Playback Resume Bookmarks ─────────────────────────────

    def get_resume_position(
        self,
        transcription_id: int,
        *,
        apply_courtesy_rollback: bool = True,
    ) -> float | None:
        """Fetch the saved resume position for a transcript in seconds.

        If apply_courtesy_rollback is True and the bookmark was saved more than 60 seconds ago,
        applies a 2.5-second courtesy rewind to ensure speech lead-in context.
        Returns None if no bookmark exists.
        """
        from datetime import UTC, datetime

        with get_session() as session:
            rec = (
                session.query(BookmarkRecord)
                .filter_by(transcription_id=transcription_id, bookmark_type="resume")
                .order_by(desc(BookmarkRecord.created_at))
                .first()
            )
            if not rec:
                return None

            pos = float(rec.timestamp or 0.0)
            if pos <= 0.0:
                return 0.0

            if apply_courtesy_rollback and rec.created_at:
                now = datetime.now(UTC)
                rec_created = (
                    rec.created_at
                    if rec.created_at.tzinfo
                    else rec.created_at.replace(tzinfo=UTC)
                )
                age = (now - rec_created).total_seconds()
                if age >= 60.0:
                    return max(0.0, round(pos - 2.5, 2))

            return round(pos, 2)

    def save_resume_position(
        self,
        transcription_id: int,
        position: float,
        *,
        audio_file_id: int | None = None,
        duration: float = 0.0,
    ) -> float:
        """Persist or update the resume position for a transcription.

        - If position >= duration - 3.0 (or >= 98% of duration), resets bookmark to 0.0 (track finished).
        - Upserts BookmarkRecord with bookmark_type="resume".
        - Returns the effectively saved position.
        """
        from datetime import UTC, datetime

        eff_pos = max(0.0, float(position))
        if duration > 0.0 and (eff_pos >= duration - 3.0 or eff_pos >= duration * 0.98):
            eff_pos = 0.0

        with get_session() as session:
            af_id = audio_file_id
            if af_id is None:
                from audiobench.storage.models import TranscriptionRecord

                tx = (
                    session.query(TranscriptionRecord)
                    .filter_by(id=transcription_id)
                    .first()
                )
                if tx and tx.audio_file_id:
                    af_id = tx.audio_file_id

            if af_id is None:
                logger.warning(
                    "Cannot save resume position for tx #%s: missing audio_file_id",
                    transcription_id,
                )
                return eff_pos

            rec = (
                session.query(BookmarkRecord)
                .filter_by(transcription_id=transcription_id, bookmark_type="resume")
                .first()
            )
            if rec:
                rec.timestamp = round(eff_pos, 2)
                rec.audio_file_id = af_id
                rec.created_at = datetime.now(UTC)
            else:
                rec = BookmarkRecord(
                    audio_file_id=af_id,
                    transcription_id=transcription_id,
                    timestamp=round(eff_pos, 2),
                    name="Resume Position",
                    bookmark_type="resume",
                    created_at=datetime.now(UTC),
                )
                session.add(rec)
            session.commit()
            logger.debug(
                "Saved resume position for tx #%s: %.2fs (af #%s)",
                transcription_id,
                eff_pos,
                af_id,
            )
            return eff_pos
