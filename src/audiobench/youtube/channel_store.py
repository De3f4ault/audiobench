"""YouTube channel workspace persistence layer.

This module is the single source of truth for channel nodes. It deliberately
does NOT interact with LanceDB. The whiteboard_text on a channel node is an
operational note — it orients the user on return. It is not corpus content
and must never be embedded or made searchable.

Key contract:
    compute_engagement_weight(channel) — pure function, never stored.
    The absence of events decays the weight naturally through elapsed time.
    If weight were stored, a channel you abandon would keep its last-computed
    score forever — nothing decays it, because decay is not an event.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from audiobench.core.logger_factory import get_logger
from audiobench.storage.models import YouTubeChannelNode, YouTubePlaylistCache

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = get_logger("youtube.channel_store")

# Decay half-life: a channel visited N days ago has weight exp(-N / HALF_LIFE_DAYS).
# At 30 days: ~0.37  |  at 60 days: ~0.14  |  at 90 days: ~0.05
HALF_LIFE_DAYS: float = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Engagement weight — computed on read, never stored
# ─────────────────────────────────────────────────────────────────────────────

def compute_engagement_weight(channel: YouTubeChannelNode) -> float:
    """Derive a [0.0, 1.0] engagement score from last_visited_at.

    This is a pure function — it reads no DB state, stores nothing, and has no
    side effects. Call it at home-screen render time to sort channels.

    Decay curve: weight = exp(-days_since_visit / HALF_LIFE_DAYS)
      - Visited today      → ~1.0
      - Visited 30 days ago → ~0.37
      - Visited 90 days ago → ~0.05
      - Never visited       → 0.0

    Phase 4 will extend this by adding a windowed Observatory event count
    as a secondary signal. The formula is designed to accept that additive
    extension without restructuring.
    """
    if channel.last_visited_at is None:
        return 0.0
    now = datetime.now(UTC)
    last_visited = channel.last_visited_at
    # Normalise to UTC if naive
    if last_visited.tzinfo is None:
        last_visited = last_visited.replace(tzinfo=UTC)
    days_since = max(0.0, (now - last_visited).total_seconds() / 86400.0)
    return math.exp(-days_since / HALF_LIFE_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# Channel node CRUD
# ─────────────────────────────────────────────────────────────────────────────

def get_channel(channel_id: str, session: Session) -> YouTubeChannelNode | None:
    """Return the channel node for a YouTube channel ID, or None."""
    return session.query(YouTubeChannelNode).filter_by(channel_id=channel_id).first()


def get_all_channels(session: Session) -> list[YouTubeChannelNode]:
    """Return all channel nodes, ordered by engagement weight (most engaged first).

    Ordering is computed in Python (not SQL) because engagement weight is a pure
    function of last_visited_at and is not stored on the model.
    """
    channels = session.query(YouTubeChannelNode).all()
    return sorted(channels, key=compute_engagement_weight, reverse=True)


def get_or_create_channel(
    channel_id: str,
    title: str,
    session: Session,
    thumbnail_url: str | None = None,
) -> tuple[YouTubeChannelNode, bool]:
    """Return (channel_node, created). Creates the node if it doesn't exist.

    Does not mark the channel as visited — call mark_visited() separately when
    the user actively opens the workspace.
    """
    existing = get_channel(channel_id, session)
    if existing:
        # Update title/thumbnail if they changed on YouTube — but never overwrite
        # a real title with an empty string. Callers that don't have a fresh title
        # pass "" (e.g. workspace entry), and that should not erase the stored title.
        if title and existing.title != title:
            existing.title = title
            logger.debug(f"Updated channel title: {channel_id} → {title!r}")
        if thumbnail_url and existing.thumbnail_url != thumbnail_url:
            existing.thumbnail_url = thumbnail_url
        return existing, False

    node = YouTubeChannelNode(
        channel_id=channel_id,
        title=title,
        thumbnail_url=thumbnail_url,
    )
    session.add(node)
    session.flush()  # get the id without full commit
    logger.info(f"Created channel node: {channel_id} ({title!r})")
    return node, True


def mark_visited(channel_id: str, session: Session) -> None:
    """Record that the user opened this channel's workspace right now.

    This is the only write that affects engagement weight. Called on workspace
    entry — not on search, not on fetch, not on note update.
    """
    channel = get_channel(channel_id, session)
    if channel:
        channel.last_visited_at = datetime.now(UTC)
        logger.debug(f"Marked visited: {channel_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Whiteboard CRUD
# ─────────────────────────────────────────────────────────────────────────────

def get_whiteboard(channel_id: str, session: Session) -> str | None:
    """Return the current whiteboard text for a channel, or None if not set."""
    channel = get_channel(channel_id, session)
    return channel.whiteboard_text if channel else None


def update_whiteboard(channel_id: str, text: str, session: Session) -> None:
    """Overwrite the channel whiteboard note.

    IMPORTANT: this text is operational — it is never sent to LanceDB.
    Do not call the embedding pipeline after this write.
    The whiteboard is a whiteboard: readable by the user, readable by the system
    to draft the next visit prompt. Nothing else.
    """
    channel = get_channel(channel_id, session)
    if not channel:
        raise ValueError(f"Channel node not found: {channel_id}")
    channel.whiteboard_text = text.strip() or None
    channel.whiteboard_updated_at = datetime.now(UTC)
    logger.debug(f"Whiteboard updated: {channel_id} ({len(text)} chars)")


# ─────────────────────────────────────────────────────────────────────────────
# Library count (stub — filled by Phase 2 via playlist cache)
# ─────────────────────────────────────────────────────────────────────────────

def get_library_count(channel_id: str, session: Session) -> int:
    """Return the number of videos from this channel in the library.

    Counts AudioFileRecord rows with a matching youtube_video_id that have
    been transcribed (have at least one expression). Does not depend on the
    playlist cache — queries the library directly.
    """
    from audiobench.storage.models import AudioFileRecord
    return (
        session.query(AudioFileRecord)
        .filter(
            AudioFileRecord.youtube_video_id.isnot(None),
            AudioFileRecord.youtube_channel_id == channel_id,
        )
        .count()
    )


def get_playlist_cache(channel_id: str, session: Session) -> YouTubePlaylistCache | None:
    """Return the cached playlist record for a channel, or None."""
    return session.query(YouTubePlaylistCache).filter_by(channel_id=channel_id).first()
