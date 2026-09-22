"""YouTube uploads playlist cache — Phase 2.

Responsible for:
  - Fetching a channel's full uploads playlist via playlistItems.list (1 unit/page)
  - Storing results in YouTubePlaylistCache with the locked videos_json shape
  - Detecting new uploads since last visit (1–2 units on workspace entry)
  - Computing the gap: available videos not yet in the library
  - Respecting the 24h TTL from the Phase 0 spike findings

API cost summary (Phase 0 spike, @KFoundation channel, 2250 videos):
  - Full catalog refresh:  ~45 units  (45 pages × 1 unit)
  - New-upload check:        1 unit   (page 1 only; 2 if count_delta > 50)
  - Daily budget:        10,000 units → 222 full refreshes/day available

Availability values in videos_json (locked in Phase 1 schema):
  "public"            — normal video, fetchable
  "unlisted"          — accessible, included in playlist, fetchable
  "private_or_removed" — API placeholder; excluded from gap and denominator
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.storage.models import YouTubePlaylistCache

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = get_logger("youtube.playlist")

# Cache TTL from Phase 0 spike findings: channel uploads ~2x/week; 24h gives at
# most 1 day of staleness at a cost of 1 extra unit per workspace entry.
CACHE_TTL_HOURS: int = 24

# If count_delta > this threshold, fetch a second page to catch bulk-upload spills.
BULK_UPLOAD_THRESHOLD: int = 50


# ─────────────────────────────────────────────────────────────────────────────
# Low-level API call
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_playlist_page(
    uploads_playlist_id: str,
    page_token: str | None = None,
    max_results: int = 50,
) -> dict:
    """Fetch one page of a channel's uploads playlist.

    Cost: 1 quota unit per call.

    Returns raw API response dict with keys: items, nextPageToken, pageInfo.
    """
    settings = get_settings()
    api_key = settings.youtube_api_key
    if not api_key:
        raise RuntimeError(
            "AUDIOBENCH_YOUTUBE_API_KEY is not set. "
            "Add it to .env to use the playlist cache."
        )

    params: dict = {
        "part": "snippet,contentDetails",
        "playlistId": uploads_playlist_id,
        "maxResults": min(max_results, 50),
        "key": api_key,
    }
    if page_token:
        params["pageToken"] = page_token

    response = httpx.get(
        "https://www.googleapis.com/youtube/v3/playlistItems",
        params=params,
        timeout=15.0,
    )
    response.raise_for_status()
    return response.json()


def _uploads_playlist_id_from_channel_id(channel_id: str) -> str:
    """Derive the uploads playlist ID from a YouTube channel ID.

    The uploads playlist ID is always the channel ID with 'UC' replaced by 'UU'.
    This is a stable YouTube convention validated in the Phase 0 spike.
    Cost: 0 quota units (pure string transform).
    """
    if not channel_id.startswith("UC"):
        raise ValueError(
            f"Expected channel ID starting with 'UC', got: {channel_id!r}"
        )
    return "UU" + channel_id[2:]


def _classify_availability(item: dict) -> str:
    """Classify a playlistItem as 'public', 'unlisted', or 'private_or_removed'.

    The title strings "Private video" and "Deleted video" are YouTube-assigned
    placeholder strings, not user-settable titles. When a video in a playlist
    is private, removed, or otherwise unavailable, the API replaces the real
    title with one of these exact strings. Title alone is therefore a sufficient
    condition — the previous implementation required description to also match
    one of a known set of strings, which is overly strict: a real private video
    with a different description text (localized, empty, or future-API-changed)
    would fall through as "public", showing up as downloadable in the gap list.
    That is the wrong direction of error.

    Validation note: as of the Phase 0 spike (2026-08-31), no private/deleted
    placeholders appeared in pages 1–20 of the @KFoundation uploads playlist
    (~1000 videos). The classification logic is validated against the YouTube
    Data API v3 documentation (playlistItems.list) and the known placeholder
    string values, but has not been tested against a real private video response
    from this specific API endpoint. If a mislabelled private video is ever
    observed in the gap list, the fix is to add the real title string here.
    """
    snippet = item.get("snippet", {})
    title = snippet.get("title", "")

    # YouTube-assigned placeholder titles for private/removed/unavailable videos.
    # These are exact strings returned by the API — they are not user titles.
    # Title alone is sufficient; description is not required to match.
    if title in ("Private video", "Deleted video"):
        return "private_or_removed"

    # Unlisted: video exists and is accessible but not in public search.
    # The API doesn't expose privacyStatus in playlistItems.list, but unlisted
    # videos appear with normal titles and full metadata — so if it's not a
    # placeholder, treat it as accessible. The distinction between public and
    # unlisted only matters if we later want to show a "(unlisted)" badge.
    # For gap computation, both count as fetchable.
    return "public"


def _item_to_video_entry(item: dict) -> dict:
    """Convert a raw playlistItem API response to the locked videos_json shape."""
    snippet = item.get("snippet", {})
    content = item.get("contentDetails", {})

    return {
        "video_id": snippet.get("resourceId", {}).get("videoId", ""),
        "title": snippet.get("title", ""),
        "published_at": (snippet.get("publishedAt") or "")[:10],  # YYYY-MM-DD
        "duration_pt": content.get("videoPublishedAt", ""),  # not available in playlistItems
        "availability": _classify_availability(item),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cache refresh
# ─────────────────────────────────────────────────────────────────────────────

def refresh_channel_cache(channel_id: str, session: Session) -> YouTubePlaylistCache:
    """Fetch the full uploads playlist and store it in YouTubePlaylistCache.

    This is the expensive operation: fetches ALL pages (1 unit/page).
    For @KFoundation (2250 videos): ~45 quota units, ~20s.

    Should be called:
    - On first workspace entry for a channel (cache is empty)
    - When the cache is older than CACHE_TTL_HOURS
    - On explicit /refresh command

    For routine workspace entries (cache is fresh), use check_for_new_uploads()
    instead — it costs 1–2 units regardless of catalog size.
    """
    uploads_id = _uploads_playlist_id_from_channel_id(channel_id)
    logger.info(
        "refresh_channel_cache: fetching full uploads playlist for %s (playlist=%s)",
        channel_id, uploads_id,
    )

    all_entries: list[dict] = []
    page_token: str | None = None
    page_count = 0
    total_results = 0

    while True:
        page_data = _fetch_playlist_page(uploads_id, page_token=page_token)
        items = page_data.get("items", [])
        page_count += 1

        if page_count == 1:
            total_results = page_data.get("pageInfo", {}).get("totalResults", 0)
            logger.debug(
                "refresh_channel_cache: %d total results, ~%d pages",
                total_results, (total_results + 49) // 50,
            )

        for item in items:
            all_entries.append(_item_to_video_entry(item))

        page_token = page_data.get("nextPageToken")
        if not page_token:
            break

    # Count available (fetchable) videos — this is the UI denominator.
    # private_or_removed items are counted in total but not in available.
    available_count = sum(
        1 for e in all_entries
        if e["availability"] in ("public", "unlisted")
    )

    logger.info(
        "refresh_channel_cache: %d pages fetched, %d total / %d available",
        page_count, len(all_entries), available_count,
    )

    # Upsert the cache record
    existing = session.query(YouTubePlaylistCache).filter_by(channel_id=channel_id).first()
    if existing:
        existing.fetched_at = datetime.now(UTC)
        existing.video_count_total = len(all_entries)
        existing.video_count_available = available_count
        existing.videos_json = json.dumps(all_entries)
        existing.next_page_token = None  # fully fetched
        return existing
    else:
        cache = YouTubePlaylistCache(
            channel_id=channel_id,
            fetched_at=datetime.now(UTC),
            video_count_total=len(all_entries),
            video_count_available=available_count,
            videos_json=json.dumps(all_entries),
            next_page_token=None,
        )
        session.add(cache)
        session.flush()
        return cache


# ─────────────────────────────────────────────────────────────────────────────
# New-upload detection (cheap — 1–2 units regardless of catalog size)
# ─────────────────────────────────────────────────────────────────────────────

def check_for_new_uploads(channel_id: str, session: Session) -> dict:
    """Check for new uploads since the last cache refresh.

    Cost: 1 unit normally; 2 units if count_delta > BULK_UPLOAD_THRESHOLD (50).

    Returns:
        {
          "new_videos":  [list of new video entries in videos_json shape],
          "count_delta": int,   # positive = additions, negative = removals
          "cache_age_hours": float,
        }

    Detection mechanism (Phase 0 spike validated):
    1. Fetch page 1 of the uploads playlist (newest-first order is guaranteed).
    2. Compare video_ids[0..49] against cached video_ids[0..49].
    3. New items = IDs in fresh page 1 not present in cached page 1.
    4. count_delta = fresh totalResults - cached total.
    5. If count_delta > BULK_UPLOAD_THRESHOLD, fetch page 2 to catch spills.
    """
    cache = session.query(YouTubePlaylistCache).filter_by(channel_id=channel_id).first()
    if not cache:
        # No cache at all — caller should call refresh_channel_cache first
        return {"new_videos": [], "count_delta": 0, "cache_age_hours": float("inf")}

    cached_entries: list[dict] = json.loads(cache.videos_json)
    cached_page1_ids = {e["video_id"] for e in cached_entries[:50]}

    now = datetime.now(UTC)
    fetched_at = cache.fetched_at
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=UTC)
    cache_age_hours = (now - fetched_at).total_seconds() / 3600.0

    uploads_id = _uploads_playlist_id_from_channel_id(channel_id)

    # Fetch fresh page 1
    page1_data = _fetch_playlist_page(uploads_id, page_token=None)
    fresh_total = page1_data.get("pageInfo", {}).get("totalResults", 0)
    fresh_items = page1_data.get("items", [])
    fresh_page1_ids = {
        item["snippet"]["resourceId"]["videoId"]
        for item in fresh_items
    }

    count_delta = fresh_total - cache.video_count_total

    # New video IDs = in fresh page 1 but not in cached page 1
    new_ids = fresh_page1_ids - cached_page1_ids

    # Bulk-upload spill: if more than 50 new videos, fetch page 2 too
    if count_delta > BULK_UPLOAD_THRESHOLD:
        logger.info(
            "check_for_new_uploads: count_delta=%d > threshold — fetching page 2",
            count_delta,
        )
        page2_token = page1_data.get("nextPageToken")
        if page2_token:
            page2_data = _fetch_playlist_page(uploads_id, page_token=page2_token)
            for item in page2_data.get("items", []):
                vid = item["snippet"]["resourceId"]["videoId"]
                if vid not in cached_page1_ids:
                    new_ids.add(vid)

    # Build full entry dicts for new videos (from the fresh page 1 items)
    id_to_item = {
        item["snippet"]["resourceId"]["videoId"]: item
        for item in fresh_items
    }
    new_videos = [
        _item_to_video_entry(id_to_item[vid])
        for vid in new_ids
        if vid in id_to_item
    ]
    # Sort new videos newest-first (by published_at)
    new_videos.sort(key=lambda e: e["published_at"], reverse=True)

    logger.debug(
        "check_for_new_uploads: %d new video(s), count_delta=%d, cache_age=%.1fh",
        len(new_videos), count_delta, cache_age_hours,
    )

    return {
        "new_videos": new_videos,
        "count_delta": count_delta,
        "cache_age_hours": cache_age_hours,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TTL check
# ─────────────────────────────────────────────────────────────────────────────

def cache_is_stale(channel_id: str, session: Session) -> bool:
    """Return True if the cache is absent or older than CACHE_TTL_HOURS."""
    cache = session.query(YouTubePlaylistCache).filter_by(channel_id=channel_id).first()
    if not cache:
        return True
    fetched_at = cache.fetched_at
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=UTC)
    age = datetime.now(UTC) - fetched_at
    return age > timedelta(hours=CACHE_TTL_HOURS)


# ─────────────────────────────────────────────────────────────────────────────
# Gap computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_gap(channel_id: str, session: Session) -> dict:
    """Compute the gap between the channel catalog and the local library.

    The gap is the set of available (public/unlisted) videos in the channel
    that have not yet been transcribed into the library.

    Returns:
        {
          "gap_count":       int,   # available videos not in library
          "library_count":   int,   # videos from this channel in library
          "available_count": int,   # video_count_available from cache
          "total_count":     int,   # video_count_total (including private)
          "gap_videos":      list,  # [{video_id, title, published_at, availability}, ...]
                                    # newest-first, available only
        }

    The denominator in the UI fraction ("89/312") is available_count, not
    total_count. Private/removed placeholders are shown as a footnote if present
    ("plus N private/unavailable"), not as part of the target number.
    """
    from audiobench.storage.models import AudioFileRecord

    cache = session.query(YouTubePlaylistCache).filter_by(channel_id=channel_id).first()
    if not cache:
        return {
            "gap_count": 0,
            "library_count": 0,
            "available_count": 0,
            "total_count": 0,
            "gap_videos": [],
        }

    # Video IDs already in the library for this channel
    in_library = {
        row.youtube_video_id
        for row in session.query(AudioFileRecord)
        .filter(
            AudioFileRecord.youtube_channel_id == channel_id,
            AudioFileRecord.youtube_video_id.isnot(None),
        )
        .all()
    }

    all_entries: list[dict] = json.loads(cache.videos_json)

    # Gap = available videos not yet in library
    gap_videos = [
        e for e in all_entries
        if e["availability"] in ("public", "unlisted")
        and e["video_id"] not in in_library
    ]

    return {
        "gap_count": len(gap_videos),
        "library_count": len(in_library),
        "available_count": cache.video_count_available,
        "total_count": cache.video_count_total,
        "gap_videos": gap_videos,  # already newest-first (playlist order)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point helper: what to do on workspace entry
# ─────────────────────────────────────────────────────────────────────────────

def _api_unavailable_result(cache_age_hours: float = 0.0) -> dict:
    """Safe fallback result for when the API is unreachable."""
    return {
        "refreshed": False,
        "new_videos": [],
        "count_delta": 0,
        "cache_age_hours": cache_age_hours,
        "api_error": True,
    }


def on_workspace_entry(channel_id: str, session: Session) -> dict:
    """Run the correct cache operation on workspace entry and return a diff summary.

    Decision tree:
      - Cache absent or stale (>24h): full refresh, no new-upload check needed
      - Cache fresh (<24h): cheap new-upload check only (1–2 units)

    Degrades gracefully if the YouTube API is unavailable (key missing, network
    error, quota exhausted). In degraded mode, the workspace still opens and shows
    the cached catalog; it just cannot report new uploads or refresh stale data.

    Returns:
        {
          "refreshed":      bool,
          "new_videos":     list,
          "count_delta":    int,
          "cache_age_hours": float,
          "api_error":      bool,  # True only on degraded mode
        }
    """
    stale = cache_is_stale(channel_id, session)

    if stale:
        logger.info("on_workspace_entry: cache stale — running full refresh for %s", channel_id)
        try:
            refresh_channel_cache(channel_id, session)
        except RuntimeError as e:
            # Expected case: API key not configured, or quota exhausted.
            # The workspace shows a user-facing ⚠ notice — no need to also
            # print a WARNING line to the terminal before the workspace renders.
            logger.debug("on_workspace_entry: API unavailable (stale cache): %s", e)
            return _api_unavailable_result()
        except Exception as e:
            logger.warning("on_workspace_entry: refresh failed: %s", e)
            return _api_unavailable_result()
        return {
            "refreshed": True,
            "new_videos": [],
            "count_delta": 0,
            "cache_age_hours": 0.0,
            "api_error": False,
        }

    try:
        result = check_for_new_uploads(channel_id, session)
    except RuntimeError as e:
        # Expected: API key not configured. UI ⚠ notice handles the user-facing message.
        logger.debug("on_workspace_entry: API unavailable (fresh cache): %s", e)
        return _api_unavailable_result(cache_age_hours=0.0)
    except Exception as e:
        logger.warning("on_workspace_entry: new-upload check failed: %s", e)
        return _api_unavailable_result()

    return {
        "refreshed": False,
        "new_videos": result["new_videos"],
        "count_delta": result["count_delta"],
        "cache_age_hours": result["cache_age_hours"],
        "api_error": False,
    }
