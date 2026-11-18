"""MemoirWriter: Generate and persist structured session memoirs.

A memoir captures:
  - narrative: flowing prose summary of the conversation
  - key_insights: JSON list of distilled insights
  - open_threads: JSON list of unresolved questions
  
The memoir is stored as a ConversationSummary row and registered as a
'session_memoir' expression in the knowledge graph.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from audiobench.core.db_session import get_session as db_session

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class Memoir:
    """Structured output of a memoir generation."""
    narrative: str
    key_insights: str   # JSON-encoded list[str]
    open_threads: str   # JSON-encoded list[str]
    refined_title: str | None = None


class CompressionLevel(Enum):
    FULL = "full"        # N-1 session: full narrative + all fields
    DIGEST = "digest"    # N-2 to N-3: narrative digest + insights + threads
    KEY_ONLY = "key_only"  # N-4+: only key insights + open threads


# ── Compression helper ───────────────────────────────────────────────────────


def compress_memoir(memoir: Memoir, level: CompressionLevel) -> str:
    """Return a compressed text representation of a memoir.

    Open threads are ALWAYS preserved at every compression level.
    """
    try:
        insights = json.loads(memoir.key_insights)
    except (json.JSONDecodeError, TypeError):
        insights = []
    try:
        threads = json.loads(memoir.open_threads)
    except (json.JSONDecodeError, TypeError):
        threads = []

    parts: list[str] = []

    if level == CompressionLevel.FULL:
        parts.append(f"Narrative: {memoir.narrative}")
        if insights:
            parts.append("Key Insights:\n" + "\n".join(f"- {i}" for i in insights))
        if threads:
            parts.append("Open Threads:\n" + "\n".join(f"- {t}" for t in threads))

    elif level == CompressionLevel.DIGEST:
        # Truncate narrative to first 300 chars
        digest = memoir.narrative[:300].rstrip()
        if len(memoir.narrative) > 300:
            digest += "..."
        parts.append(f"Summary: {digest}")
        if insights:
            parts.append("Key Insights:\n" + "\n".join(f"- {i}" for i in insights))
        if threads:
            parts.append("Open Threads:\n" + "\n".join(f"- {t}" for t in threads))

    else:  # KEY_ONLY — always has threads
        if insights:
            parts.append("Key Insights:\n" + "\n".join(f"- {i}" for i in insights))
        if threads:
            parts.append("Open Threads:\n" + "\n".join(f"- {t}" for t in threads))

    return "\n\n".join(parts)


# ── Prompt ───────────────────────────────────────────────────────────────────


_MEMOIR_SYSTEM = """You are a session archivist. Your job is to distill a conversation into a structured memoir.

Output ONLY a JSON object (no markdown fences) with these exact keys:
{
  "narrative": "<flowing prose summary of the conversation, minimum 80 words>",
  "key_insights": ["<insight 1>", "<insight 2>", ...],
  "open_threads": ["<unresolved question or thread 1>", ...],
  "refined_title": "<short title for this session, max 60 chars>"
}

Rules:
- narrative must be a coherent paragraph, not bullet points, at least 80 words
- key_insights must be a JSON array of strings (at least 1)
- open_threads must be a JSON array of strings (can be empty [])
- Do not add any keys beyond the four listed above
"""

_MEMOIR_USER_TMPL = """Conversation title: {title}

Messages:
{messages}

Generate the structured memoir now."""


# ── Writer ───────────────────────────────────────────────────────────────────


class MemoirWriter:
    """Generate, store, and register structured session memoirs."""

    def generate(
        self,
        conversation: object,
        session: object,
    ) -> Memoir:
        """Generate a memoir for a conversation/session pair.

        Args:
            conversation: ChatConversation ORM object
            session: StudySession ORM object (can be None for non-study chats)

        Returns:
            Memoir dataclass with all fields populated.
        """
        from audiobench.storage.expression_repository import ExpressionRepository
        from audiobench.storage.models import ChatMessage, ConversationSummary

        # ── 1. Fetch messages ──────────────────────────────────────────────
        with db_session() as db:
            messages = (
                db.query(ChatMessage)
                .filter_by(conversation_id=conversation.id)
                .order_by(ChatMessage.id)
                .all()
            )
            # Expunge to detach from session
            for m in messages:
                db.expunge(m)

        # ── 2. Build prompt ────────────────────────────────────────────────
        msg_text = "\n".join(
            f"[{m.role.upper()}] {m.content[:500]}" for m in messages
        )
        title = getattr(conversation, "title", "Untitled Conversation")
        user_prompt = _MEMOIR_USER_TMPL.format(title=title, messages=msg_text)

        # ── 3. Call LLM ────────────────────────────────────────────────────
        memoir = self._call_llm(user_prompt, title)

        # ── 4. Store in DB ─────────────────────────────────────────────────
        from audiobench.storage.repository import TranscriptionRepository
        tx_repo = TranscriptionRepository()
        try:
            t_ids = json.loads(getattr(conversation, "transcript_ids", "[]") or "[]")
        except Exception:
            t_ids = []
        privacy_tier = tx_repo.get_max_privacy_tier_for_transcriptions(t_ids)

        repo = ExpressionRepository()
        expr = repo.register(
            content=memoir.narrative,
            source_type="session_memoir",
            session_id=conversation.id,
            session_type=getattr(conversation, "session_type", "memoir"),
            privacy_tier=privacy_tier,
        )

        with db_session() as db:
            # Upsert ConversationSummary
            existing = db.query(ConversationSummary).filter_by(
                conversation_id=conversation.id
            ).first()
            if existing:
                existing.narrative = memoir.narrative
                existing.key_insights = memoir.key_insights
                existing.open_threads = memoir.open_threads
                existing.refined_title = memoir.refined_title
                existing.expression_id = expr.id
                existing.generated_by = "memoir_writer"
            else:
                summary = ConversationSummary(
                    conversation_id=conversation.id,
                    narrative=memoir.narrative,
                    key_insights=memoir.key_insights,
                    open_threads=memoir.open_threads,
                    refined_title=memoir.refined_title,
                    expression_id=expr.id,
                    generated_by="memoir_writer",
                )
                db.add(summary)
            db.commit()

        # ── 5. Link study session if provided ─────────────────────────────
        if session is not None:
            try:
                with db_session() as db:
                    session_row = db.get(type(session), session.id)
                    if session_row is not None:
                        session_row.memoir_id = expr.id
                        db.commit()
            except Exception:
                logger.warning("Could not link memoir to study session", exc_info=True)

        return memoir

    # ── LLM helpers ──────────────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str, title: str) -> Memoir:
        """Try to call the LLM; fall back to a minimal default on failure."""
        try:
            from audiobench.chat.providers.provider_outcome import OutcomeType
            from audiobench.chat.providers.router import router_for_settings

            router = router_for_settings()
            outcome = router.generate(
                messages=[{"role": "user", "content": f"{_MEMOIR_SYSTEM}\n\n{user_prompt}"}],
                temperature=0.3,
            )
            if outcome.type == OutcomeType.OK and outcome.content:
                return self._parse_llm_response(outcome.content, title)
        except Exception:
            logger.warning("LLM call failed during memoir generation", exc_info=True)

        return self._fallback_memoir(title)

    def _parse_llm_response(self, text: str, title: str) -> Memoir:
        """Parse LLM JSON response into a Memoir. Falls back on any parse error."""
        # Strip markdown fences if present
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(raw)
            narrative = str(data.get("narrative", "")).strip()
            if len(narrative.split()) < 20:
                raise ValueError("Narrative too short")
            key_insights = data.get("key_insights", [])
            open_threads = data.get("open_threads", [])
            refined_title = data.get("refined_title") or title

            if not isinstance(key_insights, list):
                key_insights = [str(key_insights)]
            if not isinstance(open_threads, list):
                open_threads = [str(open_threads)]

            return Memoir(
                narrative=narrative,
                key_insights=json.dumps(key_insights),
                open_threads=json.dumps(open_threads),
                refined_title=str(refined_title)[:256],
            )
        except Exception:
            logger.warning("Failed to parse LLM memoir JSON, using fallback", exc_info=True)
            return self._fallback_memoir(title)

    def _fallback_memoir(self, title: str) -> Memoir:
        """Minimal memoir used when the LLM is unavailable or parse fails."""
        return Memoir(
            narrative=(
                f"Session '{title}' was completed. "
                "The memoir could not be generated due to an LLM failure. "
                "Key discussion points were recorded in the conversation history."
            ),
            key_insights=json.dumps(["Session completed"]),
            open_threads=json.dumps([]),
            refined_title=title,
        )
