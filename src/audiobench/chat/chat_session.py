"""Chat session — orchestrates multi-turn AI conversations.

Manages conversation state (message history), context injection
from transcripts, and persistence via ChatRepository.

Usage:
    from audiobench.chat.chat_session import ChatSession
    from audiobench.chat.providers.ollama_provider import OllamaClient
    from audiobench.chat.chat_store import ChatRepository

    client = OllamaClient()
    repo = ChatRepository()
    session = ChatSession(client, repo, model="qwen4-next:110b-cloud")
    session.load_transcripts([transcript_dict])

    for chunk in session.send("What was discussed?"):
        print(chunk["content"], end="", flush=True)
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from audiobench.chat.context_builder import (
    CHAT_SYSTEM,
    CHAT_SYSTEM_NO_CONTEXT,
    TITLE_PROMPT,
)
from audiobench.core.logger_factory import get_logger
from audiobench.playback.controls import fmt_timestamp

logger = get_logger("ai.chat")

# Max words before truncating transcript context
MAX_CONTEXT_WORDS = 6000


def create_chat_client(model: str | None = None, settings: Any = None) -> Any:
    """Instantiate the appropriate LLM client for the given model name."""
    if settings is None:
        from audiobench.core.settings import get_settings

        settings = get_settings()

    model_clean = (model or "").lower().strip()

    # Check for Gemini provider
    if model_clean.startswith("gemini"):
        from audiobench.chat.providers.gemini_provider import GeminiClient

        gemini_client = GeminiClient(model=model or settings.gemini_model)
        if gemini_client.is_available():
            return gemini_client
        logger.warning(
            "Gemini model requested (%s) but GeminiClient is not available (check GEMINI_API_KEY).",
            model,
        )

    # Default to Ollama provider
    from audiobench.chat.providers.ollama_provider import OllamaClient

    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=model or settings.ollama_model,
    )


def _format_transcript_context_text(t: dict) -> tuple[str, bool]:
    """Format transcript dictionary into context text for the AI system prompt.

    Prefers segment-level timestamps [MM:SS] (or [HH:MM:SS]) with optional speaker tags
    when meaningful timestamps exist (> 0.0s). Falls back to full_text if
    segments are missing or all timestamps are 0.0s.

    Returns:
        (text, has_timestamps) tuple.
    """
    segments = t.get("segments") or []
    has_timestamps = any(
        (s.get("start") or 0.0) > 0.0 or (s.get("end") or 0.0) > 0.0
        for s in segments
    )
    if segments and has_timestamps:
        lines = []
        for seg in segments:
            seg_text = (seg.get("text") or "").strip()
            if not seg_text:
                continue
            ts = fmt_timestamp(seg.get("start", 0.0))
            speaker = seg.get("speaker")
            spk_prefix = f"[{speaker}]: " if speaker else ""
            lines.append(f"[{ts}] {spk_prefix}{seg_text}")
        if lines:
            return "\n".join(lines), True

    return t.get("full_text", "") or "", False


class ChatSession:
    """Manages a multi-turn AI chat conversation with transcript context.

    Handles:
    - Building the system prompt with injected transcript text
    - Maintaining message history for multi-turn conversations
    - Streaming responses via OllamaClient.chat_stream()
    - Persisting conversations via ChatRepository
    - Auto-generating titles after first exchange
    """

    def __init__(
        self,
        client=None,
        chat_repo=None,
        model: str = "",
        temperature: float = 0.3,
        conversation_id: int | None = None,
        show_thinking: bool = True,
        session_type: str = "chat",
        title: str = "New Chat",
    ) -> None:
        if client is None:
            client = create_chat_client(model)
        else:
            # Reconcile if client provider does not match model
            is_gem = (model or "").lower().startswith("gemini")
            cur_is_gem = type(client).__name__ == "GeminiClient"
            if is_gem and not cur_is_gem:
                from audiobench.chat.providers.gemini_provider import GeminiClient

                g_client = GeminiClient(model=model)
                if g_client.is_available():
                    client = g_client
            elif not is_gem and cur_is_gem:
                from audiobench.chat.providers.ollama_provider import OllamaClient
                from audiobench.core.settings import get_settings

                client = OllamaClient(base_url=get_settings().ollama_base_url, model=model)

        if chat_repo is None:
            from audiobench.chat.chat_store import ChatRepository

            chat_repo = ChatRepository()
        self._client = client
        self._chat_repo = chat_repo
        self._model = model
        self._temperature = temperature
        self._retry_requested: bool = False
        self._compare_model: str | None = None
        self._show_thinking = show_thinking
        self._conversation_id = conversation_id
        self._session_type = session_type
        self._initial_title = title

        # Transcript context
        self._transcripts: list[dict] = []
        self._transcript_ids: list[int] = []

        # Message history (in-memory, synced to DB)
        self._messages: list[dict] = []
        self._system_prompt: str = CHAT_SYSTEM_NO_CONTEXT
        self._title_generated = False
        self._injected_fragments: list[dict] = []

    # ── Properties ──────────────────────────────────────────

    @property
    def conversation_id(self) -> int | None:
        return self._conversation_id

    @property
    def model(self) -> str:
        return self._model

    @property
    def turn_count(self) -> int:
        """Number of user+assistant exchanges (excludes system)."""
        return sum(1 for m in self._messages if m["role"] == "user")

    @property
    def messages(self) -> list[dict]:
        """Get the message history (excludes system prompt)."""
        return list(self._messages)

    @property
    def show_thinking(self) -> bool:
        return self._show_thinking

    @show_thinking.setter
    def show_thinking(self, value: bool) -> None:
        self._show_thinking = value

    # ── Context Management ──────────────────────────────────

    def load_transcripts(self, transcripts: list[dict]) -> None:
        """Load transcript dicts into the system prompt as context.

        Each dict should have: id, file_name, full_text, word_count, created_at.
        Transcripts are injected into the system prompt.
        """
        for t in transcripts:
            if t["id"] not in self._transcript_ids:
                self._transcripts.append(t)
                self._transcript_ids.append(t["id"])

        self._rebuild_system_prompt()

        # Update DB if conversation exists
        if self._conversation_id:
            self._chat_repo.update_transcript_ids(self._conversation_id, self._transcript_ids)

    def _rebuild_system_prompt(self) -> None:
        """Rebuild the system prompt with current transcript context."""
        if not self._transcripts and not self._injected_fragments:
            self._system_prompt = CHAT_SYSTEM_NO_CONTEXT
            return

        # Build context block
        context_parts = []
        total_words = 0

        for t in self._transcripts:
            text, has_ts = _format_transcript_context_text(t)
            words = text.split()
            word_count = len(words)

            # Truncate if we'd exceed the limit
            if total_words + word_count > MAX_CONTEXT_WORDS:
                remaining = MAX_CONTEXT_WORDS - total_words
                if remaining > 100:
                    if has_ts:
                        kept_lines = []
                        w_acc = 0
                        for line in text.split("\n"):
                            l_words = len(line.split())
                            if w_acc + l_words > remaining:
                                break
                            kept_lines.append(line)
                            w_acc += l_words
                        if kept_lines:
                            text = "\n".join(kept_lines) + "\n[... truncated]"
                            word_count = w_acc
                        else:
                            text = " ".join(words[:remaining]) + "\n[... truncated]"
                            word_count = remaining
                    else:
                        text = " ".join(words[:remaining]) + "\n[... truncated]"
                        word_count = remaining
                else:
                    context_parts.append(
                        f"--- Transcript #{t['id']}: {t.get('file_name', 'unknown')} "
                        f"({t.get('word_count') or word_count} words) [SKIPPED — context limit reached] ---"
                    )
                    continue

            display_count = t.get("word_count") or word_count
            header = (
                f"--- Transcript #{t['id']}: {t.get('file_name', 'unknown')} "
                f"({display_count} words) ---"
            )
            context_parts.append(f"{header}\n{text}")
            total_words += word_count

        if self._injected_fragments:
            context_parts.append("\n--- INJECTED SEARCH FRAGMENTS ---")
            for frag in self._injected_fragments:
                context_parts.append(
                    f"[{frag['source_file']} @ {frag['timestamp']} (query: {frag.get('query', '')})]\n{frag['text']}"
                )
            context_parts.append("----------------------------------\n")

        transcript_context = "\n\n".join(context_parts)
        self._system_prompt = CHAT_SYSTEM.format(transcript_context=transcript_context)

    @property
    def injected_fragments(self) -> list[dict]:
        """List of currently injected search fragments."""
        return list(self._injected_fragments)

    def inject_search_fragments(self, fragments: list, query: str) -> None:
        """Inject search fragments into the chat context."""
        def _ts(s: float) -> str:
            m, sec = divmod(int(s), 60)
            h, m = divmod(m, 60)
            return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

        for f in fragments:
            ts = f"{_ts(f.start_time)} → {_ts(f.end_time)}"
            self._injected_fragments.append({
                "source_file": f.source_file or f"Transcript #{f.transcription_id}",
                "transcription_id": f.transcription_id,
                "timestamp": ts,
                "start_time": f.start_time,
                "end_time": f.end_time,
                "score": getattr(f, "rrf_score", 0.0),
                "text": f.text.strip(),
                "query": query,
            })

        self._rebuild_system_prompt()

    def clear_injected_fragments(self) -> int:
        """Clear all injected search fragments from context.

        Returns:
            Number of fragments removed.
        """
        count = len(self._injected_fragments)
        self._injected_fragments.clear()
        self._rebuild_system_prompt()
        return count

    def remove_transcripts(self, transcript_ids: list[int]) -> list[int]:
        """Remove multiple transcripts from context by ID.

        Returns:
            List of IDs that were successfully removed.
        """
        removed = []
        for tid in transcript_ids:
            if tid in self._transcript_ids:
                self._transcript_ids.remove(tid)
                removed.append(tid)

        if removed:
            self._transcripts = [t for t in self._transcripts if t["id"] not in removed]
            self._rebuild_system_prompt()
            if self._conversation_id:
                self._chat_repo.update_transcript_ids(self._conversation_id, self._transcript_ids)
            logger.info("Removed %d transcripts from context: %s", len(removed), removed)
        return removed

    def remove_transcript(self, transcript_id: int) -> bool:
        """Remove a transcript from context by ID.

        Returns:
            True if the transcript was found and removed.
        """
        return bool(self.remove_transcripts([transcript_id]))

    def get_context_summary(self) -> list[str]:
        """Get a summary of loaded transcripts or search fragments for display.

        Returns:
            List of formatted strings like "#3 meeting.m4a (2,341 words)"
        """
        if not self._transcripts:
            if self._injected_fragments:
                n = len(self._injected_fragments)
                q = self._injected_fragments[0].get("query")
                q_label = f" for '{q}'" if q else ""
                sources = {f.get("source_file") for f in self._injected_fragments if f.get("source_file")}
                src_count = len(sources)
                src_label = f" across {src_count} source{'s' if src_count != 1 else ''}" if src_count else ""
                return [f"{n} search fragment{'s' if n != 1 else ''}{q_label}{src_label}"]
            return ["No transcripts loaded (use /load <ID>)"]

        lines = []
        for t in self._transcripts:
            name = t.get("file_name", "unknown")
            words = t.get("word_count", 0)
            lines.append(f"#{t['id']} {name} ({words:,} words)")
        if self._injected_fragments:
            lines.append(f"+ {len(self._injected_fragments)} search fragments injected")
        return lines

    def get_context_details(self) -> dict:
        """Get structured breakdown of context for detailed UI rendering."""
        txs_info = []
        total_words = 0
        running_words = 0
        for t in self._transcripts:
            text = t.get("full_text", "")
            words = t.get("word_count", len(text.split())) if text else t.get("word_count", 0)
            total_words += words
            remaining_budget = MAX_CONTEXT_WORDS - running_words
            skipped = remaining_budget < 100
            running_words += words
            segments = t.get("segments") or []
            has_timestamps = any(
                (s.get("start") or 0.0) > 0.0 or (s.get("end") or 0.0) > 0.0
                for s in segments
            )
            txs_info.append({
                "id": t["id"],
                "file_name": t.get("file_name", "unknown"),
                "word_count": words,
                "skipped": skipped,
                "has_timestamps": has_timestamps,
                "segment_count": len(segments),
            })

        return {
            "transcripts": txs_info,
            "total_words": total_words,
            "max_words": MAX_CONTEXT_WORDS,
            "fragments": list(self._injected_fragments),
        }

    # ── Conversation Management ─────────────────────────────

    def ensure_conversation(self) -> int:
        """Ensure a conversation exists in DB, creating if needed.

        Returns:
            The conversation ID.
        """
        if self._conversation_id is None:
            self._conversation_id = self._chat_repo.create_conversation(
                model=self._model,
                transcript_ids=self._transcript_ids,
                title=self._initial_title,
                session_type=self._session_type,
            )
            # Save the system prompt as the first message
            self._chat_repo.add_message(
                self._conversation_id,
                "system",
                self._system_prompt,
            )
            logger.info("Created conversation #%d", self._conversation_id)
        return self._conversation_id

    def restore_from_db(self, tx_repo=None) -> bool:
        """Restore conversation state from database.

        Args:
            tx_repo: Optional TranscriptionRepository. When provided, full
                transcript text is re-fetched from the DB so the LLM has its
                context immediately available (same as the original /load call).
                When omitted, only IDs and messages are restored.

        Returns:
            True if conversation was found and restored.
        """
        if self._conversation_id is None:
            return False

        conv = self._chat_repo.get_conversation(self._conversation_id)
        if conv is None:
            return False

        restored_model = conv.get("model", self._model)
        if restored_model:
            try:
                self.switch_model(restored_model)
            except Exception as exc:
                logger.warning("Could not switch client to restored model %s: %s", restored_model, exc)
                self._model = restored_model
        self._transcript_ids = conv.get("transcript_ids", [])

        # Restore messages
        self._messages = []
        for msg in conv.get("messages", []):
            if msg["role"] == "system":
                self._system_prompt = msg["content"]
            else:
                self._messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "thinking": msg.get("thinking"),
                        "model_name": msg.get("model_name"),
                    }
                )

        # If there are user messages, title was likely already generated
        if any(m["role"] == "user" for m in self._messages):
            self._title_generated = True

        # Re-hydrate transcript full text so the system prompt is rebuilt
        # with the same context the user originally loaded.
        if tx_repo and (self._transcript_ids or conv.get("audio_file_id")):
            self._transcripts = []
            loaded = []
            for tid in self._transcript_ids:
                record = tx_repo.get_by_id(tid, session=None)
                if record:
                    self._transcripts.append(record)
                    loaded.append(tid)
                else:
                    logger.warning("Transcript #%d not found during restore — skipping", tid)

            # Invariant fallback: if transcripts could not be loaded by ID (e.g. upgraded or purged),
            # but conversation is bound to an audio_file_id, recover latest transcript for the file!
            if not self._transcripts and conv.get("audio_file_id"):
                latest_rec = tx_repo.get_latest_transcript_for_file(conv["audio_file_id"])
                if latest_rec:
                    self._transcripts.append(latest_rec)
                    loaded.append(latest_rec["id"])
                    logger.info(
                        "Adapted chat context to latest transcript #%d for audio_file #%d",
                        latest_rec["id"],
                        conv["audio_file_id"],
                    )

            if loaded != self._transcript_ids:
                self._transcript_ids = loaded
                if self._chat_repo and self._conversation_id:
                    self._chat_repo.update_transcript_ids(self._conversation_id, self._transcript_ids)

            # Rebuild system prompt with restored transcript text
            self._rebuild_system_prompt()
            logger.info(
                "Restored conversation #%d: %d messages, %d transcripts re-hydrated",
                self._conversation_id,
                len(self._messages),
                len(loaded),
            )
        else:
            logger.info(
                "Restored conversation #%d (%d messages)",
                self._conversation_id,
                len(self._messages),
            )
        return True

    def send(
        self,
        user_input: str,
        modality: str = "text",
        duration_seconds: float | None = None,
        audio_path: str | None = None,
        segment_id: int | None = None,
    ) -> Iterator[dict]:
        """Send a user message and stream the AI response.

        Only streams chunks — call :meth:`finalize_response` afterwards
        to persist the response and trigger background title generation
        so that the REPL prompt appears immediately.

        Args:
            user_input: The user's message text.
            modality: "text", "voice", or "voice_live".
            duration_seconds: Duration of speech if spoken.
            audio_path: Path to cached audio file if recorded.
            segment_id: Linked SegmentRecord id.

        Yields:
            Dicts with "content" and/or "thinking" keys per streamed chunk.
        """
        conv_id = self.ensure_conversation()

        # Add user message
        self._messages.append({"role": "user", "content": user_input})
        self._chat_repo.add_message(
            conv_id,
            "user",
            user_input,
            modality=modality,
            duration_seconds=duration_seconds,
            audio_path=audio_path,
            segment_id=segment_id,
        )

        # Build full message list for API
        api_messages = self._build_api_messages()

        # Accumulators (read by finalize_response)
        self._pending_content: list[str] = []
        self._pending_thinking: list[str] = []
        self._pending_user_input = user_input

        try:
            for chunk in self._client.chat_stream(
                messages=api_messages,
                model=self._model,
                temperature=self._temperature,
            ):
                content = chunk.get("content", "")
                thinking = chunk.get("thinking", "")

                if content:
                    self._pending_content.append(content)
                if thinking:
                    self._pending_thinking.append(thinking)

                yield chunk

                if chunk.get("done", False):
                    break

        except Exception:
            # Remove the user message if streaming failed
            self._messages.pop()
            self._pending_content.clear()
            self._pending_thinking.clear()
            raise

    def finalize_response(self) -> None:
        """Persist the streamed response and kick off background title gen.

        Must be called after :meth:`send` completes.  Saves the assistant
        message to the DB and starts title generation in a thread so the
        REPL prompt appears immediately.
        """

        response_text = "".join(self._pending_content)
        thinking_text = "".join(self._pending_thinking) or None
        user_input = self._pending_user_input

        # If the model returned thinking-only (no content), use thinking as the response.
        # This happens with some models (e.g. deepseek) that put everything in thinking.
        if not response_text.strip() and thinking_text:
            response_text = thinking_text

        # Save assistant message
        self._messages.append({"role": "assistant", "content": response_text})
        self._chat_repo.add_message(
            self._conversation_id,
            "assistant",
            response_text,
            thinking=thinking_text,
        )

        # Auto-generate title in background after first exchange
        if self.turn_count >= 1:
            self.generate_title_async(user_input, response_text)

        # Cleanup
        self._pending_content.clear()
        self._pending_thinking.clear()
        self._pending_user_input = ""

    def seed_initial_turn(self, query: str, answer: str) -> None:
        """Seed the session with an initial user query and assistant answer (e.g. from search synthesis)."""
        conv_id = self.ensure_conversation()
        self._messages.append({"role": "user", "content": query})
        self._chat_repo.add_message(conv_id, "user", query)
        self._messages.append({"role": "assistant", "content": answer})
        self._chat_repo.add_message(conv_id, "assistant", answer)
        self._title_generated = True

    def generate_title_async(self, first_message: str, first_response: str) -> None:
        """Trigger background generation of the conversation title."""
        if not self._title_generated:
            self._title_generated = True
            thread = threading.Thread(
                target=self._generate_title,
                args=(first_message, first_response),
                daemon=True,
            )
            thread.start()

    def _generate_title(self, first_message: str, first_response: str) -> None:
        """Generate a short title for the conversation using AI."""
        self._title_generated = True
        try:
            prompt = TITLE_PROMPT.format(
                first_message=first_message[:200],
                first_response=first_response[:200],
            )
            # GeminiClient exposes .generate() for single-shot non-streaming prompts.
            # OllamaClient exposes .chat() which returns {"content": "..."}.
            # Use whichever the current client supports.
            if hasattr(self._client, "generate"):
                title = self._client.generate(
                    prompt=prompt,
                    model=self._model,
                    temperature=0.3,
                ).strip()
            else:
                result = self._client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._model,
                    temperature=0.3,
                    think=False,
                )
                title = result.get("content", "").strip()
            if title and len(title) < 100:
                self._chat_repo.update_title(self._conversation_id, title)
                logger.info("Generated title: %s", title)
        except Exception as e:
            logger.warning("Failed to generate title: %s", e)

    def clear_history(self) -> None:
        """Clear conversation history AND transcript context.

        Resets the session to a clean state — no messages, no transcripts.
        Creates a new conversation in the DB.
        """
        self._messages.clear()
        self._transcripts.clear()
        self._transcript_ids.clear()
        if hasattr(self, "_injected_fragments"):
            self._injected_fragments.clear()
        self._rebuild_system_prompt()

        if self._conversation_id:
            old_id = self._conversation_id
            self._conversation_id = None
            self._title_generated = False
            self.ensure_conversation()
            logger.info(
                "Cleared history + context (old=#%d, new=#%d)",
                old_id,
                self._conversation_id,
            )

    def switch_model(self, model: str) -> None:
        """Switch the model mid-conversation, swapping the underlying API client if needed."""
        from audiobench.core.settings import get_settings

        settings = get_settings()
        if model.lower() in ("ollama", "default"):
            model = settings.ollama_model
        elif model.lower() == "gemini":
            model = settings.gemini_model or "gemini-2.5-flash"

        self._model = model

        _is_gemini = model.lower().startswith("gemini")
        _current_is_gemini = type(self._client).__name__ == "GeminiClient"

        if _is_gemini:
            from audiobench.chat.providers.gemini_provider import GeminiClient

            if not _current_is_gemini or getattr(self._client, "_model", None) != model:
                new_client = GeminiClient(model=model)
                if not new_client.is_available():
                    raise RuntimeError(
                        "Gemini not available — check GEMINI_API_KEY in settings."
                    )
                self._client = new_client
        else:
            from audiobench.chat.providers.ollama_provider import OllamaClient

            if _current_is_gemini:
                self._client = OllamaClient(
                    base_url=settings.ollama_base_url,
                    model=model,
                )
            else:
                if hasattr(self._client, "_model"):
                    self._client._model = model
                if hasattr(self._client, "model"):
                    self._client.model = model

        if self._conversation_id and self._chat_repo:
            engine = "gemini" if _is_gemini else "ollama"
            try:
                self._chat_repo.update_model(self._conversation_id, model, engine=engine)
            except Exception:
                pass

        logger.info("Switched model to %s (client: %s)", model, type(self._client).__name__)


    def _build_api_messages(self) -> list[dict]:
        """Build the message list for the API, handling comparison pairs.

        When consecutive assistant messages exist (from a comparison), we
        only include the primary model's response to avoid confusing the LLM.
        """
        api_msgs = [{"role": "system", "content": self._system_prompt}]
        messages = list(self._messages)
        i = 0
        while i < len(messages):
            msg = messages[i]
            if (
                msg["role"] == "assistant"
                and i + 1 < len(messages)
                and messages[i + 1]["role"] == "assistant"
            ):
                # Comparison pair — use the first response (primary model)
                api_msgs.append({"role": "assistant", "content": msg["content"]})
                i += 2  # skip both
            else:
                api_msgs.append({"role": msg["role"], "content": msg["content"]})
                i += 1
        return api_msgs
