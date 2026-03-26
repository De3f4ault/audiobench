"""Chat session — orchestrates multi-turn AI conversations.

Manages conversation state (message history), context injection
from transcripts, and persistence via ChatRepository.

Usage:
    from audiobench.chat.chat_session import ChatSession
    from audiobench.chat.providers.ollama_provider import OllamaClient
    from audiobench.chat.chat_store import ChatRepository

    client = OllamaClient()
    repo = ChatRepository()
    session = ChatSession(client, repo, model="qwen3-next:80b-cloud")
    session.load_transcripts([transcript_dict])

    for chunk in session.send("What was discussed?"):
        print(chunk["content"], end="", flush=True)
"""

from __future__ import annotations

from collections.abc import Iterator

from audiobench.chat.context_builder import (
    CHAT_SYSTEM,
    CHAT_SYSTEM_NO_CONTEXT,
    TITLE_PROMPT,
)
from audiobench.core.logger_factory import get_logger

logger = get_logger("ai.chat")

# Max words before truncating transcript context
MAX_CONTEXT_WORDS = 6000


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
        client,
        chat_repo,
        model: str,
        temperature: float = 0.3,
        conversation_id: int | None = None,
        show_thinking: bool = True,
    ) -> None:
        self._client = client
        self._chat_repo = chat_repo
        self._model = model
        self._temperature = temperature
        self._show_thinking = show_thinking
        self._conversation_id = conversation_id

        # Transcript context
        self._transcripts: list[dict] = []
        self._transcript_ids: list[int] = []

        # Message history (in-memory, synced to DB)
        self._messages: list[dict] = []
        self._system_prompt: str = CHAT_SYSTEM_NO_CONTEXT
        self._title_generated = False

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
        if not self._transcripts:
            self._system_prompt = CHAT_SYSTEM_NO_CONTEXT
            return

        # Build context block
        context_parts = []
        total_words = 0

        for t in self._transcripts:
            text = t.get("full_text", "")
            words = text.split()
            word_count = len(words)

            # Truncate if we'd exceed the limit
            if total_words + word_count > MAX_CONTEXT_WORDS:
                remaining = MAX_CONTEXT_WORDS - total_words
                if remaining > 100:
                    text = " ".join(words[:remaining]) + "\n[... truncated]"
                    word_count = remaining
                else:
                    context_parts.append(
                        f"--- Transcript #{t['id']}: {t.get('file_name', 'unknown')} "
                        f"({word_count} words) [SKIPPED — context limit reached] ---"
                    )
                    continue

            header = (
                f"--- Transcript #{t['id']}: {t.get('file_name', 'unknown')} "
                f"({word_count} words) ---"
            )
            context_parts.append(f"{header}\n{text}")
            total_words += word_count

        transcript_context = "\n\n".join(context_parts)
        self._system_prompt = CHAT_SYSTEM.format(transcript_context=transcript_context)

    def remove_transcript(self, transcript_id: int) -> bool:
        """Remove a transcript from context by ID.

        Returns:
            True if the transcript was found and removed.
        """
        if transcript_id not in self._transcript_ids:
            return False

        self._transcript_ids.remove(transcript_id)
        self._transcripts = [t for t in self._transcripts if t["id"] != transcript_id]
        self._rebuild_system_prompt()

        # Update DB if conversation exists
        if self._conversation_id:
            self._chat_repo.update_transcript_ids(self._conversation_id, self._transcript_ids)

        logger.info("Removed transcript #%d from context", transcript_id)
        return True

    def get_context_summary(self) -> list[str]:
        """Get a summary of loaded transcripts for display.

        Returns:
            List of formatted strings like "#3 meeting.m4a (2,341 words)"
        """
        if not self._transcripts:
            return ["No transcripts loaded (use /load <ID>)"]

        lines = []
        for t in self._transcripts:
            name = t.get("file_name", "unknown")
            words = t.get("word_count", 0)
            lines.append(f"#{t['id']} {name} ({words:,} words)")
        return lines

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
            )
            # Save the system prompt as the first message
            self._chat_repo.add_message(
                self._conversation_id,
                "system",
                self._system_prompt,
            )
            logger.info("Created conversation #%d", self._conversation_id)
        return self._conversation_id

    def restore_from_db(self) -> bool:
        """Restore conversation state from database.

        Returns:
            True if conversation was found and restored.
        """
        if self._conversation_id is None:
            return False

        conv = self._chat_repo.get_conversation(self._conversation_id)
        if conv is None:
            return False

        self._model = conv.get("model", self._model)
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

        logger.info(
            "Restored conversation #%d (%d messages)",
            self._conversation_id,
            len(self._messages),
        )
        return True

    def send(self, user_input: str) -> Iterator[dict]:
        """Send a user message and stream the AI response.

        Only streams chunks — call :meth:`finalize_response` afterwards
        to persist the response and trigger background title generation
        so that the REPL prompt appears immediately.

        Args:
            user_input: The user's message text.

        Yields:
            Dicts with "content" and/or "thinking" keys per streamed chunk.
        """
        conv_id = self.ensure_conversation()

        # Add user message
        self._messages.append({"role": "user", "content": user_input})
        self._chat_repo.add_message(conv_id, "user", user_input)

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
        import threading

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
        if not self._title_generated and self.turn_count >= 1:
            self._title_generated = True
            thread = threading.Thread(
                target=self._generate_title,
                args=(user_input, response_text),
                daemon=True,
            )
            thread.start()

        # Cleanup
        self._pending_content.clear()
        self._pending_thinking.clear()
        self._pending_user_input = ""

    def _generate_title(self, first_message: str, first_response: str) -> None:
        """Generate a short title for the conversation using AI."""
        self._title_generated = True
        try:
            prompt = TITLE_PROMPT.format(
                first_message=first_message[:200],
                first_response=first_response[:200],
            )
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
        """Switch the model mid-conversation."""
        self._model = model
        logger.info("Switched model to %s", model)

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
