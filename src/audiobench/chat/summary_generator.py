"""Generates structured summaries of chat conversations using an LLM."""

import json
from dataclasses import dataclass
from typing import Any

from audiobench.chat.providers.provider_errors import AIError
from audiobench.chat.providers.router import router_for_settings
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

logger = get_logger("chat.summary")


@dataclass
class ConversationSummaryResult:
    narrative: str
    drift_phases: list[dict[str, Any]]
    key_insights: list[dict[str, Any]]
    open_threads: list[dict[str, Any]]
    refined_title: str


class SummaryGenerator:
    """Uses ProviderRouter to generate structured JSON summaries of chat sessions."""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.ollama_model
        self.router = router_for_settings(settings)
        if model_name:
            self.router.set_primary_model(model_name)

    def generate(self, messages: list[dict]) -> ConversationSummaryResult | None:
        """Generate a summary from a list of chat messages."""
        if not messages or len(messages) < 6:  # typically 3 turns (user+assistant)
            return None

        # Build conversation transcript
        conv_text = ""
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                continue
            role_name = "User" if msg["role"] == "user" else "Assistant"
            conv_text += f"Turn {i // 2 + 1} - {role_name}: {msg.get('content', '')}\n\n"

        prompt = (
            "Analyze the following conversation and provide a structured summary in JSON format.\n\n"
            "The JSON must have EXACTLY these five keys:\n"
            "1. 'narrative': A cohesive paragraph (string) detailing how the conversation evolved, the main topics discussed, and the user's apparent goals.\n"
            "2. 'drift_phases': A list of objects. Each object should have 'phase' (int), 'label' (string, a short topic name), and 'turn_range' (list of two ints, [start_turn, end_turn]).\n"
            "3. 'key_insights': A list of objects. Each object should have 'turn' (int) and 'insight' (string, a key takeaway or decision made).\n"
            "4. 'open_threads': A list of objects representing unresolved questions or topics. Each object should have 'question' (string) and 'context' (string).\n"
            "5. 'refined_title': A short, descriptive title (string) for the entire conversation (max 5 words).\n\n"
            "Respond ONLY with valid JSON. Do not include markdown formatting like ```json or any other text.\n\n"
            "CONVERSATION:\n"
            f"{conv_text}"
        )

        try:
            raw_response = self.router.generate_text(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                digest_mode="silent",
            )

            # Attempt to extract JSON if it was wrapped in markdown
            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[1].strip()

            data = json.loads(raw_response)

            return ConversationSummaryResult(
                narrative=data.get("narrative", "No narrative provided."),
                drift_phases=data.get("drift_phases", []),
                key_insights=data.get("key_insights", []),
                open_threads=data.get("open_threads", []),
                refined_title=data.get("refined_title", "Chat Session"),
            )

        except AIError as e:
            logger.error("AI error during summary generation: %s", e)
            return None
        except json.JSONDecodeError as e:
            logger.error("Failed to parse summary JSON: %s\nRaw output: %s", e, raw_response)
            return None
        except Exception as e:
            logger.error("Unexpected error in summary generation: %s", e)
            return None
