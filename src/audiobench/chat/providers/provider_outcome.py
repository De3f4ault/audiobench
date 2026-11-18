"""LlmOutcome — the unified return type for ProviderRouter.generate()."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OutcomeType(Enum):
    OK        = "ok"        # Primary or fallback returned a real LLM response
    DEGRADED  = "degraded"  # Digest — usable but not LLM-generated
    ABSTAINED = "abstained" # No output; caller should skip this operation silently
    FAILED    = "failed"    # Fatal error caught; content is ""; error holds original exception


@dataclass
class LlmOutcome:
    """Always returned by ProviderRouter.generate(). Never raises on provider failure.

    Contract:
      type == OK        → content is a valid LLM response; error is None
      type == DEGRADED  → content is a digest; degraded_mode says which kind; error is None
      type == ABSTAINED → content is ""; caller should discard the turn silently
      type == FAILED    → content is ""; error holds the original ProviderFatalError;
                          use generate_text() if you want it re-raised as an exception.

    Callers that check `outcome.content` without first checking `outcome.type` will silently
    get empty strings on FAILED outcomes. Always gate on type first.
    """
    type: OutcomeType
    content: str
    provider: str | None = None           # "ollama", "gemini", "digest", None
    fallback_from: str | None = None      # set when at least one slot was skipped
    fallback_reason: str | None = None    # the error that triggered the skip
    degraded_mode: str | None = None      # "extractive", "notice", "passthrough", "silent"
    error: Exception | None = None        # set only when type == FAILED

    @property
    def is_usable(self) -> bool:
        return self.type in (OutcomeType.OK, OutcomeType.DEGRADED)
