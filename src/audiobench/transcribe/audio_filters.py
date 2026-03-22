"""Transcript post-processing filters.

Catches quality issues that slip through the engine's decoding
safeguards — repetition loops, character spam, etc.
"""

from __future__ import annotations

import re

from audiobench.core.logger_factory import get_logger

logger = get_logger("core.filters")


def collapse_repetitions(text: str, max_repeats: int = 2) -> str:
    """Collapse 3+ consecutive repeated phrases into max_repeats.

    Handles three levels of repetition:
    1. Sentence-level: "I'll go. I'll go. I'll go." → "I'll go. I'll go."
    2. Phrase-level: "na, na, na, na, na" → "na, na"
    3. Character-level: "ZZZZZZZZ" → "ZZ"

    Args:
        text: Raw segment text.
        max_repeats: Maximum allowed consecutive repetitions.

    Returns:
        Cleaned text with repetitions collapsed.
    """
    if not text or len(text) < 10:
        return text

    original = text

    # 1. Character-level: collapse 4+ identical characters → max_repeats
    text = re.sub(r"(.)\1{3,}", r"\1" * max_repeats, text)

    # 2. Word/phrase-level: collapse repeated n-grams (1-8 words)
    # Matches patterns like "word word word" or "a b c, a b c, a b c"
    for n in range(8, 0, -1):
        # Match a phrase of n words repeated 3+ times
        pattern = r"((?:\S+\s*){" + str(n) + r"})\s*(?:\1\s*){2,}"
        replacement = r"\1 " * max_repeats
        text = re.sub(pattern, replacement.strip(), text, flags=re.IGNORECASE)

    # 3. Sentence-level: split on period, collapse identical consecutive sentences
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) > 2:
        collapsed: list[str] = []
        repeat_count = 1
        for i, sentence in enumerate(sentences):
            if i > 0 and sentence.lower() == sentences[i - 1].lower():
                repeat_count += 1
                if repeat_count <= max_repeats:
                    collapsed.append(sentence)
            else:
                repeat_count = 1
                collapsed.append(sentence)
        if len(collapsed) < len(sentences):
            text = ". ".join(collapsed)
            if not text.endswith("."):
                text += "."

    # Clean up extra whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()

    if text != original:
        reduction = len(original) - len(text)
        logger.debug(
            "Repetition filter: removed %d chars (%.0f%% reduction)",
            reduction,
            reduction / len(original) * 100 if original else 0,
        )

    return text


def fix_broken_words(text: str) -> str:
    """Fix subword tokenization spacing artifacts.

    Whisper's BPE tokenizer sometimes splits words with spaces:
      "Tim e" → "Time"
      "bes t" → "best"
      "wor ld" → "world"

    Pattern: merge "X y" → "Xy" when y is a single lowercase letter
    attached to the end of a preceding word fragment.
    """
    if not text:
        return text

    # Pattern: word fragment + space + single lowercase letter at word boundary
    # e.g., "Tim e" but not "I am" (both parts are real words)
    text = re.sub(r"(\w{2,}) ([a-z])\b", _merge_if_valid, text)

    return text


def _merge_if_valid(match: re.Match) -> str:
    """Merge fragments if the combined form looks more natural."""
    prefix = match.group(1)
    suffix = match.group(2)
    merged = prefix + suffix

    # If suffix is a single letter and prefix is 2+ chars,
    # the split is almost certainly a tokenization artifact
    # (real English words rarely follow a "Xy z" pattern where z is lone)
    if len(suffix) == 1 and len(prefix) >= 2:
        return merged

    return match.group(0)
