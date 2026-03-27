"""Prompt templates for AI transcript analysis.

Each template takes a transcript text and returns a formatted prompt
suitable for LLM processing.
"""

from __future__ import annotations

# System prompt for single-question transcript analysis (audiobench ask)
TRANSCRIPT_SYSTEM = (
    "You are an expert analyst specializing in audio transcripts. "
    "Your role is to provide thorough, well-structured answers "
    "that give the user genuine insight — not just surface-level "
    "summaries.\n\n"
    "Guidelines:\n"
    "- Give detailed, substantive answers that fully address the question.\n"
    "- Support your points with specific quotes or "
    "references from the transcript.\n"
    "- Use clear structure: headers, bullet points, and "
    "numbered lists where they aid readability.\n"
    "- Match your response depth to the complexity of the question — "
    "simple questions get focused answers, complex questions get thorough analysis.\n"
    "- If you identify patterns, themes, or implications beyond the literal question, share them.\n"
    "- Be direct — no preamble, no filler, no pleasantries. Just quality analysis."
)


def summarize(transcript: str) -> str:
    """Structured summary of a transcript."""
    return (
        "Provide a thorough summary of the following transcript.\n\n"
        "Structure your summary as follows:\n"
        "1. **Overview** — A 2-3 sentence high-level summary of what this is about.\n"
        "2. **Key Topics & Themes** — The main subjects discussed, with brief context for each.\n"
        "3. **Notable Quotes** — 2-4 direct quotes that capture the essence of the content.\n"
        "4. **Key Takeaways** — The most important points the listener should remember.\n\n"
        "Be thorough but well-organized. Use the transcript's own language "
        "and specific details rather than vague generalizations.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )


def action_items(transcript: str) -> str:
    """Extract action items from a meeting transcript."""
    return (
        "Extract all action items, tasks, decisions, and commitments from this transcript.\n\n"
        "For each item, provide:\n"
        "- **Who** is responsible (or 'Unassigned' if unclear)\n"
        "- **What** they need to do (specific and actionable)\n"
        "- **Priority** (High / Medium / Low) based on the urgency "
        "conveyed\n"
        "- **Context** — a brief note on why this was discussed "
        "or any relevant deadline mentioned\n\n"
        "Also note any decisions that were made (even if they don't create action items) "
        "and any unresolved questions or topics that need follow-up.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )


def rewrite(transcript: str) -> str:
    """Clean up disfluencies and improve readability."""
    return (
        "Rewrite this transcript to be clear, polished, and readable while "
        "preserving the speaker's authentic voice and style.\n\n"
        "Guidelines:\n"
        "- Remove filler words (um, uh, like, you know) and false starts.\n"
        "- Fix grammar and improve sentence structure.\n"
        "- Break into logical paragraphs.\n"
        "- Preserve the original meaning, tone, and personality.\n"
        "- Keep distinctive phrases, metaphors, and expressions the speaker uses.\n"
        "- If there are multiple speakers, clearly indicate speaker changes.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )


def translate_text(transcript: str, target_language: str) -> str:
    """Translate transcript to a target language."""
    return (
        f"Translate the following text to {target_language}.\n\n"
        "Guidelines:\n"
        "- Translate naturally — adapt idioms and expressions to feel native "
        f"in {target_language} rather than doing a literal word-for-word translation.\n"
        "- Maintain the original formatting, paragraph structure, and speaker labels.\n"
        "- Preserve proper nouns, technical terms, and names as-is unless they "
        "have standard translations.\n\n"
        f"TEXT:\n{transcript}"
    )


def qa(transcript: str, question: str) -> str:
    """Answer a question about a transcript."""
    return (
        "Answer the following question based on the transcript below.\n\n"
        "Guidelines:\n"
        "- Provide a thorough, well-structured answer.\n"
        "- Reference specific parts of the transcript to support your answer.\n"
        "- Include relevant quotes where they add value.\n"
        "- If the question asks for analysis or interpretation, go deeper than "
        "just restating what was said — identify patterns, implications, and connections.\n"
        "- If the answer is not in the transcript, say so clearly and explain "
        "what related information IS available.\n\n"
        f"QUESTION: {question}\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )


# ── Chat Mode Prompts ──────────────────────────────────────

CHAT_SYSTEM = (
    "You are a knowledgeable AI analyst for AudioBench, helping users "
    "deeply understand their audio transcripts and engage in broader "
    "intellectual discussion. You have access to the user's transcribed "
    "content shown below.\n\n"
    "Your approach:\n"
    "- Provide thorough, insightful responses that give genuine value.\n"
    "- Reference specific parts of the transcript — use direct quotes when relevant.\n"
    "- Use clear structure (headers, bullet points, numbered lists) for complex answers.\n"
    "- Match response depth to the question: simple questions get focused answers, "
    "complex questions get detailed exploration with evidence.\n"
    "- Identify patterns, themes, and connections the user might not have noticed.\n"
    "- When analyzing, go beyond surface-level — explore implications, context, and significance.\n"
    "- Be direct and substantive — skip pleasantries, deliver quality analysis.\n"
    "- When a question relates to the loaded transcripts, ground your answer in "
    "the transcript content with specific references and quotes.\n"
    "- When a question is about a topic outside the transcripts, use your general "
    "knowledge freely. Be helpful and thorough, but note that you're drawing from "
    "general knowledge rather than the loaded transcripts.\n"
    "- You can also make connections between the transcripts and outside knowledge "
    "when it enriches the discussion.\n\n"
    "{transcript_context}"
)

CHAT_SYSTEM_NO_CONTEXT = (
    "You are a knowledgeable AI assistant for AudioBench. "
    "The user hasn't loaded any transcripts yet — suggest using /load <ID> "
    "to add transcript context for analysis.\n\n"
    "Without transcripts, you can still:\n"
    "- Discuss audio/transcription concepts and best practices\n"
    "- Help plan how to analyze their content\n"
    "- Answer general questions thoughtfully and thoroughly\n"
    "- Discuss literature, philosophy, and any topic with depth\n\n"
    "Be direct, helpful, and substantive in your responses."
)

TITLE_PROMPT = (
    "Based on this short conversation, generate a title of at most 6 words. "
    "Reply with ONLY the title, no quotes, no punctuation at the end.\n\n"
    "User: {first_message}\n"
    "Assistant: {first_response}"
)


# ── Auto-Bookmarking Prompts ──────────────────────────────

AUTO_BOOKMARK_SYSTEM = (
    "You are an expert audio analyst specializing in transcript annotation. "
    "Your task is to identify the most important moments in a transcript and "
    "return them as structured JSON. You MUST respond with ONLY a valid JSON "
    "array — no markdown, no explanation, no code fences, no preamble. "
    "Just the raw JSON array."
)


def auto_bookmark(transcript_with_timestamps: str, *, focus: str | None = None) -> str:
    """Build prompt for AI auto-bookmark extraction.

    Args:
        transcript_with_timestamps: Transcript text with exact second timestamps.
        focus: Optional user instruction to focus extraction.

    Returns:
        Formatted prompt string.
    """
    focus_block = ""
    if focus:
        focus_block = (
            f"\n**FOCUS INSTRUCTION**: {focus}\n"
            "Prioritize bookmarks related to this focus, but still include "
            "other significant moments.\n"
        )

    return (
        "Analyze the transcript below and identify the most important moments.\n"
        "Each line is prefixed with the EXACT timestamp in seconds: [123.45s]\n\n"
        "For each moment, return a JSON object with these fields:\n"
        '  - "timestamp": EXACT start time in seconds (float) — MUST match a [XXs] marker\n'
        '  - "end_timestamp": EXACT end time in seconds (float) for regions, or null for points\n'
        '  - "name": concise label (max 80 chars) — use the speaker\'s own words when possible\n'
        '  - "type": one of the 5 types below\n'
        '  - "notes": 1-3 sentences explaining WHY this moment matters\n\n'
        "═══ TYPE GUIDE WITH EXAMPLES ═══\n\n"
        '  "highlight" (⭐) — Key insight, decision, or turning point\n'
        '     Example: A speaker says "we\'ve decided to pivot to Rust"\n'
        '     Example: A profound realization or emotional breakthrough\n\n'
        '  "todo" (📌) — Action item, task, commitment, or follow-up\n'
        '     Example: "I need to call Sarah about the budget by Friday"\n'
        '     Example: Any stated intention to do something later\n\n'
        '  "note" (📝) — Interesting context, background info, or observation\n'
        '     Example: Historical background that explains a later point\n'
        '     Example: A tangential but interesting fact or anecdote\n\n'
        '  "bookmark" (🔖) — Important structural moment (intro, conclusion, topic shift)\n'
        '     Example: "Let me now talk about the second quarter"\n'
        '     Example: Opening/closing of a recording or major section\n\n'
        '  "edit" (✂️) — Dead air, filler, tangent, repetition, or cut candidate\n'
        '     Example: Long pause, "um um um", off-topic rambling\n'
        '     Example: Technical interruption, audio glitch, repeated content\n'
        '     ALWAYS use both timestamp AND end_timestamp for edit regions\n\n'
        "═══ RULES ═══\n\n"
        "1. TIMESTAMP PRECISION: Use the EXACT second values from the [XXs] markers.\n"
        "   Do NOT round or approximate. If a segment starts at [223.45s], use 223.45.\n"
        "2. COUNT: Return 5-15 bookmarks scaled to transcript length:\n"
        "   - Under 5 minutes: 3-5 bookmarks\n"
        "   - 5-15 minutes: 5-8 bookmarks\n"
        "   - 15-30 minutes: 8-12 bookmarks\n"
        "   - 30+ minutes: 10-15 bookmarks\n"
        "3. TYPE DIVERSITY: Use ALL 5 types where applicable. Always scan for:\n"
        "   - At least one ✂️ edit region (dead air, filler, cut candidates)\n"
        "   - Any 📌 todo/action items if present\n"
        "   - Structural 🔖 bookmarks for major topic shifts\n"
        "4. REGIONS vs POINTS: Use regions (start + end) for extended passages.\n"
        "   Use points (end_timestamp = null) only for brief, instant moments.\n"
        "5. QUALITY: Only bookmark truly significant moments. No padding.\n"
        "6. OUTPUT: Respond with ONLY the JSON array. No other text.\n"
        f"{focus_block}\n"
        f"TRANSCRIPT:\n{transcript_with_timestamps}"
    )


