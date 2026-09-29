"""LLM Prompts centralized for both CLI and Web UI usage."""

# --- Gemini Engine Prompts ---

GEMINI_TRANSCRIPTION_PROMPT = """\
Transcribe the following audio accurately and completely.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "<ISO 639-1 code>",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The transcribed text for this segment.",
      "words": [
        {"word": "The", "start": 0.0, "end": 0.3},
        {"word": "transcribed", "start": 0.35, "end": 0.9}
      ]
    }
  ]
}

Rules:
- Split into natural segments (sentences or clauses, ~5-15 seconds each).
- Include word-level timestamps if possible.
- Detect the spoken language automatically.
- Preserve the original language — do NOT translate unless asked.
- CRITICAL: Do NOT hallucinate speech on silence or background noise.
- CRITICAL: If a phrase repeats endlessly, STOP transcribing it after 2 repetitions.
- Return raw JSON only. No explanation, no markdown fences.
"""

GEMINI_DIARIZATION_PROMPT = """\
Transcribe the following audio accurately and completely, identifying each speaker.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "<ISO 639-1 code>",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The transcribed text for this segment.",
      "speaker": "Speaker 1",
      "words": [
        {"word": "The", "start": 0.0, "end": 0.3},
        {"word": "transcribed", "start": 0.35, "end": 0.9}
      ]
    }
  ]
}

Rules:
- Identify each distinct speaker and label them consistently (Speaker 1, Speaker 2, etc.).
- Start a new segment when the speaker changes OR at natural sentence boundaries.
- Split into natural segments (sentences or clauses, ~5-15 seconds each).
- Include word-level timestamps if possible.
- Detect the spoken language automatically.
- Preserve the original language — do NOT translate unless asked.
- CRITICAL: Do NOT hallucinate speech on silence or background noise.
- CRITICAL: If a phrase repeats endlessly, STOP transcribing it after 2 repetitions.
- Return raw JSON only. No explanation, no markdown fences.
"""

GEMINI_TRANSLATE_PROMPT = """\
Transcribe the following audio and translate everything to English.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The translated English text for this segment.",
      "words": []
    }
  ]
}

Rules:
- Translate ALL speech to English.
- Split into natural segments (sentences or clauses).
- CRITICAL: Do NOT hallucinate speech on silence or background noise.
- CRITICAL: If a phrase repeats endlessly, STOP transcribing it after 2 repetitions.
- Return raw JSON only. No explanation, no markdown fences.
"""

GEMINI_DIARIZATION_TRANSLATE_PROMPT = """\
Transcribe the following audio, translate everything to English, and identify each speaker.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The translated English text for this segment.",
      "speaker": "Speaker 1",
      "words": []
    }
  ]
}

Rules:
- Identify each distinct speaker and label them consistently (Speaker 1, Speaker 2, etc.).
- Start a new segment when the speaker changes OR at natural sentence boundaries.
- Translate ALL speech to English.
- Split into natural segments (sentences or clauses).
- CRITICAL: Do NOT hallucinate speech on silence or background noise.
- CRITICAL: If a phrase repeats endlessly, STOP transcribing it after 2 repetitions.
- Return raw JSON only. No explanation, no markdown fences.
"""

GEMINI_TRANSCRIPTION_TEXT_ONLY_PROMPT = """\
Transcribe the following audio accurately and completely.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "<ISO 639-1 code>",
  "segments": [
    {
      "id": 0,
      "text": "The transcribed text for this segment."
    }
  ]
}

Rules:
- Split into natural segments (sentences or clauses, ~5-15 seconds each).
- Detect the spoken language automatically.
- Preserve the original language — do NOT translate unless asked.
- Do NOT include any timestamps.
- CRITICAL: Do NOT hallucinate speech on silence or background noise.
- CRITICAL: If a phrase repeats endlessly, STOP transcribing it after 2 repetitions.
- Return raw JSON only. No explanation, no markdown fences.
"""

GEMINI_DIARIZATION_TEXT_ONLY_PROMPT = """\
Transcribe the following audio accurately and completely, identifying each speaker.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "<ISO 639-1 code>",
  "segments": [
    {
      "id": 0,
      "text": "The transcribed text for this segment.",
      "speaker": "Speaker 1"
    }
  ]
}

Rules:
- Identify each distinct speaker and label them consistently (Speaker 1, Speaker 2, etc.).
- Start a new segment ONLY when the speaker changes. Put all contiguous speech by the same speaker into a single segment, even if it is long.
- Detect the spoken language automatically.
- Preserve the original language — do NOT translate unless asked.
- Do NOT include any timestamps.
- CRITICAL: Do NOT hallucinate speech on silence or background noise.
- CRITICAL: If a phrase repeats endlessly, STOP transcribing it after 2 repetitions.
- Return raw JSON only. No explanation, no markdown fences.
"""

SESSION_SUMMARY_PROMPT = """\
You are a research assistant. Below is a transcript of a research session: a series of searches and the AI-synthesized answers for each.

Write a concise executive summary using EXACTLY this structure (use these headings verbatim):

**Session Overview**
2-3 sentences describing the overarching theme and how the inquiry evolved across searches.

**Key Insights**
A bullet list of the most important conclusions or ideas that emerged. Each bullet should be a statement of something learned, not a question.

**Open Threads**
A bullet list of unresolved tensions, contradictions, or gaps that the session raised but did not close. Write each as a declarative statement naming the tension — do NOT pose follow-up questions or suggest directions. Example: "The boundary between productive pattern-use and rigid conditioning was raised but not resolved."

Constraints: 300-500 words total. No title line. No preamble. Start directly with **Session Overview**.

---
{transcript}
"""
