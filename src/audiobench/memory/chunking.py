"""Semantic chunking pipeline.

Implements text cleaning, speaker diarization splitting, and semantic boundary detection
using a sliding window over sentence groups.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from audiobench.core.constants import ENGLISH_STOP_WORDS
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.memory.boilerplate import clean_audio_disclaimers, is_pure_boilerplate
from audiobench.memory.singletons import get_boundary_embedder

try:
    import os

    import nltk  # type: ignore[import-untyped]

    is_offline = get_settings().offline_mode or os.environ.get("HF_HUB_OFFLINE") == "1"
    if not is_offline:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                pass

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
    from nltk.tokenize import sent_tokenize  # type: ignore[import-untyped]
except ImportError:
    # Fallback if nltk is somehow missing
    def sent_tokenize(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


logger = get_logger("memory.chunking")

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class SpeakerBlock:
    speaker: str
    text: str


@dataclass
class Chunk:
    content: str
    uuid: str
    tier: int
    speaker: str | None = None
    segment_ids: list[int] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None


@dataclass
class ParentGroup:
    parent_text: str
    children: list[Chunk]


# ---------------------------------------------------------------------------
# 4.1 Text Cleaning
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Clean Whisper hallucinations, null bytes, and repetition loops."""
    # Strip NUL bytes
    text = text.replace("\x00", "")

    # Strip Whisper non-speech tags
    tags_to_remove = [
        r"\[BLANK_AUDIO\]",
        r"\[MUSIC\]",
        r"\[music\]",
        r"\[Music\]",
        r"\[Silence\]",
        r"\[silence\]",
        r"\(silence\)",
        r"\(Music\)",
        r"\[Applause\]",
        r"\(Applause\)",
        r"\[Laughter\]",
        r"\(Laughter\)",
    ]
    for tag in tags_to_remove:
        text = re.sub(tag, "", text)

    # Collapse repeated word runs (e.g., "the the the" -> "the")
    # This is a naive regex for exact word repetition
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Detect trailing repetition loops (e.g. model gets stuck repeating the last sentence)
    sentences = sent_tokenize(text)
    if len(sentences) >= 3:
        # Check last 3 sentences. If they are identical, trim them and assume it's a loop.
        if sentences[-1].lower() == sentences[-2].lower() == sentences[-3].lower():
            # Find the start of the loop (first sentence that is identical to the last)
            loop_sentence = sentences[-1].lower()
            trim_idx = len(sentences)
            for i in range(len(sentences) - 1, -1, -1):
                if sentences[i].lower() == loop_sentence:
                    trim_idx = i
                else:
                    break
            text = " ".join(sentences[: trim_idx + 1])  # Keep one instance

    # Strip publisher preambles and closing production credits.
    # Covers all major audiobook producers, streaming platforms, public-domain
    # repositories (LibriVox and phonetic variants), and podcast conventions.
    text = clean_audio_disclaimers(text)

    return text


# ---------------------------------------------------------------------------
# 4.2 AdvancedSemanticChunker
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class AdvancedSemanticChunker:
    """Chunks text using sliding window cosine similarity on sentence groups."""

    def __init__(self, breakpoint_percentile: float, max_tokens: int, sentence_group_size: int):
        self.breakpoint_percentile = breakpoint_percentile
        self.max_tokens = max_tokens
        self.sentence_group_size = sentence_group_size

    def chunk(self, text: str, speaker: str | None = None) -> list[Chunk]:
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        # If it's too short to form two groups, return as one chunk
        if len(sentences) <= self.sentence_group_size:
            return [Chunk(content=text, uuid=str(uuid.uuid4()), tier=3, speaker=speaker)]

        # Group sentences
        groups = []
        for i in range(0, len(sentences), self.sentence_group_size):
            group_text = " ".join(sentences[i : i + self.sentence_group_size])
            groups.append(group_text)

        def _stable_uuid(txt: str) -> str:
            return str(uuid.uuid5(uuid.NAMESPACE_OID, txt))

        if len(groups) < 2:
            return [Chunk(content=text, uuid=_stable_uuid(text), tier=3, speaker=speaker)]

        # Embed groups
        embedder = get_boundary_embedder()
        # SentenceTransformers returns a list of tensors or numpy arrays
        embeddings = embedder.encode(groups, convert_to_numpy=True)

        # Calculate cosine distances (1 - cosine_similarity) between adjacent groups
        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            distances.append(1.0 - sim)

        # Determine threshold
        if not distances:
            return [Chunk(content=text, uuid=_stable_uuid(text), tier=3, speaker=speaker)]

        threshold = np.percentile(distances, self.breakpoint_percentile)

        # Split groups based on threshold
        chunks = []
        current_chunk_sentences: list[str] = []

        # We need to map groups back to original sentences for precise chunking
        group_idx = 0
        sentence_idx = 0

        while sentence_idx < len(sentences):
            current_chunk_sentences.append(sentences[sentence_idx])

            # If this sentence is the end of a group, check if we need to split
            # A group ends when we've added `sentence_group_size` sentences, or we reach the end
            if (sentence_idx + 1) % self.sentence_group_size == 0 or (sentence_idx + 1) == len(
                sentences
            ):
                # If there's a distance measured for the current group
                if group_idx < len(distances):
                    if distances[group_idx] > threshold:
                        # Split here
                        chunk_text = " ".join(current_chunk_sentences)
                        chunks.append(
                            Chunk(
                                content=chunk_text,
                                uuid=_stable_uuid(chunk_text),
                                tier=3,
                                speaker=speaker,
                            )
                        )
                        current_chunk_sentences = []
                group_idx += 1

            sentence_idx += 1

        # Add remaining sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(
                Chunk(content=chunk_text, uuid=_stable_uuid(chunk_text), tier=3, speaker=speaker)
            )

        # Enforce max_tokens safeguard (SentenceSplitter fallback)
        final_chunks: list[Chunk] = []
        try:
            from llama_index.core.node_parser import SentenceSplitter

            splitter = SentenceSplitter(chunk_size=self.max_tokens, chunk_overlap=20)
        except ImportError:
            splitter = None

        for c in chunks:
            if splitter is not None and len(c.content.split()) > self.max_tokens:
                # Approximation of tokens by words, if it's too long, use LLamaIndex's splitter
                sub_texts = splitter.split_text(c.content)
                for st in sub_texts:
                    final_chunks.append(
                        Chunk(content=st, uuid=_stable_uuid(st), tier=3, speaker=speaker)
                    )
            else:
                final_chunks.append(c)

        return final_chunks


# ---------------------------------------------------------------------------
# 4.3 / 4.4 Content-Aware Router and Speaker Splitting
# ---------------------------------------------------------------------------


def speaker_turn_splitter(diarized_segments: list[dict[str, Any]]) -> list[SpeakerBlock]:
    """Splits transcript on speaker turn boundaries.

    Expects segments with 'speaker' and 'text' keys.
    """
    blocks = []
    current_speaker = None
    current_text: list[str] = []

    for seg in diarized_segments:
        spk = seg.get("speaker", "UNKNOWN")
        txt = seg.get("text", "").strip()

        if not txt:
            continue

        if spk != current_speaker:
            if current_speaker is not None and current_text:
                blocks.append(SpeakerBlock(speaker=current_speaker, text=" ".join(current_text)))
            current_speaker = spk
            current_text = [txt]
        else:
            current_text.append(txt)

    if current_speaker is not None and current_text:
        blocks.append(SpeakerBlock(speaker=current_speaker, text=" ".join(current_text)))

    return blocks


def norm_punct(t: str) -> str:
    """Normalize whitespace around punctuation and typographic variants."""
    t = t.replace("—", " ").replace("–", " ")
    t = re.sub(r"\s+([?.!,;:\'\"])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def strip_all_punct(t: str) -> str:
    """Strip all non-alphanumeric characters for robust fallback matching."""
    return re.sub(r"[^a-zA-Z0-9]", "", t.lower())


def align_cft_to_ft(cft: str, ft: str) -> list[int]:
    """Map each character index in cleaned_ft back to raw full_text."""
    cft_to_ft = [0] * len(cft)
    ft_ptr = 0
    cft_ptr = 0

    while cft_ptr < len(cft) and ft_ptr < len(ft):
        c = cft[cft_ptr]
        idx = ft.find(c, ft_ptr)
        if idx != -1:
            ft_ptr = idx
            cft_to_ft[cft_ptr] = ft_ptr
            cft_ptr += 1
            ft_ptr += 1
        else:
            cft_ptr += 1
    return cft_to_ft


def attribute_chunks_to_segments(
    chunks: list[Chunk],
    raw_text: str,
    segments: list[dict[str, Any]],
) -> list[Chunk]:
    """Map semantic chunks back to physical transcript segments.

    Sets segment_ids, start_time, and end_time on each Chunk instance.
    Uses monotonic subsequence alignment to ground cleaned chunk text in raw full_text spans.
    """
    if not chunks or not segments or not raw_text:
        return chunks

    raw_ft = raw_text or ""
    # 1. Compute segment spans in raw full_text
    seg_spans = []
    curr_offset = 0
    for seg in segments:
        sid = seg.get("id") if isinstance(seg, dict) else getattr(seg, "id", None)
        stext = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        s_start = seg.get("start_time", 0.0) if isinstance(seg, dict) else getattr(seg, "start_time", 0.0)
        s_end = seg.get("end_time", 0.0) if isinstance(seg, dict) else getattr(seg, "end_time", 0.0)

        st_clean = stext.strip()
        if not st_clean:
            continue
        idx = raw_ft.find(st_clean, curr_offset)
        if idx == -1:
            idx = raw_ft.find(st_clean)
        if idx != -1:
            seg_spans.append({
                "id": sid,
                "char_start": idx,
                "char_end": idx + len(st_clean),
                "start_time": s_start,
                "end_time": s_end,
            })
            curr_offset = idx + len(st_clean)

    if not seg_spans:
        return chunks

    # 2. Clean full_text and align to raw full_text
    cft = _clean_text(raw_ft)
    cft_to_ft = align_cft_to_ft(cft, raw_ft)

    cft_norm = norm_punct(cft)
    ft_alphanum = strip_all_punct(raw_ft)
    ft_alphanum_to_raw = [raw_i for raw_i, ch in enumerate(raw_ft) if ch.isalnum()]

    cft_cursor = 0

    for chunk in chunks:
        c_clean = chunk.content.strip()
        if not c_clean:
            continue

        c_start_raw: int | None = None
        c_end_raw: int | None = None

        # Tier 1: exact cft match
        idx = cft.find(c_clean, cft_cursor)
        if idx != -1:
            cft_cursor = idx
            c_start_raw = cft_to_ft[idx]
            end_idx = min(len(cft_to_ft) - 1, idx + len(c_clean) - 1)
            c_end_raw = cft_to_ft[end_idx] + 1
        else:
            # Tier 2: prefix cft match
            prefix = c_clean[:80]
            idx = cft.find(prefix, cft_cursor)
            if idx != -1:
                cft_cursor = idx
                c_start_raw = cft_to_ft[idx]
                if idx + len(c_clean) >= len(cft):
                    c_end_raw = len(raw_ft)
                else:
                    end_idx = min(len(cft_to_ft) - 1, idx + len(c_clean) - 1)
                    c_end_raw = cft_to_ft[end_idx] + 1
            else:
                # Tier 3: normalized cft match
                c_norm = norm_punct(c_clean)
                idx_norm = cft_norm.find(c_norm)
                if idx_norm != -1:
                    prefix_words = c_clean.split()[:4]
                    pw = " ".join(prefix_words)
                    idx_pw = cft.find(pw, cft_cursor)
                    if idx_pw == -1:
                        idx_pw = cft.find(pw)
                    if idx_pw != -1:
                        cft_cursor = idx_pw
                        c_start_raw = cft_to_ft[idx_pw]
                        end_idx = min(len(cft_to_ft) - 1, idx_pw + len(c_clean) - 1)
                        c_end_raw = cft_to_ft[end_idx] + 1

        # Fallback search directly in raw full_text
        if c_start_raw is None:
            # Tier 4: exact raw ft match
            idx_raw = raw_ft.find(c_clean)
            if idx_raw != -1:
                c_start_raw = idx_raw
                c_end_raw = idx_raw + len(c_clean)
            else:
                # Tier 5: normalized raw ft match
                raw_words = [w for w in c_clean.split() if len(w) >= 2]
                words = [w for w in raw_words if w.lower() not in ENGLISH_STOP_WORDS]
                if len(words) < 4 and len(raw_words) >= 4:
                    words = raw_words
                if len(words) >= 4:
                    search_phrase = " ".join(words[:4])
                    idx_sp = raw_ft.find(search_phrase)
                    if idx_sp == -1:
                        idx_sp = raw_ft.lower().find(search_phrase.lower())
                    if idx_sp != -1:
                        c_start_raw = idx_sp
                        c_end_raw = min(len(raw_ft), idx_sp + len(c_clean))

        if c_start_raw is None:
            # Tier 6: prefix in raw ft fallback
            idx_pfx = raw_ft.find(c_clean[:40])
            if idx_pfx == -1:
                idx_pfx = raw_ft.lower().find(c_clean[:40].lower())
            if idx_pfx != -1:
                c_start_raw = idx_pfx
                c_end_raw = min(len(raw_ft), idx_pfx + len(c_clean))

        if c_start_raw is None:
            # Tier 7: alphanumeric stripped fallback
            c_alpha = strip_all_punct(c_clean)
            pos_alpha = ft_alphanum.find(c_alpha[:50])
            if pos_alpha != -1 and pos_alpha < len(ft_alphanum_to_raw):
                c_start_raw = ft_alphanum_to_raw[pos_alpha]
                end_pos_alpha = min(len(ft_alphanum_to_raw) - 1, pos_alpha + len(c_alpha) - 1)
                c_end_raw = ft_alphanum_to_raw[end_pos_alpha] + 1

        if c_start_raw is not None and c_end_raw is not None:
            matched_segs = [
                s for s in seg_spans
                if max(s["char_start"], c_start_raw) < min(s["char_end"], c_end_raw)
            ]
            if matched_segs:
                chunk.segment_ids = [s["id"] for s in matched_segs if s["id"] is not None]
                start_times = [s["start_time"] for s in matched_segs if s["start_time"] is not None]
                end_times = [s["end_time"] for s in matched_segs if s["end_time"] is not None]
                if start_times:
                    chunk.start_time = min(start_times)
                if end_times:
                    chunk.end_time = max(end_times)

    return chunks


def content_aware_router(
    text: str,
    diarized_segments: list[dict[str, Any]] | None = None,
    segments: list[dict[str, Any]] | None = None,
) -> list[Chunk]:
    """Route text to appropriate chunking strategy based on length and diarization."""
    settings = get_settings()
    cleaned_text = _clean_text(text)

    def _stable_uuid(txt: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, txt))

    # Guard: if the entire cleaned transcript is pure boilerplate (e.g. a standalone
    # publisher stamp file), skip it entirely.  The sweep loop in server.py already
    # handles an empty return value with 'if chunks:'.
    if is_pure_boilerplate(cleaned_text):
        logger.debug("Skipping pure-boilerplate transcript (all content is publisher stamp)")
        return []

    # 1. Short text -> single chunk
    if len(cleaned_text) < settings.chunk_short_threshold:
        chunk = Chunk(content=cleaned_text, uuid=_stable_uuid(cleaned_text), tier=3)
        if is_pure_boilerplate(chunk.content):
            return []
        chunks = [chunk]
        if segments:
            chunks = attribute_chunks_to_segments(chunks, text, segments)
        return chunks

    chunker = AdvancedSemanticChunker(
        breakpoint_percentile=settings.chunk_breakpoint_percentile,
        max_tokens=settings.chunk_max_tokens,
        sentence_group_size=settings.chunk_sentence_group_size,
    )

    chunks = []

    # 2. Diarized text -> split by speaker, then chunk
    if diarized_segments:
        # We need to clean the text inside the segments too
        clean_segments = []
        for seg in diarized_segments:
            clean_segments.append(
                {"speaker": seg.get("speaker", "UNKNOWN"), "text": _clean_text(seg.get("text", ""))}
            )

        blocks = speaker_turn_splitter(clean_segments)
        for block in blocks:
            # If a block is very short, don't semantic-chunk it further
            if len(block.text) < settings.chunk_short_threshold:
                chunk = Chunk(
                    content=block.text,
                    uuid=_stable_uuid(block.text),
                    tier=3,
                    speaker=block.speaker,
                )
                if not is_pure_boilerplate(chunk.content):
                    chunks.append(chunk)
            else:
                chunks.extend(
                    c for c in chunker.chunk(block.text, speaker=block.speaker)
                    if not is_pure_boilerplate(c.content)
                )
        if segments and chunks:
            chunks = attribute_chunks_to_segments(chunks, text, segments)
        return chunks

    # 3. Long text -> AdvancedSemanticChunker
    # Long threshold check (if very long, might need different percentile or max_tokens handling, handled in Chunker)
    if len(cleaned_text) >= settings.chunk_long_threshold:
        pass  # The chunker logic handles this

    chunks = [
        c for c in chunker.chunk(cleaned_text)
        if not is_pure_boilerplate(c.content)
    ]
    if segments and chunks:
        chunks = attribute_chunks_to_segments(chunks, text, segments)
    return chunks


# ---------------------------------------------------------------------------
# 4.5 Parent-Child Grouper
# ---------------------------------------------------------------------------


def parent_child_grouper(chunks: list[Chunk]) -> list[ParentGroup]:
    """Group sentence-level chunks into topic-level parents.

    For now, this simply groups adjacent chunks by speaker. If no speaker,
    it groups into fixed sizes or uses semantic similarity.
    Implementation here is a basic placeholder for Phase 4.5
    """
    if not chunks:
        return []

    groups = []
    current_children: list[Chunk] = []
    current_speaker = chunks[0].speaker

    for chunk in chunks:
        # Basic grouping strategy: break on speaker change, or every 5 chunks if no speaker
        if chunk.speaker != current_speaker or len(current_children) >= 5:
            if current_children:
                parent_text = " ".join([c.content for c in current_children])
                groups.append(ParentGroup(parent_text=parent_text, children=current_children))
            current_children = [chunk]
            current_speaker = chunk.speaker
        else:
            current_children.append(chunk)

    if current_children:
        parent_text = " ".join([c.content for c in current_children])
        groups.append(ParentGroup(parent_text=parent_text, children=current_children))

    return groups
