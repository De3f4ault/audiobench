"""Tests for AI synthesis citation cleaning and normalization."""

from __future__ import annotations

import pytest

from audiobench.chat.synthesis_engine import _clean_synthesis_citations


def test_clean_synthesis_citations_plain():
    text = "Alan Watts explores the nature of ego [1] while Dostoyevsky focuses on alienation [2]."
    assert _clean_synthesis_citations(text) == text


def test_clean_synthesis_citations_dagger_offset():
    # OpenAI RAG / browsing style dagger offsets
    text = "The nature of consciousness is discussed [ 2†L9-L15] and highlighted in [ 3†L1-L10]."
    expected = "The nature of consciousness is discussed [2] and highlighted in [3]."
    assert _clean_synthesis_citations(text) == expected


def test_clean_synthesis_citations_pipe_offset():
    text = "Suffering and redemption are addressed [ 1|L27-L34] and later reaffirmed [ 4|L5-L10]."
    expected = "Suffering and redemption are addressed [1] and later reaffirmed [4]."
    assert _clean_synthesis_citations(text) == expected


def test_clean_synthesis_citations_spaces_and_lines():
    text = "Citation with spaces [ 5 ] and line label [6 L12-L18] are normalized."
    expected = "Citation with spaces [5] and line label [6] are normalized."
    assert _clean_synthesis_citations(text) == expected


def test_clean_synthesis_citations_empty_and_no_citations():
    assert _clean_synthesis_citations("") == ""
    assert _clean_synthesis_citations("Just plain text with no citations.") == "Just plain text with no citations."
