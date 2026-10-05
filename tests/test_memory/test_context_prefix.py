"""Unit tests for build_context_prefix."""

from audiobench.memory.context_prefix import build_context_prefix


def test_build_context_prefix_all_fields():
    prefix = build_context_prefix(
        author="Viktor E. Frankl",
        work_title="Man's Search for Meaning",
        chapter_title="Experiences in a Concentration Camp",
    )
    assert prefix == "[Author: Viktor E. Frankl | Work: Man's Search for Meaning | Chapter: Experiences in a Concentration Camp]\n\n"


def test_build_context_prefix_partial_fields():
    # Author and Work only
    p1 = build_context_prefix(
        author="Fyodor Dostoyevsky",
        work_title="Crime and Punishment",
    )
    assert p1 == "[Author: Fyodor Dostoyevsky | Work: Crime and Punishment]\n\n"

    # Chapter only
    p2 = build_context_prefix(
        chapter_title="Chapter 1: The New Rules",
    )
    assert p2 == "[Chapter: Chapter 1: The New Rules]\n\n"


def test_build_context_prefix_never_embeds_full_recording():
    # Explicit test: 'Full Recording' MUST be omitted from prefix
    prefix = build_context_prefix(
        author="Alan Watts",
        work_title="Out of Your Mind",
        chapter_title="Full Recording",
    )
    assert prefix == "[Author: Alan Watts | Work: Out of Your Mind]\n\n"
    assert "Full Recording" not in prefix

    # Case-insensitive check
    prefix_case = build_context_prefix(
        author="Alan Watts",
        chapter_title="full recording",
    )
    assert prefix_case == "[Author: Alan Watts]\n\n"
    assert "full recording" not in prefix_case


def test_build_context_prefix_filters_dummy_strings():
    prefix = build_context_prefix(
        author="Unknown",
        work_title="Untitled",
        chapter_title="None",
    )
    assert prefix == ""


def test_build_context_prefix_handles_empty_and_whitespace():
    assert build_context_prefix(author="   ", work_title="", chapter_title=None) == ""
    assert build_context_prefix() == ""
