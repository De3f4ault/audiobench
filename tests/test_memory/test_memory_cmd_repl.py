"""Tests for REPL slash commands and session state in memory_cmd.py."""

from __future__ import annotations

from audiobench.cli.commands.memory_cmd import SearchSessionState, _parse_slash_command
from audiobench.memory.rrf_fusion import FusedResult


def make_hit(sid: int, text: str = "test fragment", source: str = "test.m4a") -> FusedResult:
    return FusedResult(
        segment_id=sid,
        start_time=0.0,
        end_time=10.0,
        text=text,
        source_file=source,
        rrf_score=0.05,
        stream_contributions=(("fts5", 1),),
    )


def test_search_session_state_defaults():
    state = SearchSessionState()
    assert state.preset == "balanced"
    assert state.max_per_source == 3
    assert not state.pinned_fragments
    assert not state.pinned_expr_ids


def test_parse_slash_command_pin_unpin():
    state = SearchSessionState()
    hits = [make_hit(101, "first"), make_hit(102, "second"), make_hit(103, "third")]

    # Pin index 1 and 3
    msg = _parse_slash_command("/pin 1 3", state, last_sources=hits)
    assert msg is not None
    assert "Pinned 2 fragment(s)" in msg
    assert 101 in state.pinned_fragments
    assert 103 in state.pinned_fragments
    assert 102 not in state.pinned_fragments
    assert state.pinned_expr_ids == {101, 103}

    # Check /pins output
    pins_msg = _parse_slash_command("/pins", state, last_sources=hits)
    assert pins_msg is not None
    assert "Pinned Fragments (2):" in pins_msg
    assert "#101" in pins_msg

    # Unpin index 1
    unpin_msg = _parse_slash_command("/unpin 1", state, last_sources=hits)
    assert unpin_msg is not None
    assert "Unpinned 1 fragment(s)" in unpin_msg
    assert 101 not in state.pinned_fragments
    assert 103 in state.pinned_fragments
    assert state.pinned_expr_ids == {103}

    # Unpin all
    unpin_all_msg = _parse_slash_command("/unpin all", state, last_sources=hits)
    assert unpin_all_msg is not None
    assert "All pins cleared" in unpin_all_msg
    assert len(state.pinned_fragments) == 0
    assert len(state.pinned_expr_ids) == 0


def test_parse_slash_command_set_max_per_source():
    state = SearchSessionState()
    msg = _parse_slash_command("/set max-per-source 5", state)
    assert msg is not None
    assert "Max per source set to 5" in msg
    assert state.max_per_source == 5

    msg_err = _parse_slash_command("/set max-per-source abc", state)
    assert msg_err is not None
    assert "Invalid integer value" in msg_err
    assert state.max_per_source == 5


def test_parse_slash_command_forget_resets_pins_and_limits():
    state = SearchSessionState()
    state.max_per_source = 10
    state.pinned_fragments[999] = make_hit(999)
    state.pinned_expr_ids.add(999)

    msg = _parse_slash_command("/forget", state)
    assert msg is not None
    assert "reset to defaults" in msg
    assert state.max_per_source == 3
    assert len(state.pinned_fragments) == 0
    assert len(state.pinned_expr_ids) == 0
