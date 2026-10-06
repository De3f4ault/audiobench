"""Unit and integration tests for audiobench.memory.graph_linker."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from audiobench.memory.graph_linker import (
    IngestLinkResult,
    drain_pending_links,
    enqueue_for_linking,
    get_pending_queue_size,
    is_linkable_chunk,
    link_expressions_batch,
    seed_unlinked_islands,
)


class TestIsLinkableChunk:
    def test_rejects_empty_and_short_strings(self) -> None:
        assert is_linkable_chunk("") is False
        assert is_linkable_chunk("   ") is False
        assert is_linkable_chunk("Chapter IV.") is False
        assert is_linkable_chunk("Part Two: The Return.") is False
        assert is_linkable_chunk("Short sentence under sixty characters length here.") is False

    def test_rejects_low_alphabetic_density(self) -> None:
        # 60+ chars but mostly numbers/punctuation/timestamps
        timestamps = "[00:12:45] --> [00:12:50] 1234567890 9876543210 #@!$%^&*()_+~"
        assert len(timestamps) >= 60
        assert is_linkable_chunk(timestamps) is False

    def test_rejects_boilerplate_disclaimers(self) -> None:
        librivox = (
            "This is a LibriVox recording. All LibriVox recordings are in the public domain. "
            "For more information or to volunteer, please visit librivox.org."
        )
        assert len(librivox) >= 60
        assert is_linkable_chunk(librivox) is False

        gutenberg = (
            "This eBook is for the use of anyone anywhere in the United States and most other parts of the world "
            "at no cost and with almost no restrictions whatsoever. Project Gutenberg."
        )
        assert is_linkable_chunk(gutenberg) is False

    def test_accepts_substantive_literary_content(self) -> None:
        dostoyevsky = (
            "If he is not killed and recovers it will be first rate, for then he will bear "
            "witness that Dmitri Fyodorovitch was not the murderer and Smerdyakov alone committed the crime."
        )
        assert len(dostoyevsky) >= 60
        assert is_linkable_chunk(dostoyevsky) is True

        frankl = (
            "Everything can be taken from a man but one thing: the last of the human freedoms—"
            "to choose one's attitude in any given set of circumstances, to choose one's own way."
        )
        assert len(frankl) >= 60
        assert is_linkable_chunk(frankl) is True


class TestQueueAndDrain:
    def test_enqueue_and_queue_size(self) -> None:
        initial_size = get_pending_queue_size()
        enqueued = enqueue_for_linking([900001, 900002, 900003])
        assert enqueued == 3
        assert get_pending_queue_size() == initial_size + 3

        # Drain them to restore queue
        drained = drain_pending_links(batch_size=100)
        # We don't assert edges because 900001 doesn't exist in DB, but chunks_processed should match
        assert drained.chunks_processed >= 3

    def test_queue_overflow_handling(self) -> None:
        with patch("audiobench.memory.graph_linker._pending_queue") as mock_q:
            mock_q.put_nowait.side_effect = queue.Full
            enqueued = enqueue_for_linking([1, 2, 3])
            assert enqueued == 0


class TestLinkExpressionsBatchMocked:
    def test_empty_input_returns_zero(self) -> None:
        result = link_expressions_batch([])
        assert result.chunks_processed == 0
        assert result.edges_created == 0
        assert result.linked_work_ids == []

    def test_work_isolation_and_threshold(self) -> None:
        """Verify candidates in the same work_id are excluded and threshold is respected."""
        mock_index = MagicMock()
        mock_index.has_item.return_value = True
        mock_index.get_items.return_value = np.zeros((1, 768), dtype=np.float32)
        # Return 3 candidates:
        # cand 102: same work (wid=1), dist=0.1 (sim 0.90) -> MUST BE EXCLUDED
        # cand 103: diff work (wid=2), dist=0.3 (sim 0.70 < 0.78) -> MUST BE EXCLUDED
        # cand 104: diff work (wid=3), dist=0.15 (sim 0.85 >= 0.78) -> MUST BE LINKED
        mock_index.query_batch.return_value = (
            np.array([[102, 103, 104]]),
            np.array([[0.10, 0.30, 0.15]]),
        )

        sample_rows_src = [(101, 1, "A very deep and meaningful contemplation on human existence and mortality across ages.")]
        sample_rows_cand = [
            (102, 1, "Another passage from the exact same book that should never be linked as cross-work."),
            (103, 2, "A passage from a second book that has insufficient similarity for a semantic edge."),
            (104, 3, "A profound passage from a third book with high conceptual resonance and alignment."),
        ]

        with patch("audiobench.memory.graph_linker.get_session") as mock_session_ctx:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session

            # First query fetches source, second fetches candidates
            mock_session.execute.side_effect = [
                MagicMock(fetchall=lambda: sample_rows_src),
                MagicMock(fetchall=lambda: sample_rows_cand),
                MagicMock(),  # INSERT statement
            ]

            result = link_expressions_batch(
                [101],
                max_edges_per_chunk=2,
                threshold=0.78,
                index=mock_index,
            )

            assert result.chunks_processed == 1
            assert result.edges_created == 1
            assert 1 in result.linked_work_ids
            assert 3 in result.linked_work_ids
            assert 2 not in result.linked_work_ids


class TestSeedUnlinkedIslands:
    def test_seed_unlinked_islands_calls_linking(self) -> None:
        with patch("audiobench.memory.graph_linker.get_session") as mock_session_ctx, \
             patch("audiobench.memory.graph_linker.link_expressions_batch") as mock_batch:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value.fetchall.return_value = [(201,), (202,)]
            mock_batch.return_value = IngestLinkResult(2, 4, 12.5, [10, 20])

            res = seed_unlinked_islands(limit=10)
            assert res.chunks_processed == 2
            assert res.edges_created == 4
            mock_batch.assert_called_once()


class TestGraphSeedCommand:
    def test_graph_seed_dispatch(self) -> None:
        from audiobench.cli.repl.backslash_commands import cmd_graph
        mock_session = MagicMock()
        with patch("audiobench.memory.graph_linker.seed_unlinked_islands") as mock_seed, \
             patch("audiobench.cli.repl.backslash_commands.console.print") as mock_print:
            mock_seed.return_value = IngestLinkResult(50, 80, 45.2, [1, 2])
            cmd_graph(mock_session, "seed 500")
            mock_seed.assert_called_once_with(limit=500)

