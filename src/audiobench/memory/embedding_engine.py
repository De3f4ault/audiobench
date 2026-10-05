"""Embedding Engine — strictly typed entry point for vectorized embeddings.

Enforces task-specific prefix discipline for Nomic text embeddings.

Thread safety
-------------
NomicBert (nomic-ai/nomic-embed-text-v1.5) uses a lazily-extended Rotary
Position Embedding (RoPE) cache.  The extension is **not thread-safe**: if
two threads call ``model.encode()`` simultaneously for the first time at a
new sequence length they race to write the same cache tensor, which corrupts
its dimensions and causes::

    RuntimeError: The size of tensor a (N) must match the size of tensor b (M)
                  at non-singleton dimension 1

All public methods below hold ``_primary_inference_lock`` (an ``RLock``)
around every ``model.encode()`` call so that the sweep thread and the
asyncio-executor query threads never overlap.

Text truncation
---------------
Texts are truncated to ``_MAX_CHARS`` characters before encoding.  The Nomic
model supports up to 8192 tokens (~32 000 chars), but the RoPE cache is
cheapest to extend incrementally.  Capping at a generous 12 000 chars
(≈ 3 000 tokens) keeps all transcription chunks within the expected range
while still allowing very long documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from audiobench.memory.singletons import get_primary_embedder, get_primary_inference_lock

# Conservative character ceiling — prevents unexpectedly long texts from
# triggering RoPE cache extensions during production inference.
_MAX_CHARS = 12_000


def _truncate(text: str) -> str:
    """Truncate text to _MAX_CHARS to bound the token count."""
    return text[:_MAX_CHARS] if len(text) > _MAX_CHARS else text


class EmbeddingEngine:
    """Entry point for generating vector embeddings.

    Enforces the 'search_document:' and 'search_query:' prefix discipline
    required by the nomic-embed-text-v1.5 model.

    All inference calls are serialised via an RLock (see module docstring).
    """

    def embed_for_storage(self, text: str) -> np.ndarray:
        """Embed text intended for the storage layer.

        Args:
            text: The raw text content to embed.

        Returns:
            A 768-dimensional float32 numpy array.
        """
        prefixed_text = f"search_document: {_truncate(text)}"
        model = get_primary_embedder()
        lock = get_primary_inference_lock()
        with lock:
            return model.encode(prefixed_text)

    def embed_for_query(self, text: str) -> np.ndarray:
        """Embed text intended as a search query.

        Args:
            text: The user's query string.

        Returns:
            A 768-dimensional float32 numpy array.
        """
        prefixed_text = f"search_query: {_truncate(text)}"
        model = get_primary_embedder()
        lock = get_primary_inference_lock()
        with lock:
            return model.encode(prefixed_text)

    def embed_batch_for_storage(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Embed a batch of texts for storage in a single forward pass.

        Texts are truncated and sorted by length before encoding so that each
        sub-batch has minimal padding, which is kinder to the RoPE cache.

        Args:
            texts:      Raw text strings (no prefix — added internally).
            batch_size: sentence-transformers sub-batch size (default 64).

        Returns:
            List of 768-dimensional float vectors (as Python lists), in the
            same order as the input *texts*.
        """
        if not texts:
            return []

        # Truncate, then preserve original order via indexed sort
        truncated = [_truncate(t) for t in texts]
        prefixed = [f"search_document: {t}" for t in truncated]

        model = get_primary_embedder()
        lock = get_primary_inference_lock()
        with lock:
            result = model.encode(
                prefixed,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        return result.tolist()
