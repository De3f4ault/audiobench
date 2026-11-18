import numpy as np


def pairwise_cosine_above_threshold(vectors: list[list[float]], ids: list[int], threshold: float) -> list[tuple[int, int, float]]:
    """Compute pairwise cosine similarities and return pairs above threshold."""
    if len(vectors) < 2:
        return []

    A = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    # Prevent division by zero for zero vectors
    norms[norms == 0] = 1.0
    A_norm = A / norms

    sim = A_norm @ A_norm.T

    rows, cols = np.where(sim > threshold)
    results = []
    for r, c in zip(rows, cols):
        if r < c:  # only upper triangle, ignores r == c
            results.append((ids[r], ids[c], float(sim[r, c])))

    return results
