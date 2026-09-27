"""Exact (brute-force) vector index with permission pre-filtering.

At the corpus sizes this project targets (≤ ~200k chunks) an exact dot product over a float32
matrix takes a few milliseconds, and it makes ACL filtering *exact*: we score only the rows a
principal may read. Approximate indexes (HNSW) filter after the graph walk, which can silently
return fewer than k results for users who can see little. The `VectorIndex` interface keeps the
door open for swapping in Qdrant/pgvector with native payload filtering later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorIndex:
    def __init__(self, ids: np.ndarray, matrix: np.ndarray, model_name: str):
        assert len(ids) == len(matrix)
        self.ids = ids.astype(np.int64)
        self.matrix = matrix.astype(np.float32)
        self.model_name = model_name
        self._pos = {int(cid): i for i, cid in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=self.ids, matrix=self.matrix, model_name=np.array(self.model_name))

    @classmethod
    def load(cls, path: Path) -> VectorIndex:
        data = np.load(path, allow_pickle=False)
        return cls(data["ids"], data["matrix"], str(data["model_name"]))

    def search(self, query_vec: np.ndarray, allowed: set[int] | None, limit: int) -> list[tuple[int, float]]:
        if allowed is None:
            rows = np.arange(len(self.ids))
        else:
            rows = np.fromiter((self._pos[c] for c in allowed if c in self._pos), dtype=np.int64)
        if rows.size == 0:
            return []
        scores = self.matrix[rows] @ query_vec.astype(np.float32)
        k = min(limit, rows.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[rows[i]]), float(scores[i])) for i in top]

    def vectors_for(self, chunk_ids: list[int]) -> np.ndarray:
        return self.matrix[[self._pos[c] for c in chunk_ids]]
