import os
import json
from typing import List, Tuple

import numpy as np

from .vector_store import VectorDBIndex


class NumpyVectorStore(VectorDBIndex):
    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None
        self.ids: List[str] = []

    def build(self, embeddings: np.ndarray, ids: List[str]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D array")
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ids length must match number of rows in embeddings")
        # Store L2-normalized for cosine via dot
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        self.matrix = embeddings / norms
        self.ids = list(ids)

    def query(self, vectors: np.ndarray, k: int = 10) -> List[List[Tuple[str, float]]]:
        if self.matrix is None:
            raise RuntimeError("index is empty")
        # Normalize queries
        q = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        sims = q @ self.matrix.T  # cosine similarities
        results: List[List[Tuple[str, float]]] = []
        for i in range(sims.shape[0]):
            idx = np.argpartition(-sims[i], kth=min(k, sims.shape[1] - 1))[:k]
            order = idx[np.argsort(-sims[i][idx])]
            results.append([(self.ids[j], float(sims[i][j])) for j in order])
        return results

    def persist(self, path: str) -> None:
        if self.matrix is None:
            raise RuntimeError("nothing to persist")
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "index.matrix.npy"), self.matrix)
        with open(os.path.join(path, "index.ids.json"), "w") as f:
            json.dump(self.ids, f)

    def load(self, path: str) -> None:
        self.matrix = np.load(os.path.join(path, "index.matrix.npy"))
        with open(os.path.join(path, "index.ids.json")) as f:
            self.ids = json.load(f)

