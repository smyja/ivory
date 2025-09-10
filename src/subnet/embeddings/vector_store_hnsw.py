from typing import List, Tuple

import numpy as np

from .vector_store import VectorDBIndex


class HNSWVectorStore(VectorDBIndex):
    def __init__(self, space: str = "cosine", ef_construction: int = 200, M: int = 16) -> None:
        try:
            import hnswlib  # type: ignore
        except Exception as e:
            raise ImportError("hnswlib is not installed") from e
        self._hnswlib = hnswlib
        self._index = None
        self._ids: List[str] = []
        self._space = space
        self._ef_construction = ef_construction
        self._M = M

    def build(self, embeddings: np.ndarray, ids: List[str]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D array")
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ids length must match number of rows in embeddings")
        dim = embeddings.shape[1]
        index = self._hnswlib.Index(space=self._space, dim=dim)
        index.init_index(max_elements=embeddings.shape[0], ef_construction=self._ef_construction, M=self._M)
        index.add_items(embeddings, list(range(embeddings.shape[0])))
        index.set_ef(64)
        self._index = index
        self._ids = list(ids)

    def query(self, vectors: np.ndarray, k: int = 10) -> List[List[Tuple[str, float]]]:
        if self._index is None:
            raise RuntimeError("index is empty")
        labels, dists = self._index.knn_query(vectors, k=k)
        results: List[List[Tuple[str, float]]]= []
        for i in range(labels.shape[0]):
            items: List[Tuple[str, float]] = []
            for j in range(labels.shape[1]):
                idx = int(labels[i, j])
                if idx < 0 or idx >= len(self._ids):
                    continue
                # hnswlib returns distances; for cosine space, distance ~ 1 - sim
                sim = 1.0 - float(dists[i, j])
                items.append((self._ids[idx], sim))
            results.append(items)
        return results

    def persist(self, path: str) -> None:
        import os, json
        if self._index is None:
            raise RuntimeError("nothing to persist")
        os.makedirs(path, exist_ok=True)
        self._index.save_index(os.path.join(path, "index.hnswlib.bin"))
        with open(os.path.join(path, "index.ids.json"), "w") as f:
            json.dump(self._ids, f)

    def load(self, path: str) -> None:
        import os, json
        if self._index is None:
            # Cannot know dim/space here easily; require a build before load in this simple skeleton
            raise RuntimeError("HNSWVectorStore.load() requires an initialized index. Build once before load.")
        self._index.load_index(os.path.join(path, "index.hnswlib.bin"))
        with open(os.path.join(path, "index.ids.json")) as f:
            self._ids = json.load(f)

