import os
import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np


class VectorDBIndex(ABC):
    @abstractmethod
    def build(self, embeddings: np.ndarray, ids: List[str]) -> None:
        ...

    @abstractmethod
    def query(self, vectors: np.ndarray, k: int = 10) -> List[List[Tuple[str, float]]]:
        ...

    @abstractmethod
    def persist(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...


def dataset_embedding_dir(dataset_name: str) -> str:
    safe = dataset_name.replace("/", "__").replace("\\", "__")
    d = os.path.join("datasets", safe, "embeddings")
    os.makedirs(d, exist_ok=True)
    return d

