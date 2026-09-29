"""Векторный индекс машин поверх FAISS.

Хранит вектор + метаданные (камера, время, путь к кропу) на каждую
запись. Поиск — косинусная близость (векторы на входе уже L2-нормированы
в VehicleEmbedder, так что IndexFlatIP == cosine similarity).

Для прототипа плоского индекса достаточно — на десятках тысяч записей
FAISS Flat всё ещё быстрый. Если записей станет много (500k+), поменять
IndexFlatIP на IndexIVFFlat/HNSW — интерфейс класса не изменится.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from src.config import IndexConfig


@dataclass
class VehicleRecord:
    record_id: int
    camera_id: str
    timestamp: str  # ISO 8601 — нужен для фильтра "физически возможных" совпадений
    image_path: str
    attributes: dict | None = None


class VehicleIndex:
    def __init__(self, dim: int, cfg: IndexConfig | None = None):
        self.cfg = cfg or IndexConfig()
        self.dim = dim
        self._records: list[VehicleRecord] = []
        self._build_empty()

    def _build_empty(self):
        import faiss

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self.dim)

    def add(self, vector: np.ndarray, record: VehicleRecord):
        self._index.add(vector.reshape(1, -1).astype("float32"))
        self._records.append(record)

    def add_batch(self, vectors: np.ndarray, records: list[VehicleRecord]):
        assert len(vectors) == len(records)
        self._index.add(vectors.astype("float32"))
        self._records.extend(records)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[VehicleRecord, float]]:
        if self._index.ntotal == 0:
            return []
        scores, ids = self._index.search(vector.reshape(1, -1).astype("float32"), top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((self._records[idx], float(score)))
        return results

    def save(self):
        self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self.cfg.index_path))
        with open(self.cfg.meta_path, "w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, dim: int, cfg: IndexConfig | None = None) -> "VehicleIndex":
        import faiss

        cfg = cfg or IndexConfig()
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj.dim = dim
        obj._faiss = faiss
        obj._index = faiss.read_index(str(cfg.index_path))
        obj._records = []
        with open(cfg.meta_path, encoding="utf-8") as f:
            for line in f:
                obj._records.append(VehicleRecord(**json.loads(line)))
        return obj
