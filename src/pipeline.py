"""Склейка всего пайплайна: кадр -> детекция -> кроп -> эмбеддинг + атрибуты
-> поиск похожих -> ответ.

Это единственное место, которое знает про все компоненты сразу — FastAPI
и CLI-скрипты дергают только PipelineResult, не разбираясь в деталях
детектора/эмбеддера/индекса.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np

from src.attributes.attributes import AttributeTagger, VehicleProfile
from src.config import AppConfig, config as default_config
from src.detection.detector import VehicleDetector
from src.embedding.embedder import VehicleEmbedder
from src.search.index import VehicleIndex, VehicleRecord


@dataclass
class Match:
    record: VehicleRecord
    score: float


@dataclass
class PipelineResult:
    profile: VehicleProfile
    matches: list[Match]
    crop_box: tuple


class VehiclePipeline:
    def __init__(self, cfg: AppConfig | None = None, index: VehicleIndex | None = None):
        self.cfg = cfg or default_config
        self.detector = VehicleDetector(self.cfg.detector)
        self.embedder = VehicleEmbedder(self.cfg.embedder)
        self.tagger = AttributeTagger(device=self.cfg.embedder.device)
        self.index = index  # None, пока не построен / не загружен через build_index.py

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        from PIL import Image

        return np.array(Image.open(image).convert("RGB"))

    def identify(self, image: Union[str, Path, np.ndarray]) -> PipelineResult:
        """Главный сценарий: по фото с одной камеры найти профиль машины
        и похожие записи по остальным камерам."""
        np_image = self._load_image(image)
        detections = self.detector.detect(np_image)
        if not detections:
            raise ValueError("Машина на кадре не найдена")

        # берём самую уверенную детекцию — для прототипа этого достаточно,
        # мультиобъектный кадр можно обрабатывать циклом по detections
        best = max(detections, key=lambda d: d.conf)
        crop = self.detector.crop(np_image, best)

        vector = self.embedder.encode(crop)
        profile = self.tagger.classify(crop)

        matches = []
        if self.index is not None:
            for record, score in self.index.search(vector, top_k=self.cfg.top_k):
                matches.append(Match(record=record, score=score))

        return PipelineResult(profile=profile, matches=matches, crop_box=best.box)

    def enroll(self, image: Union[str, Path, np.ndarray], camera_id: str,
               timestamp: str | None = None, save_crop_to: Path | None = None) -> VehicleRecord:
        """Добавить кадр в индекс — вызывается при построении базы или когда
        приходит новый кадр с камеры в проде."""
        if self.index is None:
            raise RuntimeError("Индекс не инициализирован — сначала build_index.py")

        np_image = self._load_image(image)
        detections = self.detector.detect(np_image)
        if not detections:
            raise ValueError("Машина на кадре не найдена")

        best = max(detections, key=lambda d: d.conf)
        crop = self.detector.crop(np_image, best)
        vector = self.embedder.encode(crop)
        profile = self.tagger.classify(crop)

        image_path = ""
        if save_crop_to is not None:
            from PIL import Image

            save_crop_to.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop).save(save_crop_to)
            image_path = str(save_crop_to)

        record = VehicleRecord(
            record_id=len(self.index._records),
            camera_id=camera_id,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            image_path=image_path,
            attributes=profile.__dict__,
        )
        self.index.add(vector, record)
        return record
