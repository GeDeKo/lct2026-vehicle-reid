"""Детекция автомобилей на кадре камеры.

Baseline: предобученный YOLO (COCO), без дообучения — для машин этого
обычно достаточно, чтобы вырезать чистый кроп под эмбеддинг. Дообучение
имеет смысл только если камеры дают нестандартные ракурсы (сверху, ночь)
и baseline промахивается.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from src.config import DetectorConfig


@dataclass
class Detection:
    box: tuple  # (x1, y1, x2, y2) в пикселях исходного кадра
    conf: float
    cls_id: int


class VehicleDetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self._model = None  # ленивая загрузка — тяжёлые веса не тянем при импорте

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # импорт внутри, чтобы модуль был лёгким без torch

            self._model = YOLO(self.cfg.weights)
            self._model.to(self.cfg.device)

    def detect(self, image: Union[str, Path, np.ndarray]) -> list[Detection]:
        """Возвращает список детекций машин/автобусов/грузовиков на кадре."""
        self._load()
        results = self._model.predict(
            source=image,
            conf=self.cfg.conf_threshold,
            classes=list(self.cfg.classes),
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        box=tuple(map(float, xyxy)),
                        conf=float(box.conf[0]),
                        cls_id=int(box.cls[0]),
                    )
                )
        return detections

    @staticmethod
    def crop(image: np.ndarray, detection: Detection, pad: float = 0.05) -> np.ndarray:
        """Кроп машины с небольшим паддингом — так эмбеддер видит края кузова."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = detection.box
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * pad))
        y1 = max(0, int(y1 - bh * pad))
        x2 = min(w, int(x2 + bw * pad))
        y2 = min(h, int(y2 + bh * pad))
        return image[y1:y2, x1:x2]
