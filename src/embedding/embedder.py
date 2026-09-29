"""Эмбеддинг кропа машины в вектор для re-id.

Baseline без дообучения: DINOv2 или CLIP "из коробки" уже дают неплохой
metric-learning сигнал (похожие машины лежат рядом в пространстве).
Дообучение (Triplet + ArcFace, LoRA) добавляем поверх этого же интерфейса
позже, не меняя остальной пайплайн — .encode() должен возвращать то же самое.
"""
from pathlib import Path
from typing import Union

import numpy as np

from src.config import EmbedderConfig


class VehicleEmbedder:
    def __init__(self, cfg: EmbedderConfig | None = None):
        self.cfg = cfg or EmbedderConfig()
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return

        if self.cfg.backend == "dinov2":
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(self.cfg.dinov2_model)
            self._model = AutoModel.from_pretrained(self.cfg.dinov2_model)

            if self.cfg.checkpoint_path is not None and Path(self.cfg.checkpoint_path).exists():
                ckpt = torch.load(self.cfg.checkpoint_path, map_location="cpu", weights_only=False)
                self._model.load_state_dict(ckpt["backbone_state_dict"])

            self._model.to(self.cfg.device).eval()
            self._torch = torch

        elif self.cfg.backend == "clip":
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.cfg.clip_model, pretrained=self.cfg.clip_pretrained
            )
            self._model = model.to(self.cfg.device).eval()
            self._processor = preprocess
            self._torch = torch

        else:
            raise ValueError(f"Неизвестный backend: {self.cfg.backend}")

    def encode(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Возвращает L2-нормированный вектор — так поиск по FAISS сводится
        к простому inner product (косинусная близость)."""
        self._load()
        from PIL import Image

        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = Image.fromarray(image).convert("RGB")

        with self._torch.no_grad():
            if self.cfg.backend == "dinov2":
                inputs = self._processor(images=pil_image, return_tensors="pt").to(self.cfg.device)
                outputs = self._model(**inputs)
                vec = outputs.last_hidden_state[:, 0, :]  # CLS-токен
            else:  # clip
                tensor = self._processor(pil_image).unsqueeze(0).to(self.cfg.device)
                vec = self._model.encode_image(tensor)

        vec = vec.squeeze(0).cpu().numpy().astype("float32")
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode_batch(self, images: list) -> np.ndarray:
        return np.stack([self.encode(img) for img in images])
