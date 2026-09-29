"""Датасет для обучения атрибутных голов (марка, тип кузова, цвет) по
реальной разметке из metadata.jsonl, собранной parse_drom.py.

Модель (в узком смысле — "Camry", "Vesta" и т.п.) сюда сознательно не
включена: при 18-25 фото на класс у "тонких" 86 моделей риск переобучения
высокий (см. обсуждение в плане/чате) — марка и кузов агрегируются по
всему датасету и обучаются надёжно, а модель лучше добирать данными
отдельно, когда датасет глубже.
"""
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.camera_augment import camera_ify  # noqa: E402

UNKNOWN = "unknown"


class AttributeDataset(Dataset):
    def __init__(self, root_dir: Path, processor=None, image_size: int = 224, camera_prob: float = 0.3):
        self.root_dir = Path(root_dir)
        self.processor = processor
        self.image_size = image_size
        self.camera_prob = camera_prob

        self.samples: list[dict] = []
        brands, bodies, colors = set(), set(), set()

        for model_dir in sorted(self.root_dir.iterdir()):
            meta_path = model_dir / "metadata.jsonl"
            if not meta_path.exists():
                continue
            with open(meta_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    brand = rec.get("brand") or UNKNOWN
                    body = rec.get("body_type") or UNKNOWN
                    color = rec.get("color") or UNKNOWN
                    brands.add(brand)
                    bodies.add(body)
                    colors.add(color)
                    for rel_photo in rec.get("photos", []):
                        # parse_drom.py пишет путь вида "raw/drom/<brand_model>/<ad_id>/00.jpg"
                        # (относительно data/) — приводим к абсолютному от root_dir (= .../data/raw/drom)
                        photo_path = self.root_dir.parent.parent / rel_photo
                        if photo_path.exists():
                            self.samples.append({"path": photo_path, "brand": brand, "body": body, "color": color})

        if not self.samples:
            raise RuntimeError(f"В {root_dir} не нашлось сэмплов с существующими фото — проверь пути в metadata.jsonl")

        self.brand_vocab = sorted(brands)
        self.body_vocab = sorted(bodies)
        self.color_vocab = sorted(colors)
        self.brand_to_idx = {b: i for i, b in enumerate(self.brand_vocab)}
        self.body_to_idx = {b: i for i, b in enumerate(self.body_vocab)}
        self.color_to_idx = {c: i for i, c in enumerate(self.color_vocab)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("RGB")
        if random.random() < self.camera_prob:
            img = camera_ify(img)
        img = img.resize((self.image_size, self.image_size))

        if self.processor is not None:
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        else:
            import numpy as np

            arr = np.array(img, dtype="float32") / 255.0
            pixel_values = torch.from_numpy(arr).permute(2, 0, 1)

        return (
            pixel_values,
            self.brand_to_idx[sample["brand"]],
            self.body_to_idx[sample["body"]],
            self.color_to_idx[sample["color"]],
        )
