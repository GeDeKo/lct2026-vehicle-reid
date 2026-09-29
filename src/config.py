"""Общий конфиг проекта. Пути и гиперпараметры собраны в одном месте,
чтобы не расползались по коду — на ноутбуке без GPU DEVICE переключается
переменной окружения `export DEVICE=cpu`, на сервере с A100 по умолчанию
уже "cuda".
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

DEFAULT_DEVICE = os.environ.get("DEVICE", "cuda")


@dataclass
class DetectorConfig:
    weights: str = "yolov8n.pt"       # авто-скачается ultralytics при первом запуске
    conf_threshold: float = 0.35
    classes: tuple = (2, 5, 7)        # car, bus, truck (индексы COCO)
    device: str = DEFAULT_DEVICE


@dataclass
class EmbedderConfig:
    backend: str = "dinov2"           # "dinov2" | "clip"
    dinov2_model: str = "facebook/dinov2-large"
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    embedding_dim: int = 1024
    device: str = DEFAULT_DEVICE
    # чекпоинт из scripts/train_embedder.py (дообученный backbone_state_dict);
    # None -> используется чистый претрейн с HuggingFace без дообучения
    checkpoint_path: Path | None = INDEX_DIR / "checkpoints_v5" / "best.pt"


@dataclass
class IndexConfig:
    index_path: Path = INDEX_DIR / "vehicles.faiss"
    meta_path: Path = INDEX_DIR / "vehicles_meta.jsonl"
    metric: str = "cosine"            # реализовано через нормализацию + inner product


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    top_k: int = 10


config = AppConfig()
