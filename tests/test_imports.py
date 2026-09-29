"""Дымовой тест: модули должны импортироваться без тяжёлых зависимостей
(torch/ultralytics/faiss ставятся на сервере с A100, локально их может не
быть) — проверяем, что структура пайплайна не сломана хотя бы на уровне
синтаксиса и путей импорта.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_config_imports():
    from src.config import config

    assert config.top_k > 0
    assert config.embedder.embedding_dim > 0


def test_search_index_dataclass():
    from src.search.index import VehicleRecord

    record = VehicleRecord(record_id=0, camera_id="cam01", timestamp="2026-01-01T00:00:00",
                            image_path="x.jpg")
    assert record.camera_id == "cam01"


def test_attributes_dataclass():
    from src.attributes.attributes import VehicleProfile

    profile = VehicleProfile(color="черный", body_type="седан")
    assert profile.features == []
