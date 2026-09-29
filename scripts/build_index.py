#!/usr/bin/env python3
"""Строит FAISS-индекс из папки с кропами машин.

Ожидаемая структура папки (произвольные имена файлов, camera_id и timestamp
парсятся из имени файла вида <camera_id>__<timestamp>__<любой_суффикс>.jpg,
либо передаются одинаковыми для всех через флаги, если разметки нет):

    data/processed/gallery/
        cam01__2026-09-01T10:00:00__0001.jpg
        cam02__2026-09-01T10:05:00__0002.jpg
        ...

Запуск:
    python scripts/build_index.py --images-dir data/processed/gallery
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.pipeline import VehiclePipeline
from src.search.index import VehicleIndex


def parse_filename(path: Path) -> tuple[str, str]:
    parts = path.stem.split("__")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "unknown", "1970-01-01T00:00:00"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.jpg")
    args = parser.parse_args()

    pipeline = VehiclePipeline(cfg=config)
    pipeline.index = VehicleIndex(dim=config.embedder.embedding_dim, cfg=config.index)

    images = sorted(args.images_dir.glob(args.pattern))
    if not images:
        print(f"В {args.images_dir} не найдено файлов по маске {args.pattern}")
        return

    for i, image_path in enumerate(images, 1):
        camera_id, timestamp = parse_filename(image_path)
        try:
            pipeline.enroll(image_path, camera_id=camera_id, timestamp=timestamp)
        except ValueError as e:
            print(f"[{image_path.name}] пропущено: {e}")
            continue
        if i % 20 == 0 or i == len(images):
            print(f"{i}/{len(images)} обработано")

    pipeline.index.save()
    print(f"Индекс сохранён: {config.index.index_path} ({len(pipeline.index._records)} записей)")


if __name__ == "__main__":
    main()
