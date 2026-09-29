#!/usr/bin/env python3
"""Готовит компактное подмножество BoxCars116k под перенос на сервер.

Полный датасет — 9.1 ГБ, 116k файлов; по нашему медленному SSH-туннелю
это часы. Вместо этого:
  - берём треки (идентичности) минимум с --min-instances фото —
    для re-id метрики важны разные ракурсы одной машины, трек из
    1 фото бесполезен;
  - ограничиваем число фото на трек --max-per-track (без этого немного
    треков с 40+ фото займут непропорционально много места);
  - переупаковываем в структуру, которую понимает VehicleIdentityDataset
    (`root/boxcars/<track_id>/NN.jpg`) — как будто это ещё одна "модель"
    рядом с data/raw/drom/*, чтобы не трогать код датасета;
  - конвертируем PNG → JPEG (quality) — реальный выигрыш в размере.

Запуск:
    python scripts/prepare_boxcars_subset.py --max-tracks 8000
"""
import argparse
import pickle
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BOXCARS_DIR = ROOT / "data/raw/boxcars116k/BoxCars116k"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-instances", type=int, default=3)
    parser.add_argument("--max-per-track", type=int, default=6)
    parser.add_argument("--max-tracks", type=int, default=8000)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/processed/boxcars_subset/boxcars")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(BOXCARS_DIR / "dataset.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin-1")

    samples = [s for s in data["samples"] if len(s["instances"]) >= args.min_instances]
    print(f"Треков с >= {args.min_instances} фото: {len(samples)} из {len(data['samples'])}")

    random.seed(args.seed)
    random.shuffle(samples)
    samples = samples[: args.max_tracks]
    print(f"Беру {len(samples)} треков (--max-tracks)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_images = 0
    total_tracks_written = 0

    for i, sample in enumerate(samples, 1):
        track_id = sample["id"]
        instances = sample["instances"][: args.max_per_track]
        track_dir = args.out_dir / str(track_id)
        track_dir.mkdir(exist_ok=True)

        written = 0
        for j, inst in enumerate(instances):
            src = BOXCARS_DIR / "images" / inst["path"]
            if not src.exists():
                continue
            try:
                img = Image.open(src).convert("RGB")
            except Exception as e:
                print(f"пропускаю {src}: {e}")
                continue
            dst = track_dir / f"{j:02d}.jpg"
            img.save(dst, "JPEG", quality=args.jpeg_quality)
            written += 1

        if written == 0:
            track_dir.rmdir()
            continue

        total_images += written
        total_tracks_written += 1

        if i % 500 == 0 or i == len(samples):
            print(f"{i}/{len(samples)} треков обработано, {total_images} фото сохранено")

    print(f"Готово: {total_tracks_written} треков, {total_images} фото в {args.out_dir}")


if __name__ == "__main__":
    main()
