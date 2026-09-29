#!/usr/bin/env python3
"""Считает mAP и CMC (Rank-1, Rank-5) на re-id датасете формата VeRi-776:

    dataset/
        query/    <vehicle_id>_c<camera_id>_<frame>.jpg
        gallery/  <vehicle_id>_c<camera_id>_<frame>.jpg

vehicle_id одинаковый у одной и той же машины на разных камерах — это и
есть ground truth для расчёта метрик. Используем эти метрики и на своих
данных, если разметку удаётся получить (например, из парсера Drom/Avito,
где несколько фото — это один и тот же vehicle_id).

Запуск:
    python scripts/eval_reid.py --dataset-dir data/raw/veri776
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.embedding.embedder import VehicleEmbedder
from src.embedding.reid_eval import compute_metrics


def parse_id_cam(path: Path) -> tuple[str, str]:
    # VeRi-776: 0002_c002_00030600_0.jpg -> vehicle_id=0002, camera_id=c002
    parts = path.stem.split("_")
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    embedder = VehicleEmbedder(config.embedder)

    def encode_split(split_dir: Path):
        vecs, ids, cams = [], [], []
        paths = sorted((args.dataset_dir / split_dir).glob("*.jpg"))
        for i, path in enumerate(paths, 1):
            vid, cam = parse_id_cam(path)
            vecs.append(embedder.encode(path))
            ids.append(vid)
            cams.append(cam)
            if i % 50 == 0:
                print(f"{split_dir}: {i}/{len(paths)}")
        return np.stack(vecs), np.array(ids), np.array(cams)

    print("Кодирую query...")
    q_vecs, q_ids, q_cams = encode_split(Path("query"))
    print("Кодирую gallery...")
    g_vecs, g_ids, g_cams = encode_split(Path("gallery"))

    # та же камера + тот же vehicle_id — не засчитываем (это тот же трек, не re-id)
    exclude = (g_ids[None, :] == q_ids[:, None]) & (g_cams[None, :] == q_cams[:, None])
    metrics = compute_metrics(q_vecs, q_ids, g_vecs, g_ids, exclude=exclude, top_k=args.top_k)
    print("\n== Результаты ==")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
