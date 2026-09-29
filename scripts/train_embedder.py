#!/usr/bin/env python3
"""Дообучение эмбеддера (DINOv2 + ArcFace) на объединённом датасете
(Drom + BoxCars и любые другие источники той же структуры).

Запуск на сервере с A100:
    source .venv/bin/activate
    python scripts/train_embedder.py --epochs 15 --batch-size 64

Быстрая проверка, что пайплайн вообще собирается и крутится (данные
подтягиваются, backbone+голова считаются, чекпоинт пишется, валидация
считает метрики) без реального обучения:
    python scripts/train_embedder.py --dry-run

Логирует loss/accuracy каждые --log-every шагов в stdout (удобно смотреть
через `tail -f` или Monitor), раз в эпоху считает re-id метрики (mAP,
Rank-1, Rank-5) на отложенных identity и сохраняет чекпоинты каждую эпоху
в data/index/checkpoints/, плюс "best" по val mAP (не по train accuracy —
train accuracy растёт даже при переобучении и не значит ничего для
реального re-id качества).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.arcface import ArcFaceHead
from src.embedding.reid_dataset import VehicleIdentityDataset, collect_identities, split_identities
from src.embedding.reid_eval import compute_metrics


@torch.no_grad()
def embed_paths(backbone, processor, paths: list[Path], device: str, batch_size: int = 32) -> np.ndarray:
    """Прогоняет список путей к фото через backbone (без аугментаций камеры —
    валидация должна быть на чистых данных) и возвращает L2-нормированные
    эмбеддинги."""
    from PIL import Image

    backbone.eval()
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        out = backbone(**inputs).last_hidden_state[:, 0, :]
        out = out.cpu().numpy().astype("float32")
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.where(norms > 0, norms, 1.0)
        vecs.append(out)
    return np.concatenate(vecs, axis=0)


def build_val_query_gallery(val_identities: list[list[Path]]):
    """1 фото identity -> query, остальные фото той же identity -> gallery.
    Query и gallery не пересекаются по фото, поэтому в compute_metrics не
    нужна маска exclude (в отличие от eval_reid.py на VeRi-776, где query и
    gallery могут содержать один и тот же трек с той же камеры)."""
    query_paths, query_ids, gallery_paths, gallery_ids = [], [], [], []
    for identity_idx, photos in enumerate(val_identities):
        query_paths.append(photos[0])
        query_ids.append(identity_idx)
        for p in photos[1:]:
            gallery_paths.append(p)
            gallery_ids.append(identity_idx)
    return query_paths, np.array(query_ids), gallery_paths, np.array(gallery_ids)


def validate(backbone, processor, val_identities: list[list[Path]], device: str) -> dict:
    query_paths, query_ids, gallery_paths, gallery_ids = build_val_query_gallery(val_identities)
    query_vecs = embed_paths(backbone, processor, query_paths, device)
    gallery_vecs = embed_paths(backbone, processor, gallery_paths, device)
    backbone.train()
    top_k = max(1, min(5, len(gallery_paths)))
    return compute_metrics(query_vecs, query_ids, gallery_vecs, gallery_ids, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, nargs="+",
                         default=[Path("data/raw/drom"), Path("data/processed/boxcars_subset")],
                         help="один или несколько каталогов с identity-папками; "
                              "объединяются в один датасет классов для ArcFace")
    parser.add_argument("--backbone", default="facebook/dinov2-base")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--arcface-margin", type=float, default=0.30,
                         help="при холодном старте с большим числом классов margin мешает сходимости "
                              "(логиты верного класса ещё не разделены — margin делает их только хуже); "
                              "для быстрого запуска с нуля разумно 0.05-0.1, дотюнивать выше по мере сходимости")
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--camera-prob", type=float, default=0.4)
    parser.add_argument("--min-photos", type=int, default=2,
                         help="identity с меньшим числом фото выбрасываются; для val нужно "
                              "минимум 2 (1 в query + 1+ в gallery)")
    parser.add_argument("--val-split", type=float, default=0.1,
                         help="доля IDENTITY (не фото!), уходящих в val — режется по identity, "
                              "чтобы val проверял перенос на невиданные машины, а не память о ракурсе")
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None,
                         help="ограничить число шагов оптимизатора за эпоху (для --dry-run/отладки)")
    parser.add_argument("--dry-run", action="store_true",
                         help="epochs=1, max-steps=5 (если не заданы явно), чекпоинты в "
                              "data/index/checkpoints/dry_run/ — быстро проверить, что пайплайн "
                              "целиком собирается и крутится, без реального обучения")
    args = parser.parse_args()

    if args.dry_run:
        if "--epochs" not in sys.argv:
            args.epochs = 1
        if args.max_steps is None:
            args.max_steps = 5
        if args.out_dir is None:
            args.out_dir = Path("data/index/checkpoints/dry_run")
    if args.out_dir is None:
        args.out_dir = Path("data/index/checkpoints")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cpu":
        print("ВНИМАНИЕ: CUDA недоступна, обучение будет очень медленным", flush=True)

    from transformers import AutoImageProcessor, AutoModel

    print(f"Загружаю бэкбон {args.backbone}...", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.backbone)
    backbone = AutoModel.from_pretrained(args.backbone).to(device)
    embedding_dim = backbone.config.hidden_size

    print(f"Сканирую датасеты: {[str(d) for d in args.data_dir]}", flush=True)
    all_identities = collect_identities(args.data_dir, min_photos=args.min_photos)
    train_identities, val_identities = split_identities(
        all_identities, val_fraction=args.val_split, seed=args.val_seed)
    print(f"Identity всего: {len(all_identities)} -> train {len(train_identities)}, "
          f"val {len(val_identities)} (val держится полностью вне ArcFace-классов, "
          f"это невиданные на train машины)", flush=True)

    dataset = VehicleIdentityDataset(identities=train_identities, processor=processor,
                                      camera_prob=args.camera_prob)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, drop_last=True, pin_memory=(device == "cuda"))

    head = ArcFaceHead(embedding_dim, dataset.num_classes,
                        scale=args.arcface_scale, margin=args.arcface_margin).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ])
    steps_per_epoch = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * steps_per_epoch))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0
    global_step = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for step, (pixel_values, labels) in enumerate(loader, 1):
            if args.max_steps and step > args.max_steps:
                break

            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = backbone(pixel_values=pixel_values)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS-токен

            logits = head(embeddings, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            global_step += 1

            if step % args.log_every == 0 or (args.max_steps and step == args.max_steps):
                elapsed = time.time() - start_time
                acc = running_correct / max(1, running_total)
                avg_loss = running_loss / max(1, min(step, args.log_every))
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"epoch={epoch} step={step}/{steps_per_epoch} global_step={global_step} "
                      f"loss={avg_loss:.4f} train_acc(arcface_proxy)={acc:.4f} lr={lr_now:.2e} "
                      f"elapsed={elapsed:.0f}s", flush=True)
                running_loss, running_correct, running_total = 0.0, 0, 0

        print(f"Считаю re-id метрики на val ({len(val_identities)} identity)...", flush=True)
        val_metrics = validate(backbone, processor, val_identities, device)
        rank_k_key = next(k for k in val_metrics if k.startswith("Rank-") and k != "Rank-1")
        print(f"=== EPOCH {epoch} VAL: mAP={val_metrics['mAP']:.4f} "
              f"Rank-1={val_metrics['Rank-1']:.4f} "
              f"{rank_k_key}={val_metrics[rank_k_key]:.4f} ===", flush=True)

        ckpt_path = args.out_dir / f"epoch{epoch:03d}.pt"
        torch.save({
            "backbone_state_dict": backbone.state_dict(),
            "head_state_dict": head.state_dict(),
            "epoch": epoch,
            "embedding_dim": embedding_dim,
            "num_classes": dataset.num_classes,
            "backbone_name": args.backbone,
            "val_metrics": val_metrics,
        }, ckpt_path)
        print(f"=== EPOCH {epoch} DONE, чекпоинт сохранён: {ckpt_path} ===", flush=True)

        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            best_path = args.out_dir / "best.pt"
            torch.save(torch.load(ckpt_path), best_path)
            print(f"=== новый лучший чекпоинт (val mAP={best_map:.4f}): {best_path} ===", flush=True)

    print("DRY_RUN_OK" if args.dry_run else "=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===", flush=True)


if __name__ == "__main__":
    main()
