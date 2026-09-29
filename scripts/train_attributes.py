#!/usr/bin/env python3
"""Обучение атрибутных голов (марка, тип кузова, цвет) поверх бэкбона.

Мультизадачно: один бэкбон (тот же DINOv2, что и для re-id — логично
шарить его, чтобы не тащить две модели в проде) + три линейные головы.
Метки "unknown" в лоссе не участвуют (маскируются), чтобы не учить
модель предсказывать "хз" как отдельный полноценный класс там, где
парсер просто не нашёл значение в объявлении.

Запуск на сервере:
    source .venv/bin/activate
    python scripts/train_attributes.py --epochs 10 --batch-size 64
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attributes.train_dataset import AttributeDataset, UNKNOWN


class AttributeModel(nn.Module):
    def __init__(self, backbone, embedding_dim: int, n_brand: int, n_body: int, n_color: int):
        super().__init__()
        self.backbone = backbone
        self.brand_head = nn.Linear(embedding_dim, n_brand)
        self.body_head = nn.Linear(embedding_dim, n_body)
        self.color_head = nn.Linear(embedding_dim, n_color)

    def forward(self, pixel_values):
        emb = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        return self.brand_head(emb), self.body_head(emb), self.color_head(emb)


def masked_ce(logits, labels, unknown_idx: int, criterion: nn.CrossEntropyLoss) -> torch.Tensor:
    mask = labels != unknown_idx
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return criterion(logits[mask], labels[mask])


def accuracy(logits, labels, unknown_idx: int) -> tuple[int, int]:
    mask = labels != unknown_idx
    if mask.sum() == 0:
        return 0, 0
    preds = logits[mask].argmax(dim=1)
    return (preds == labels[mask]).sum().item(), mask.sum().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/drom"))
    parser.add_argument("--backbone", default="facebook/dinov2-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("data/index/checkpoints"))
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    from transformers import AutoImageProcessor, AutoModel

    print(f"Загружаю бэкбон {args.backbone}...", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.backbone)
    backbone = AutoModel.from_pretrained(args.backbone)
    embedding_dim = backbone.config.hidden_size

    dataset = AttributeDataset(args.data_dir, processor=processor)
    print(f"Сэмплов: {len(dataset)}, брендов={len(dataset.brand_vocab)}, "
          f"кузовов={len(dataset.body_vocab)}, цветов={len(dataset.color_vocab)}", flush=True)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    model = AttributeModel(backbone, embedding_dim, len(dataset.brand_vocab),
                            len(dataset.body_vocab), len(dataset.color_vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    brand_unk = dataset.brand_to_idx.get(UNKNOWN, -1)
    body_unk = dataset.body_to_idx.get(UNKNOWN, -1)
    color_unk = dataset.color_to_idx.get(UNKNOWN, -1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # словари меток нужны на инференсе, чтобы расшифровать индекс обратно в слово
    vocabs_path = args.out_dir / "attribute_vocabs.json"
    with open(vocabs_path, "w", encoding="utf-8") as f:
        json.dump({"brand": dataset.brand_vocab, "body": dataset.body_vocab, "color": dataset.color_vocab}, f,
                   ensure_ascii=False, indent=2)
    print(f"Словари меток сохранены: {vocabs_path}", flush=True)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, step_count = 0.0, 0

        for step, (pixel_values, brand_l, body_l, color_l) in enumerate(train_loader, 1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            brand_l, body_l, color_l = brand_l.to(device), body_l.to(device), color_l.to(device)

            brand_logits, body_logits, color_logits = model(pixel_values)
            loss = (masked_ce(brand_logits, brand_l, brand_unk, criterion)
                    + masked_ce(body_logits, body_l, body_unk, criterion)
                    + masked_ce(color_logits, color_l, color_unk, criterion))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step_count += 1

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(f"epoch={epoch} step={step}/{len(train_loader)} "
                      f"loss={running_loss / step_count:.4f} elapsed={elapsed:.0f}s", flush=True)

        # валидация раз в эпоху
        model.eval()
        correct = {"brand": 0, "body": 0, "color": 0}
        total = {"brand": 0, "body": 0, "color": 0}
        with torch.no_grad():
            for pixel_values, brand_l, body_l, color_l in val_loader:
                pixel_values = pixel_values.to(device)
                brand_l, body_l, color_l = brand_l.to(device), body_l.to(device), color_l.to(device)
                brand_logits, body_logits, color_logits = model(pixel_values)
                c, t = accuracy(brand_logits, brand_l, brand_unk); correct["brand"] += c; total["brand"] += t
                c, t = accuracy(body_logits, body_l, body_unk); correct["body"] += c; total["body"] += t
                c, t = accuracy(color_logits, color_l, color_unk); correct["color"] += c; total["color"] += t

        accs = {k: correct[k] / max(1, total[k]) for k in correct}
        mean_acc = sum(accs.values()) / len(accs)
        print(f"=== EPOCH {epoch} VAL: brand_acc={accs['brand']:.4f} body_acc={accs['body']:.4f} "
              f"color_acc={accs['color']:.4f} mean={mean_acc:.4f} ===", flush=True)

        ckpt_path = args.out_dir / f"attributes_epoch{epoch:03d}.pt"
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                    "embedding_dim": embedding_dim, "backbone_name": args.backbone,
                    "n_brand": len(dataset.brand_vocab), "n_body": len(dataset.body_vocab),
                    "n_color": len(dataset.color_vocab)}, ckpt_path)

        if mean_acc > best_val_acc:
            best_val_acc = mean_acc
            torch.save(torch.load(ckpt_path), args.out_dir / "attributes_best.pt")
            print(f"=== новый лучший чекпоинт атрибутов (mean_acc={best_val_acc:.4f}) ===", flush=True)

    print("=== ОБУЧЕНИЕ АТРИБУТОВ ЗАВЕРШЕНО ===", flush=True)


if __name__ == "__main__":
    main()
