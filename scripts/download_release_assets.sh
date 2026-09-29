#!/usr/bin/env bash
# Скачивает обученный чекпоинт эмбеддера и демо-индекс из GitHub Release —
# они не лежат в git (слишком тяжёлые для обычного репозитория), но нужны,
# чтобы API реально отдавал дообученную модель, а не претрейн с нуля.
#
# Запускать из корня репозитория:
#   ./scripts/download_release_assets.sh
set -euo pipefail

RELEASE_URL="https://github.com/GeDeKo/lct2026-vehicle-reid/releases/download/v1.0-model"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT/data/index/checkpoints_v5"

echo "Скачиваю чекпоинт эмбеддера (1.2 ГБ, дообученный DINOv2-Large + ArcFace)..."
curl -L --fail -o "$ROOT/data/index/checkpoints_v5/best.pt" "$RELEASE_URL/checkpoint_best.pt"

echo "Скачиваю демо-индекс (FAISS + метаданные)..."
curl -L --fail -o "$ROOT/data/index/vehicles.faiss" "$RELEASE_URL/vehicles.faiss"
curl -L --fail -o "$ROOT/data/index/vehicles_meta.jsonl" "$RELEASE_URL/vehicles_meta.jsonl"

echo "Скачиваю кропы для превью совпадений..."
curl -L --fail -o /tmp/crops.tar.gz "$RELEASE_URL/crops.tar.gz"
tar -xzf /tmp/crops.tar.gz -C "$ROOT/data/index/"
rm -f /tmp/crops.tar.gz

echo "Готово. Проверка:"
ls -lh "$ROOT/data/index/checkpoints_v5/best.pt" "$ROOT/data/index/vehicles.faiss" "$ROOT/data/index/vehicles_meta.jsonl"
echo "Кропов: $(find "$ROOT/data/index/crops" -name '*.jpg' | wc -l)"
