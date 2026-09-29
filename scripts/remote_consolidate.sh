#!/usr/bin/env bash
# Запускать НА СЕРВЕРЕ (A100), не локально:
#   ssh -p <port> user@host 'bash -s' < scripts/remote_consolidate.sh
# или скопировать на сервер и запустить там напрямую.
#
# Идемпотентно:
#   1. Собирает все файлы проекта, разбросанные по $HOME, в единый ~/lct2026.
#   2. Распаковывает project/boxcars тарболы, если они ещё не распакованы.
#   3. Печатает итоговое дерево и то, чего не хватает (drom/boxcars),
#      чтобы было видно, что реально доехало, а что нет.
set -euo pipefail

PROJECT_DIR="$HOME/lct2026"
mkdir -p "$PROJECT_DIR"

echo "=== 1. Ищу файлы проекта, разбросанные по \$HOME ==="
# известные артефакты прошлых scp/rsync попыток, которые могли остаться
# прямо в $HOME вместо ~/lct2026
for f in "$HOME"/lct2026_project.tar.gz "$HOME"/boxcars_subset.tar.gz \
         "$HOME"/boxcars116k.zip "$HOME"/BoxCars116k.zip; do
    if [ -f "$f" ] && [ "$(dirname "$f")" != "$PROJECT_DIR" ]; then
        echo "переношу $f -> $PROJECT_DIR/"
        mv -n "$f" "$PROJECT_DIR/"
    fi
done

# любые каталоги верхнего уровня в $HOME, похожие на части проекта, но не
# сам ~/lct2026 (например, если что-то когда-то распаковали не туда)
for d in "$HOME"/drom "$HOME"/boxcars116k "$HOME"/boxcars_subset "$HOME"/data; do
    if [ -d "$d" ] && [ "$d" != "$PROJECT_DIR" ]; then
        echo "ВНИМАНИЕ: найден чужеродный каталог $d — переносить вручную " \
             "(не делаю mv автоматически на всякий случай, проверь содержимое)"
    fi
done

echo "=== 2. Распаковываю тарболы внутри ~/lct2026 (если ещё не распакованы) ==="
cd "$PROJECT_DIR"

if [ -f lct2026_project.tar.gz ] && [ ! -f src/config.py ]; then
    echo "распаковываю lct2026_project.tar.gz"
    tar -xzf lct2026_project.tar.gz
fi

mkdir -p data/processed
if [ -f boxcars_subset.tar.gz ] && [ ! -d data/processed/boxcars_subset ]; then
    echo "распаковываю boxcars_subset.tar.gz -> data/processed/boxcars_subset"
    tar -xzf boxcars_subset.tar.gz -C data/processed/
fi

echo "=== 3. Итоговое состояние ~/lct2026 ==="
echo "-- код --"
[ -f src/config.py ] && echo "OK  src/config.py" || echo "НЕТ src/config.py (код не распакован!)"
[ -f scripts/train_embedder.py ] && echo "OK  scripts/train_embedder.py" || echo "НЕТ scripts/train_embedder.py"

echo "-- датасеты --"
drom_count=$(find data/raw/drom -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
boxcars_count=$(find data/processed/boxcars_subset -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
echo "data/raw/drom: $drom_count фото (локально было 8066 — если меньше, значит Drom доехал не полностью)"
echo "data/processed/boxcars_subset: $boxcars_count фото (локально было 29176)"

echo "-- venv --"
[ -d .venv ] && echo "OK  .venv существует" || echo "НЕТ .venv — нужно поднимать заново"

echo "-- всё, что осталось в \$HOME вне ~/lct2026 (должно быть пусто) --"
find "$HOME" -maxdepth 1 -type f \( -name "*.tar.gz" -o -name "*.zip" \) 2>/dev/null
find "$HOME" -maxdepth 1 -type d ! -path "$HOME" ! -path "$PROJECT_DIR" ! -name ".*" 2>/dev/null

echo "=== ГОТОВО ==="
