# ЛЦТ 2026 — Vehicle Re-ID

Сервис идентификации автомобиля по фото с городских камер без опоры на
госномер: детекция → эмбеддинг → векторный поиск похожих + атрибуты
(марка, модель, кузов, цвет, особенности).

Полный план, обоснование выбора моделей, датасетов и рисков —
в [LCT_2026_ReID_plan.md](./LCT_2026_ReID_plan.md).

## Структура

```
src/
  config.py           # все пути и гиперпараметры в одном месте
  detection/           # YOLO — находит машину на кадре
  embedding/           # DINOv2 / CLIP — превращает кроп в вектор
  attributes/           # марка/модель/цвет/кузов + zero-shot "особенности"
  search/               # FAISS индекс: добавление, поиск, save/load
  pipeline.py           # склейка всего в сценарий identify()/enroll()
  api/main.py           # FastAPI: POST /identify, POST /enroll, GET /health
scripts/
  parse_drom.py         # сбор датасета с auto.drom.ru (атрибуты + пары фото)
  download_datasets.md  # где взять готовые re-id датасеты (VeRi-776, BoxCars116k и др.)
  camera_augment.py     # имитация городской камеры поверх фото объявлений (домен-гэп)
  build_index.py        # построить FAISS-индекс из папки с кропами
  eval_reid.py           # посчитать mAP / CMC Rank-1 / Rank-5
tests/
  test_imports.py        # дымовой тест, без тяжёлых ML-зависимостей
```

## Быстрый старт

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # на ноутбуке без GPU часть ML-пакетов можно пропустить —
                                       # API и парсер работают и без torch/ultralytics/faiss

# 1. собрать датасет для атрибутов (не требует GPU)
python scripts/parse_drom.py --brand toyota --model camry --max-ads 100

# 2. поднять сервис (нужен torch/ultralytics/faiss — ставить на сервере с A100)
uvicorn src.api.main:app --reload --port 8000

# 3. построить индекс из готовых кропов и прогнать оценку качества
python scripts/build_index.py --images-dir data/processed/gallery
python scripts/eval_reid.py --dataset-dir data/raw/veri776
```

## Статус

- [x] Каркас пайплайна: детекция → эмбеддинг → поиск → атрибуты (baseline, без дообучения)
- [x] Парсер auto.drom.ru — проверен вживую, отдаёт марку/модель/цвет/кузов/поколение + фото
- [x] Датасет собран: **96 моделей, 2420 объявлений, 8066 фото, 1.5 ГБ** (`data/raw/drom/`), включая китайские бренды (Chery, Geely, Haval, Changan, Omoda, JAC, GAC, Tank, Voyah, Zeekr, BYD и др.) — список моделей и как дособрать ещё см. `scripts/download_datasets.md`
- [x] FastAPI-сервис (`/identify`, `/enroll`, `/health`)
- [x] Скрипт метрик re-id (mAP, CMC Rank-1/Rank-5) под формат VeRi-776
- [x] `camera_augment.py` — имитация городской камеры (blur, шум, JPEG, ночь, перспектива) поверх Drom-фото, проверено вживую — частично закрывает разрыв доменов для re-id
- [ ] **BoxCars116k** (116k фото, 27k размеченных треков одной машины, реальные CCTV-камеры, регистрация не нужна) — найден рабочий бэкап на Google Drive (6 ГБ), не скачан: на диске всего 11 ГБ свободно. Ждём, когда освободите место, — см. `scripts/download_datasets.md`
- [ ] Доступ к серверу с A100 — заблокирован на сетевом уровне (см. переписку с девопсом), обучение и тяжёлые модели (torch/ultralytics/faiss) пока не запускались
- [ ] Дообучение эмбеддера (Triplet + ArcFace) — после доступа к GPU
- [ ] Веб-демо (фронтенд)
- [ ] Уточнить у организаторов: метрику оценки, датасет от них, сроки и формат сдачи (см. раздел 6 плана)

### Датасет: как дособрать ещё

`scripts/parse_drom.py` идемпотентный — повторный запуск с тем же brand/model
не трогает уже собранное, добирает только новое:

```bash
python scripts/parse_drom.py --brand toyota --model camry --max-ads 50
```

Список из 96 уже собранных пар brand/model — в `data/raw/drom/*/metadata.jsonl`
(по одной папке на модель). Слаги для новых марок/моделей стоит сверять
скриптом `scripts/_verify_models.py` перед большим прогоном — auto.drom.ru
использует не всегда очевидные URL (`bmw/3-series`, `chery/tiggo_7_pro`,
`mercedes-benz/e-class` и т.п.).
