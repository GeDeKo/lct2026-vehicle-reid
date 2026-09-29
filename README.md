# ЛЦТ 2026 — Vehicle Re-ID

Сервис идентификации автомобиля по фото с городской камеры без опоры на
госномер: детекция → дообученный эмбеддинг → векторный поиск похожих +
атрибуты (цвет, кузов, особенности). Веб-интерфейс на React/TypeScript,
бэкенд на FastAPI.

**Итоговое качество (честный val-сплит, невиданные на train машины):
mAP 0.759, Rank-1 0.793, Rank-5 0.915.**

Полная техническая документация (архитектура, обоснование backbone,
история обучения, честный анализ ошибок, API, инструкция по
развёртыванию) — в [DOCUMENTATION.md](./DOCUMENTATION.md).
Изначальный план команды — в [LCT_2026_ReID_plan.md](./LCT_2026_ReID_plan.md).
Презентация — в [presentation/](./presentation/).

## Быстрый старт

Подробно, с полной инструкцией по установке зависимостей и двумя
способами запуска (у себя на GPU или через наш сервер) — в
[DOCUMENTATION.md → «Как развернуть»](./DOCUMENTATION.md#как-развернуть).
Коротко:

```bash
git clone https://github.com/GeDeKo/lct2026-vehicle-reid.git
cd lct2026-vehicle-reid
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

./scripts/download_release_assets.sh   # обученный чекпоинт + демо-индекс из GitHub Release
uvicorn src.api.main:app --host 0.0.0.0 --port 8800

# в отдельном терминале
cd web && npm install && npm run build && npx vite preview --port 5500
```

Открыть `http://localhost:5500`.

## Структура

```
src/
  config.py           # все пути и гиперпараметры в одном месте
  detection/           # YOLO — находит машину на кадре
  embedding/           # DINOv2-Large + ArcFace — дообученный эмбеддер
  attributes/           # zero-shot CLIP: цвет/кузов/особенности
  search/               # FAISS индекс: добавление, поиск, save/load
  pipeline.py           # склейка всего в сценарий identify()/enroll()
  api/main.py           # FastAPI: POST /identify, POST /enroll, GET /health
scripts/
  parse_drom.py               # сбор датасета с auto.drom.ru
  prepare_boxcars_subset.py   # подготовка BoxCars116k (реальные камеры)
  camera_augment.py           # имитация городской камеры (домен-гэп)
  train_embedder.py           # дообучение ArcFace, honest val-сплит по identity
  train_attributes.py         # головы марка/кузов/цвет (обучены, пока не в проде)
  build_index.py              # построить FAISS-индекс из папки с кропами
  eval_reid.py                 # mAP / CMC Rank-1 / Rank-5 на VeRi-776-формате
  download_release_assets.sh   # скачать обученный чекпоинт + демо-индекс
web/           # React + TypeScript фронтенд (Vite)
tests/
  test_imports.py        # дымовой тест, без тяжёлых ML-зависимостей
```

## Датасет

Объединённый датасет для дообучения ArcFace — **10 387 identities**:

- **Drom.ru** (свой парсер) — 97 моделей, 8066 фото, несколько ракурсов
  на объявление.
- **BoxCars116k** — 8001 трек, 29176 фото с 137 реальных камер
  видеонаблюдения (закрывает разрыв доменов между студийными фото и
  городскими камерами).

## Что не сделано (честно)

- Головы атрибутов (марка/модель) обучены, но не подключены в прод — в
  API пока только zero-shot CLIP для цвета/кузова.
- Реранкинг (k-reciprocal) и фильтр по времени/месту камер не реализованы.
- Нет сравнения с внешним бенчмарком (VeRi-776) — доступ не получен за
  время хакатона.

Подробнее — раздел «Известные ограничения» в `DOCUMENTATION.md`.
