"""REST API поверх пайплайна: то, что дергает веб-демо и что можно
встроить в городские системы.

Запуск: uvicorn src.api.main:app --reload --port 8000
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.config import ROOT, config
from src.pipeline import VehiclePipeline
from src.search.index import VehicleIndex

WEB_DIR = ROOT / "web"

app = FastAPI(title="Vehicle Re-ID", description="Идентификация машин без опоры на госномер")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для демо; в проде сузить до домена веб-интерфейса
    allow_methods=["*"],
    allow_headers=["*"],
)

# картинки матчей (image_path хранится относительно ROOT, вида "data/index/crops/...")
# отдаём отсюда же, чтобы фронтенду не нужен был отдельный файловый сервер
app.mount("/data", StaticFiles(directory=str(ROOT / "data")), name="data")

pipeline = VehiclePipeline(cfg=config)


@app.on_event("startup")
def _load_index():
    """Индекс подгружаем при старте, если он уже построен build_index.py.
    Если файла нет — сервис всё равно поднимется, просто matches будет пустым
    (полезно для локальной разработки без готового индекса)."""
    if config.index.index_path.exists() and config.index.meta_path.exists():
        pipeline.index = VehicleIndex.load(config.embedder.embedding_dim, config.index)


class MatchOut(BaseModel):
    camera_id: str
    timestamp: str
    image_path: str
    score: float


class ProfileOut(BaseModel):
    color: str | None
    body_type: str | None
    features: list[str]
    brand: str | None
    model: str | None


class IdentifyResponse(BaseModel):
    profile: ProfileOut
    matches: list[MatchOut]


@app.get("/health")
def health():
    return {"status": "ok", "index_size": len(pipeline.index._records) if pipeline.index else 0}


@app.get("/")
def web_index():
    """Отдаём фронтенд (web/index.html) с того же порта, что и API — без
    отдельного сервера и без CORS-возни при демо."""
    from fastapi.responses import FileResponse

    return FileResponse(str(WEB_DIR / "index.html"))


@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...)):
    try:
        image = np.array(Image.open(file.file).convert("RGB"))
        result = pipeline.identify(image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return IdentifyResponse(
        profile=ProfileOut(**asdict(result.profile)),
        matches=[
            MatchOut(
                camera_id=m.record.camera_id,
                timestamp=m.record.timestamp,
                image_path=m.record.image_path,
                score=m.score,
            )
            for m in result.matches
        ],
    )


@app.post("/enroll")
async def enroll(file: UploadFile = File(...), camera_id: str = "unknown"):
    if pipeline.index is None:
        raise HTTPException(status_code=409, detail="Индекс не построен — запустите scripts/build_index.py")

    image = np.array(Image.open(file.file).convert("RGB"))
    crop_path = Path(config.index.index_path).parent / "crops" / f"{camera_id}_{len(pipeline.index._records)}.jpg"
    try:
        record = pipeline.enroll(image, camera_id=camera_id, save_crop_to=crop_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # pipeline.enroll() пишет абсолютный путь — фронтенду и /data-статике
    # нужен путь относительно ROOT (см. app.mount("/data", ...) выше)
    record.image_path = str(crop_path.relative_to(ROOT))
    pipeline.index.save()

    return {"record_id": record.record_id}
