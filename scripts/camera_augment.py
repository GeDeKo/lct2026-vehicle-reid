#!/usr/bin/env python3
"""Имитация городской камеры поверх студийных/уличных фото с Drom.

Зачем: фото объявлений и кадры городских камер — разные домены (см. план,
раздел 3 и 6). Простое дообучение на чистых фото объявлений может не
перенестись на реальные камеры. Эта аугментация не решает проблему
полностью (реальные датасеты с камер всё равно нужны — см.
download_datasets.md и BoxCars116k), но частично закрывает разрыв дёшево
и без новых данных: одно и то же фото превращается в 3-5 "деградированных"
версий — как будто снято разными камерами в разных условиях.

Каждый вызов случайно комбинирует:
  - понижение разрешения (реальные камеры часто дают низкое разрешение)
  - motion blur (машина в движении / некачественная оптика)
  - JPEG-пересжатие с низким качеством (сетевые камеры сильно сжимают поток)
  - шум датчика (шумная матрица, особенно ночью)
  - имитация ночи/плохого освещения (понижение яркости и контраста)
  - лёгкая перспективная деформация (камера сверху, а не на уровне глаз)

Использование:
    python scripts/camera_augment.py --input-dir data/raw/drom --out-dir data/processed/camera_aug --variants 3
"""
import argparse
import io
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def simple_motion_blur(img: Image.Image, strength: int) -> Image.Image:
    """Без scipy: направленное размытие через box-blur вдоль случайного угла —
    грубее, чем свёртка с настоящим motion-kernel, но без лишней зависимости."""
    angle = random.choice([0, 45, 90, 135])
    blurred = img.filter(ImageFilter.GaussianBlur(radius=strength / 2))
    if angle in (45, 135):
        blurred = blurred.rotate(angle, expand=False, resample=Image.BICUBIC)
        blurred = blurred.filter(ImageFilter.GaussianBlur(radius=strength / 3))
        blurred = blurred.rotate(-angle, expand=False, resample=Image.BICUBIC)
    return blurred


def add_sensor_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.array(img, dtype="float32")
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(noisy)


def jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def downscale_upscale(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w / factor)), max(1, int(h / factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def simulate_night(img: Image.Image, brightness: float, contrast: float) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def perspective_warp(img: Image.Image, strength: float) -> Image.Image:
    """Грубая имитация вида сверху: сжимаем верх кадра сильнее низа."""
    w, h = img.size
    shift = int(w * strength)
    coeffs = _find_perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],
        [(shift, 0), (w - shift, 0), (w, h), (0, h)],
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)


def _find_perspective_coeffs(src_pts, dst_pts):
    matrix = []
    for (x, y), (X, Y) in zip(dst_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    a = np.array(matrix, dtype="float32")
    b = np.array(src_pts, dtype="float32").reshape(8)
    res = np.linalg.solve(a.T @ a + np.eye(8) * 1e-6, a.T @ b)
    return res.tolist()


def camera_ify(img: Image.Image, seed: int | None = None) -> Image.Image:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = img.convert("RGB")

    # ночь/плохое освещение — не всегда, реальные камеры снимают и днём
    if random.random() < 0.4:
        img = simulate_night(img, brightness=random.uniform(0.4, 0.75), contrast=random.uniform(0.7, 0.95))

    if random.random() < 0.5:
        img = simple_motion_blur(img, strength=random.randint(2, 6))

    if random.random() < 0.3:
        img = perspective_warp(img, strength=random.uniform(0.03, 0.08))

    img = downscale_upscale(img, factor=random.uniform(1.5, 3.5))

    if random.random() < 0.6:
        img = add_sensor_noise(img, sigma=random.uniform(3, 12))

    img = jpeg_recompress(img, quality=random.randint(25, 55))
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="папка с исходными фото (рекурсивно ищет *.jpg)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--variants", type=int, default=3, help="сколько деградированных версий на одно фото")
    parser.add_argument("--limit", type=int, default=None, help="ограничить число исходных фото (для быстрого теста)")
    args = parser.parse_args()

    images = sorted(args.input_dir.rglob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"В {args.input_dir} не найдено .jpg файлов")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, src_path in enumerate(images, 1):
        try:
            img = Image.open(src_path)
        except Exception as e:
            print(f"пропускаю {src_path}: {e}")
            continue

        rel = src_path.relative_to(args.input_dir)
        for v in range(args.variants):
            out_path = args.out_dir / rel.parent / f"{rel.stem}_cam{v}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            camera_ify(img, seed=hash((str(rel), v)) % (2**31)).save(out_path, quality=90)
            total += 1

        if i % 50 == 0 or i == len(images):
            print(f"{i}/{len(images)} исходных фото обработано, создано {total} вариантов")

    print(f"Готово: {total} camera-style версий в {args.out_dir}")


if __name__ == "__main__":
    main()
