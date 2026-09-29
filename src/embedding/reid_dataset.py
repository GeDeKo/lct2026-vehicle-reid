"""Датасет для дообучения эмбеддера на собранных данных (Drom + BoxCars и
любые другие источники в той же структуре).

Каждая папка `<root>/<марка_модель или "boxcars">/<ad_id или track_id>/*.jpg`
— один физический автомобиль с нескольких ракурсов. Это и есть "класс" для
ArcFace (см. arcface.py). Мало фото на класс (2-6), это нормально
для re-id-постановки — тут не общая классификация, а метрика близости.

Чтобы модель училась не только "разным ракурсам одной студийной съёмки",
а действительно переносилась на кадры городских камер, часть сэмплов на
лету прогоняется через camera_augment.camera_ify (имитация камеры: blur,
шум, сжатие, ночь, перспектива) — тогда в один класс попадают и чистое
фото, и его "закамеренная" версия, и модель учится узнавать машину
несмотря на деградацию.
"""
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.camera_augment import camera_ify  # noqa: E402


def collect_identities(roots: Path | list[Path], min_photos: int = 2) -> list[list[Path]]:
    """Сканирует один или несколько root-каталогов вида
    `<root>/<model_dir>/<identity_dir>/*.jpg` и возвращает список identity —
    каждая identity это список путей к её фото. Несколько root
    (например data/raw/drom + data/processed/boxcars_subset) просто
    объединяются в один общий список классов для ArcFace — так объединяются
    датасеты без физического слияния файлов на диске."""
    if isinstance(roots, (str, Path)):
        roots = [roots]

    identities: list[list[Path]] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise RuntimeError(f"Каталог датасета не найден: {root}")
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            for ad_dir in sorted(model_dir.iterdir()):
                if not ad_dir.is_dir():
                    continue
                photos = sorted(ad_dir.glob("*.jpg"))
                if len(photos) >= min_photos:
                    identities.append(photos)

    if not identities:
        raise RuntimeError(f"В {roots} не найдено ни одной папки с >= {min_photos} фото")

    return identities


class VehicleIdentityDataset(Dataset):
    """label = индекс identity в списке; в каждом __getitem__ отдаём одно фото
    (случайное из папки), с вероятностью camera_prob прогнанное через
    имитацию камеры.

    Передавай либо root_dir (один каталог или список — тогда identity
    собираются с диска через collect_identities), либо уже готовый список
    identities (например, train-часть после split_identities) — так train и
    val строятся из одного и того же скана диска, без риска рассинхрона."""

    def __init__(self, root_dir: Path | list[Path] | None = None, *,
                 identities: list[list[Path]] | None = None,
                 processor=None, image_size: int = 224,
                 camera_prob: float = 0.4, min_photos: int = 2):
        if identities is None:
            if root_dir is None:
                raise ValueError("Нужен либо root_dir, либо identities")
            identities = collect_identities(root_dir, min_photos=min_photos)

        self.processor = processor
        self.image_size = image_size
        self.camera_prob = camera_prob
        self.identities = identities
        self.num_classes = len(self.identities)

    def __len__(self):
        # эпоха = по 2 сэмпла на каждую identity (для triplet-подобного сигнала
        # внутри батча достаточно двух разных ракурсов одной машины на класс)
        return len(self.identities) * 2

    def _load_and_transform(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if random.random() < self.camera_prob:
            img = camera_ify(img)
        return img.resize((self.image_size, self.image_size))

    def __getitem__(self, idx: int):
        label = idx % len(self.identities)
        photos = self.identities[label]
        path = random.choice(photos)
        img = self._load_and_transform(path)

        if self.processor is not None:
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        else:
            import numpy as np

            arr = np.array(img, dtype="float32") / 255.0
            pixel_values = torch.from_numpy(arr).permute(2, 0, 1)

        return pixel_values, label


def split_identities(identities: list[list[Path]], val_fraction: float = 0.1,
                      seed: int = 42) -> tuple[list[list[Path]], list[list[Path]]]:
    """Делит identity (не фото!) на train/val. Важно резать по identity, а
    не по фото: если два ракурса одной машины разъедутся по train и val,
    val перестаёт быть честной проверкой на "невиданных раньше" машинах —
    модель могла запомнить именно эту машину на train. По той же причине
    val-identity не участвуют в ArcFace-классификации вообще (это была бы
    классификация по классам, которые видели на train, а не re-id)."""
    rng = random.Random(seed)
    shuffled = identities[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_identities = shuffled[:n_val]
    train_identities = shuffled[n_val:]
    return train_identities, val_identities
