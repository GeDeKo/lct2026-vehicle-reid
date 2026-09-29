"""Атрибуты машины для «цифрового профиля».

Два источника:
1. Обученные головы поверх бэкбона эмбеддера (марка/модель/кузов/цвет) —
   их добавляем, когда размечен датасет с Drom/Avito (см. scripts/parse_drom.py).
2. Zero-shot через CLIP для «видимых особенностей» (наклейки, тонировка,
   повреждения, диски) — размечать такое дорого, а zero-shot даёт быстрый
   baseline для прототипа.

Пока обученных голов нет — AttributeTagger работает полностью в zero-shot
режиме через CLIP-текстовые промпты. Как только появятся датасет и обученные
головы, метод classify() подменяется без изменения остального пайплайна.
"""
from dataclasses import dataclass, field

COLOR_PROMPTS = ["белая машина", "чёрная машина", "серебристая машина", "серая машина",
                 "красная машина", "синяя машина", "зелёная машина", "жёлтая машина"]
BODY_PROMPTS = ["седан", "хэтчбек", "внедорожник", "кроссовер", "универсал", "минивэн", "пикап"]
FEATURE_PROMPTS = ["машина с тонированными стёклами", "машина с наклейками на кузове",
                    "машина с повреждением кузова", "машина со спортивными дисками",
                    "чистая машина без повреждений"]


@dataclass
class VehicleProfile:
    color: str | None = None
    body_type: str | None = None
    features: list[str] = field(default_factory=list)
    brand: str | None = None   # заполняется обученной головой позже
    model: str | None = None   # заполняется обученной головой позже


class AttributeTagger:
    def __init__(self, clip_model: str = "ViT-L-14", clip_pretrained: str = "openai",
                 device: str = "cpu", feature_threshold: float = 0.28):
        self.clip_model_name = clip_model
        self.clip_pretrained = clip_pretrained
        self.device = device
        self.feature_threshold = feature_threshold
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained
        )
        tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._torch = torch

    def _best_match(self, pil_image, prompts: list[str]) -> tuple[str, float]:
        image_tensor = self._preprocess(pil_image).unsqueeze(0).to(self.device)
        text_tokens = self._tokenizer(prompts).to(self.device)
        with self._torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0)
        best_idx = int(sims.argmax())
        return prompts[best_idx], float(sims[best_idx])

    def classify(self, image) -> VehicleProfile:
        self._load()
        from PIL import Image
        import numpy as np

        pil_image = image if hasattr(image, "convert") else Image.fromarray(np.asarray(image))
        pil_image = pil_image.convert("RGB")

        color, _ = self._best_match(pil_image, COLOR_PROMPTS)
        body, _ = self._best_match(pil_image, BODY_PROMPTS)

        features = []
        image_tensor = self._preprocess(pil_image).unsqueeze(0).to(self.device)
        text_tokens = self._tokenizer(FEATURE_PROMPTS).to(self.device)
        with self._torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0)
        for prompt, sim in zip(FEATURE_PROMPTS, sims.tolist()):
            if sim >= self.feature_threshold and "чистая машина" not in prompt:
                features.append(prompt)

        return VehicleProfile(color=color, body_type=body, features=features)
