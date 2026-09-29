"""ArcFace head для дообучения эмбеддера как re-id метрики.

Идея (стандартный приём в re-id, например "Bag of Tricks", Luo et al. 2019):
каждая уникальная машина (ad_id папка на Drom) — это один класс. Обучаем
обычную классификацию с ArcFace-лоссом поверх эмбеддингов бэкбона; на
инференсе классификационную голову выбрасываем, а сам эмбеддинг (до
головы) уже хорошо разделяет разные машины — это и есть то, что нужно
для re-id (поиск ближайшего соседа по векторам).

ArcFace вместо обычного softmax — потому что он явно раздвигает угловое
расстояние между классами, а для нас "разные ракурсы одной машины должны
лежать кучно, а разные машины — далеко" это ближе к делу, чем просто
"разделимость для классификации".
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.30):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        emb_norm = F.normalize(embeddings, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cosine = emb_norm @ w_norm.t()

        if labels is None:
            # инференс/валидация без лоcса — просто логиты
            return cosine * self.scale

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale
