"""Общая логика подсчёта re-id метрик (mAP, CMC Rank-k) — переиспользуется
и в `scripts/eval_reid.py` (готовые датасеты формата VeRi-776, нужно
исключать совпадения с той же камеры), и в валидации во время обучения
(`scripts/train_embedder.py`, где query и gallery заведомо не пересекаются
по фото, исключать нечего)."""
import numpy as np


def compute_metrics(
    query_vecs: np.ndarray,
    query_ids: np.ndarray,
    gallery_vecs: np.ndarray,
    gallery_ids: np.ndarray,
    exclude: np.ndarray | None = None,
    top_k: int = 5,
) -> dict:
    """exclude — необязательная булева маска [n_query, n_gallery]: True значит
    "не засчитывать эту пару" (например, тот же трек/камера — не настоящий
    re-id матч). Без неё сравниваются все пары query x gallery."""
    sims = query_vecs @ gallery_vecs.T  # векторы L2-нормированы -> cosine
    aps, rank1_hits, rankk_hits = [], 0, 0

    for i in range(len(query_vecs)):
        order = np.argsort(-sims[i])
        if exclude is not None:
            order = [j for j in order if not exclude[i, j]]

        matches = [gallery_ids[j] == query_ids[i] for j in order]

        if not any(matches):
            aps.append(0.0)
            continue

        hits, precisions = 0, []
        for rank, is_match in enumerate(matches, start=1):
            if is_match:
                hits += 1
                precisions.append(hits / rank)
        aps.append(sum(precisions) / hits)

        if matches[0]:
            rank1_hits += 1
        if any(matches[:top_k]):
            rankk_hits += 1

    n = len(query_vecs)
    return {
        "mAP": float(np.mean(aps)),
        "Rank-1": rank1_hits / n,
        f"Rank-{top_k}": rankk_hits / n,
    }
