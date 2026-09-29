#!/usr/bin/env python3
"""Парсер объявлений auto.drom.ru: марка/модель/цвет/кузов + фото машины
с нескольких ракурсов (одно объявление = одна машина = естественные
положительные пары для re-id).

Проверено вручную (сентябрь 2026): auto.drom.ru отдаёт обычный
server-rendered HTML без антибота и капчи на страницы каталога и
объявлений; specific-агенты (AhrefsBot/SemrushBot/MegaIndex) в
robots.txt запрещены полностью, для Googlebot запрещены служебные
query-параметры (?search, ?clever и т.п.) — общего "User-agent: *"
запрета нет. Мы их не используем и не используем /search/, /clever/,
/profile.php и т.д. Если сайт всё же начнёт капчить/банить —
НЕ обходить это (запрещено), а снизить частоту или остановиться и
уточнить у организаторов.

Использование:
    python scripts/parse_drom.py --brand toyota --model camry --max-ads 50

Результат:
    data/raw/drom/<brand>_<model>/<ad_id>/*.jpg
    data/raw/drom/<brand>_<model>/metadata.jsonl
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SPEC_ROW_RE = re.compile(
    r'data-ftid="specification-([a-z\-]+)"><th[^>]*data-ftid="property">([^<]*)</th>'
    r'<td[^>]*data-ftid="value">(.*?)</td>'
)
AD_URL_RE_TMPL = r'https://auto\.drom\.ru/[a-z\-]+/{brand}/{model}/\d+\.html'
PHOTO_RE = re.compile(r'https://s\d*\.?auto\.drom\.ru/photo/v2/([\w\-]+)/gen1200\.jpg')


def polite_get(url: str, session: requests.Session, delay_range: tuple[float, float]) -> str | None:
    time.sleep(random.uniform(*delay_range))
    resp = session.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        print(f"[{resp.status_code}] пропускаю {url}")
        return None
    resp.encoding = resp.apparent_encoding or "windows-1251"
    return resp.text


def strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html).strip()


def parse_specs(html: str) -> dict:
    specs = {}
    for key, _label, value in SPEC_ROW_RE.findall(html):
        specs[key] = strip_tags(value)
    return specs


def parse_photo_urls(html: str) -> list[str]:
    hashes = list(dict.fromkeys(PHOTO_RE.findall(html)))  # dedup, сохраняем порядок
    return [f"https://s.auto.drom.ru/photo/v2/{h}/gen1200.jpg" for h in hashes]


def collect_ad_urls(brand: str, model: str, max_pages: int, session: requests.Session,
                     delay_range: tuple[float, float]) -> list[str]:
    """auto.drom.ru пагинирует путём, не query-параметром: страница 1 —
    сам листинг, дальше .../page2/, .../page3/ и т.д. (?page=N молча
    отдаёт страницу 1 и создаёт иллюзию пагинации — проверено вручную)."""
    ad_re = re.compile(AD_URL_RE_TMPL.format(brand=brand, model=model))
    seen: set[str] = set()
    urls: list[str] = []
    for page in range(1, max_pages + 1):
        list_url = (
            f"https://auto.drom.ru/{brand}/{model}/"
            if page == 1
            else f"https://auto.drom.ru/{brand}/{model}/page{page}/"
        )
        html = polite_get(list_url, session, delay_range)
        if html is None:
            break
        found = [u for u in dict.fromkeys(ad_re.findall(html)) if u not in seen]
        if not found:
            print(f"страница {page}: новых объявлений не найдено, останавливаюсь")
            break
        seen.update(found)
        urls.extend(found)
        print(f"страница {page}: +{len(found)} объявлений (всего {len(urls)})")
    return urls


def download_photo(url: str, dest: Path, session: requests.Session, delay_range: tuple[float, float]):
    time.sleep(random.uniform(*delay_range))
    resp = session.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 200:
        dest.write_bytes(resp.content)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brand", required=True, help="напр. toyota")
    parser.add_argument("--model", required=True, help="напр. camry")
    parser.add_argument("--max-ads", type=int, default=50, help="сколько НОВЫХ объявлений добрать за этот запуск")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-photos-per-ad", type=int, default=8)
    parser.add_argument("--delay-min", type=float, default=1.5, help="мин. пауза между запросами, сек")
    parser.add_argument("--delay-max", type=float, default=3.0, help="макс. пауза между запросами, сек")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    delay_range = (args.delay_min, args.delay_max)
    out_dir = args.out_dir or Path(__file__).resolve().parent.parent / "data" / "raw" / "drom" / f"{args.brand}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    session = requests.Session()

    # запуск идемпотентный: уже собранные объявления (по ad_id из прошлых
    # запусков) не трогаем и не считаем в --max-ads — так можно смело
    # дозапускать скрипт, чтобы «дособрать ещё», не заботясь о дублях
    known_ids: set[str] = set()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    known_ids.add(json.loads(line)["ad_id"])
        print(f"Уже собрано ранее: {len(known_ids)} объявлений — пропущу их")

    print(f"Собираю ссылки на объявления {args.brand}/{args.model}...")
    all_urls = collect_ad_urls(args.brand, args.model, args.max_pages, session, delay_range)
    new_urls = [u for u in all_urls if u.rstrip("/").split("/")[-1].replace(".html", "") not in known_ids]
    ad_urls = new_urls[: args.max_ads]
    print(f"Найдено новых: {len(new_urls)}, буду обработать {len(ad_urls)}")

    with open(meta_path, "a", encoding="utf-8") as meta_file:
        for i, ad_url in enumerate(ad_urls, 1):
            ad_id = ad_url.rstrip("/").split("/")[-1].replace(".html", "")
            print(f"[{i}/{len(ad_urls)}] {ad_url}")

            html = polite_get(ad_url, session, delay_range)
            if html is None:
                continue

            specs = parse_specs(html)
            photo_urls = parse_photo_urls(html)[: args.max_photos_per_ad]
            if not photo_urls:
                print("  фото не найдено, пропускаю")
                continue

            ad_dir = out_dir / ad_id
            ad_dir.mkdir(exist_ok=True)
            saved_paths = []
            for j, photo_url in enumerate(photo_urls):
                dest = ad_dir / f"{j:02d}.jpg"
                download_photo(photo_url, dest, session, delay_range)
                if dest.exists() and dest.stat().st_size > 0:
                    saved_paths.append(str(dest.relative_to(out_dir.parent.parent.parent)))

            record = {
                "ad_id": ad_id,
                "source_url": ad_url,
                "brand": args.brand,
                "model": args.model,
                "body_type": specs.get("frametype"),
                "color": specs.get("color"),
                "generation": specs.get("generation"),
                "year": specs.get("year"),
                "mileage": specs.get("mileage"),
                "photos": saved_paths,
            }
            meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            meta_file.flush()

    print(f"Готово. Метаданные: {meta_path}")


if __name__ == "__main__":
    main()
