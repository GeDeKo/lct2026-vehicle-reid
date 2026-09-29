"""Разовая утилита: проверяет, что URL марка/модель существует на
auto.drom.ru и там реально есть объявления (а не 404 / пустая страница).
Не часть основного пайплайна — только чтобы не тратить часы на дозбор
по неверным слагам."""
import re
import sys
import time

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
AD_RE_TMPL = r'https://auto\.drom\.ru/[a-z\-]+/{brand}/{model}/\d+\.html'

CANDIDATES = [
    # --- китайские бренды (несколько вариантов слага на модель, где не уверен) ---
    ("chery", "tiggo7pro"), ("chery", "tiggo-7-pro"),
    ("chery", "tiggo4"), ("chery", "tiggo-4"),
    ("chery", "tiggo8"), ("chery", "tiggo-8"),
    ("chery", "arrizo8"), ("chery", "arrizo-8"),
    ("geely", "coolray"),
    ("geely", "atlas"),
    ("geely", "monjaro"),
    ("geely", "tugella"),
    ("geely", "emgrand"),
    ("haval", "jolion"),
    ("haval", "f7"),
    ("haval", "dargo"),
    ("haval", "m6"),
    ("haval", "h9"),
    ("changan", "cs35plus"), ("changan", "cs35-plus"),
    ("changan", "cs55plus"), ("changan", "cs55-plus"),
    ("changan", "unik"), ("changan", "uni-k"),
    ("exeed", "txl"),
    ("exeed", "lx"),
    ("omoda", "c5"),
    ("omoda", "s5"),
    ("jetour", "x70plus"), ("jetour", "x70-plus"),
    ("jetour", "dashing"),
    ("jac", "js4"),
    ("jac", "s3"),
    ("gac", "gs4"),
    ("gac", "gs8"),
    ("faw", "bestune-t77"), ("faw", "t77"),
    ("tank", "300"),
    ("tank", "500"),
    ("voyah", "free"),
    ("zeekr", "001"),
    ("dongfeng", "ax7"),
    ("belgee", "x50"),
    ("livan", "x3"),
    ("solaris", "hc"),
    ("moskvich", "3"),
    ("bydauto", "song-plus"), ("bydauto", "songplus"), ("byd", "song-plus"),
    ("bydauto", "f3"), ("byd", "f3"),
    ("bydauto", "han"), ("byd", "han"),
    # --- мейнстрим, которых ещё нет в списке ---
    ("lada", "niva-travel"), ("lada", "nivatravel"),
    ("lada", "largus"),
    ("lada", "xray"),
    ("kia", "sportage"),
    ("kia", "seltos"),
    ("kia", "k5"),
    ("hyundai", "creta"),
    ("hyundai", "sonata"),
    ("hyundai", "tucson"),
    ("volkswagen", "tiguan"),
    ("volkswagen", "passat"),
    ("skoda", "rapid"),
    ("skoda", "kodiaq"),
    ("renault", "duster"),
    ("renault", "arkana"),
    ("nissan", "qashqai"),
    ("nissan", "x-trail"), ("nissan", "xtrail"),
    ("toyota", "rav4"),
    ("toyota", "land_cruiser_prado"), ("toyota", "land-cruiser-prado"), ("toyota", "landcruiserprado"),
    ("toyota", "highlander"),
    ("mazda", "3"),
    ("mazda", "cx5"), ("mazda", "cx-5"),
    ("ford", "focus"),
    ("ford", "explorer"),
    ("mitsubishi", "outlander"),
    ("mitsubishi", "asx"),
    ("subaru", "forester"),
    ("honda", "cr-v"), ("honda", "crv"),
    ("honda", "civic"),
    ("peugeot", "408"),
    ("citroen", "c4"),
    ("opel", "astra"),
    ("chevrolet", "niva"),
    ("bmw", "3"),
    ("bmw", "x5"),
    ("mercedes-benz", "e"), ("mercedes", "e"),
    ("mercedes-benz", "glc"), ("mercedes", "glc"),
    ("audi", "a4"),
    ("audi", "q5"),
    ("lexus", "rx"),
    ("lexus", "es"),
    ("infiniti", "qx60"),
    ("uaz", "patriot"),
    ("datsun", "on-do"), ("datsun", "ondo"),
    ("ravon", "r2"),
]


def check(brand: str, model: str, session: requests.Session) -> int:
    url = f"https://auto.drom.ru/{brand}/{model}/"
    try:
        resp = session.get(url, headers=HEADERS, timeout=10)
    except requests.RequestException:
        return 0
    if resp.status_code != 200:
        return -1
    resp.encoding = resp.apparent_encoding or "windows-1251"
    pat = re.compile(AD_RE_TMPL.format(brand=brand, model=model))
    return len(set(pat.findall(resp.text)))


def main():
    session = requests.Session()
    seen_brands = set()
    valid = []
    tested_models = set()  # чтобы не проверять оба варианта модели, если первый уже дал результат
    for brand, model in CANDIDATES:
        key_group = (brand, model.replace("-", ""))
        # пропускаем альтернативный слаг, если для этой пары brand+normalized model уже нашли рабочий
        base_key = (brand,)
        n = check(brand, model, session)
        status = "OK" if n > 0 else ("404/empty" if n <= 0 else "?")
        print(f"{brand:15s} {model:20s} -> {n:3d} объявлений на 1 стр. [{status}]")
        if n > 0:
            valid.append((brand, model, n))
        time.sleep(0.6)

    print("\n=== ИТОГ: рабочие пары (brand, model) ===")
    for brand, model, n in valid:
        print(f'("{brand}", "{model}"),  # {n} на странице')
    print(f"\nвсего рабочих: {len(valid)} из {len(CANDIDATES)} проверенных")


if __name__ == "__main__":
    main()
