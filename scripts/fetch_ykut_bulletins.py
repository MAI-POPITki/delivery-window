"""
Парсер официального ежедневного архива гидрологических бюллетеней ФГБУ «Якутское УГМС»
(https://ykuthydromet.ru/reki/, 2022–2026 гг.).

Загрузка АИС ГМВО (gmvo.skniivh.ru / gvr.rwec.ru) невозможна: оба ресурса официально выведены
из эксплуатации в пользу закрытой ГИС ЦП «Вода» (требуется клиент Континент TLS).
Вместо них открытый архив ФГБУ «Якутское УГМС» содержит 800+ датированных суточных сводок
за 2022–2026 гг. с точными датами ледостава, появления воды на льду/закраин и вскрытия по постам.

Запуск:
    python scripts/fetch_ykut_bulletins.py
    python scripts/fetch_ykut_bulletins.py --cached /tmp/ykut_reki.html

Выход:
    data/external/ykut_reki_phenology.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from okno.ice import to_season_day
from okno.paths import EXTERNAL

URL = "https://ykuthydromet.ru/reki/"

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

# Строгие правила сопоставления гидропоста (исключаем ложные совпадения вроде «Верхоянский Перевоз» на Алдане)
STATION_SPECS = [
    {
        "point_id": "yakutsk_crossing", "gauge_code": "", "river": "Лена",
        "station_name": "Табага — Якутск — Кангалассы",
        "pattern": r"(?:у\s+(?:г\.|гидропоста)\s*якутск|у\s+(?:с\.|гидропоста)\s*табаг|табаг[аеу]?\s*[–-]\s*(?:г\.?\s*)?якутск|якутск\s*[–-]\s*кангаласс)",
    },
    {
        "point_id": "gauge_3156", "gauge_code": "3156", "river": "Нюя",
        "station_name": "Нюя — Курум / Беченча",
        "pattern": r"(?:у\s+(?:с\.|п\.|гидропоста)\s*(?:беченч|курум)|беченч[аеи]?\s*\(\s*курум\s*\))",
    },
    {
        "point_id": "gauge_3219", "gauge_code": "3219", "river": "Алдан",
        "station_name": "Алдан — Томмот",
        "pattern": r"(?:у\s+(?:г\.|гидропоста)\s*томмот|суон-тит\s*[–-]\s*томмот)",
    },
    {
        "point_id": "gauge_3277", "gauge_code": "3277", "river": "Аллах-Юнь",
        "station_name": "Аллах-Юнь — Аллах",
        "pattern": r"(?:у\s+(?:гидропоста|с\.)\s*аллах)",
    },
    {
        "point_id": "gauge_3403", "gauge_code": "3403", "river": "Малая Куонапка",
        "station_name": "Малая Куонапка — Джалинда (Жилинда)",
        "pattern": r"(?:малая\s+куонапка\s+у\s+(?:с\.|п\.|гидропоста)\s*(?:жилинд|джалинд)|у\s+(?:с\.|гидропоста)\s*(?:жилинд|джалинд))",
    },
    {
        "point_id": "gauge_3414", "gauge_code": "3414", "river": "Яна",
        "station_name": "Яна — Верхоянск",
        "pattern": r"(?:у\s+(?:г\.|гидропоста)\s*верхоянск(?!ий)|верхоянск\s*[–-]\s*(?:батагай|сайды))",
    },
    {
        "point_id": "gauge_3483", "gauge_code": "3483", "river": "Бытантай",
        "station_name": "Бытантай — Асар",
        "pattern": r"(?:у\s+(?:гидропоста|с\.)\s*асар)",
    },
    {
        "point_id": "gauge_3489", "gauge_code": "3489", "river": "Индигирка",
        "station_name": "Индигирка — Усть-Нера (Индигирский)",
        "pattern": r"(?:у\s+(?:п\.|с\.|гидропоста)\s*(?:усть-нер|индигирск)|усть-нер[аеу]?\s*[–-]\s*(?:хонуу?|белая))",
    },
    {
        "point_id": "gauge_1018", "gauge_code": "1018", "river": "Колыма",
        "station_name": "Колыма — Зырянка",
        "pattern": r"(?:у\s+(?:п\.|гидропоста|г\.?п\.?)\s*зырянк)",
    },
    {
        "point_id": "gauge_1367", "gauge_code": "1367", "river": "Берёзовка",
        "station_name": "Берёзовка — Берёзовка",
        "pattern": r"(?:у\s+(?:с\.|гидропоста)\s*бер[её]зовк)",
    },
    {
        "point_id": "gauge_1578", "gauge_code": "1578", "river": "Ясачная",
        "station_name": "Ясачная — Нелемное",
        "pattern": r"(?:у\s+(?:с\.|гидропоста)\s*нелемн)",
    },
    {
        "point_id": "gauge_1801", "gauge_code": "1801", "river": "Колыма",
        "station_name": "Колыма — Среднеколымск",
        "pattern": r"(?:у\s+(?:г\.|гидропоста)\s*среднеколымск)",
    },
    {
        "point_id": "gauge_1802", "gauge_code": "1802", "river": "Колыма",
        "station_name": "Колыма — Колымское",
        "pattern": r"(?:у\s+(?:с\.|гидропоста)\s*колымск[оа])",
    },
    {
        "point_id": "gauge_1805", "gauge_code": "1805", "river": "Колыма",
        "station_name": "Колыма — Черский",
        "pattern": r"(?:у\s+(?:п\.|гидропоста)\s*черский)",
    },
    {
        "point_id": "zhigansk", "gauge_code": "", "river": "Лена",
        "station_name": "Лена — Жиганск",
        "pattern": r"(?:у\s+(?:п\.|с\.|гидропоста)\s*жиганск)",
    },
    {
        "point_id": "saskylakh", "gauge_code": "", "river": "Анабар",
        "station_name": "Анабар — Саскылах",
        "pattern": r"(?:у\s+(?:с\.|п\.|гидропоста)\s*саскылах)",
    },
    {
        "point_id": "olenek", "gauge_code": "", "river": "Оленёк",
        "station_name": "Оленёк — Оленёк",
        "pattern": r"(?:у\s+(?:с\.|п\.|гидропоста)\s*оленек)",
    },
    {
        "point_id": "chokurdakh", "gauge_code": "", "river": "Индигирка",
        "station_name": "Индигирка — Чокурдах",
        "pattern": r"(?:у\s+(?:п\.|с\.|гидропоста)\s*чокурдах)",
    },
]

_EXCLUDE_RE = re.compile(
    r"ожида|прогноз|предполага|осуществится|за\s+исключением|сохраняется\s+ледостав|"
    r"сохраняется\s+[«\"]?зимний\s+режим|продолжается\s+ледоход|ниже\s+гидропоста|выше\s+гидропоста|"
    r"нижняя\s+кромка\s+ледохода\s+(?:наблюдается|отмечается)\s+(?:ниже|выше)",
    re.I,
)
_FREEZEUP_EXCLUDE_RE = re.compile(r"лишь\s+(?:на|у)\b|шугоход|заберег|остается\s+свободн|не\s+установил", re.I)
_FREEZEUP_RE = re.compile(
    r"(?:установил|образовал|установление|образование)\w*\s+(?:неполный\s+|сплошной\s+)?(?:ледостав|ледян\w+\s+покров)",
    re.I,
)
_MELT_START_RE = re.compile(
    r"(?:отмеча\w+\s+(?:вода\s+на\s+льду|закраин)|наблюда\w+\s+(?:вода\s+на\s+льду|закраин)|"
    r"вода\s+(?:на\s+льду|течет\s+поверх\s+льда)|закраины|лед\s+(?:потемнел|подняло)|"
    r"начал\w*\s+подвижк\w+\s+льда|подвижк[аи]\s+льда|разводья)",
    re.I,
)
_BREAKUP_RE = re.compile(
    r"(?:начал\w*\s+ледоход|осуществил\w*\s+вскрыт|вскрытие\s+(?:реки|р\.))",
    re.I,
)


def fetch_html(cached: Path | None = None) -> str:
    if cached is not None and cached.exists():
        return cached.read_text(encoding="utf-8")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "replace")


def parse_posts(raw_html: str) -> list[tuple[dt.date, str, list[str]]]:
    parts = re.split(r"<div\s+class=post\s+data-year=(\d+)\s+data-month=(\d+)\s+data-day=(\d+)>", raw_html)
    posts = []
    for i in range(1, len(parts), 4):
        y, m, d, body = int(parts[i]), int(parts[i + 1]), int(parts[i + 2]), parts[i + 3]
        href_m = re.search(r"class=file\s+href=([^\s>]+)", body)
        href = href_m.group(1) if href_m and href_m.group(1).startswith("http") else URL
        cm = re.search(r"<div\s+class=content[^>]*>(.*)", body, re.S)
        c = cm.group(1) if cm else body
        c = re.sub(r"</?(?:p|div|br|li|tr|h[1-6])[^>]*>", "\n", c, flags=re.I)
        txt = html_lib.unescape(re.sub(r"<[^>]+>", " ", c))
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in txt.splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("Гидролог ")]
        posts.append((dt.date(y, m, d), href, lines))
    posts.sort(key=lambda x: x[0])
    return posts


_ABBR_TOKEN = "§"


def split_sentences(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        # Защищаем сокращения вроде «р.», «г.», «с.», «п.», «мкр.», «г.п.», «ст.», «млн.»
        protected = re.sub(r"\b(г\.п|мкр|усл|ст|млн|тыс|[ргспдо])\.", rf"\1{_ABBR_TOKEN}", ln, flags=re.I)
        # Склеенные на стыке HTML-тегов: «ледоходДнем», «см).У», «подняло».На»
        protected = re.sub(r"([а-яё])([А-ЯЁ][а-яё])", r"\1\n\2", protected)
        protected = re.sub(r"(?<!\d)([.!?][»)\"]*)\s*(?=[А-ЯЁ])", r"\1\n", protected)
        protected = re.sub(r"(?<!\d)([.!?][»)\"]*)\s*(?=\d{1,2}\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр))", r"\1\n", protected, flags=re.I)
        for chunk in protected.splitlines():
            s = chunk.replace(_ABBR_TOKEN, ".").strip()
            if s:
                out.append(s)
    return out


def extract_event_date(sent: str, bdate: dt.date) -> dt.date:
    if re.search(r"\bвчера\b", sent, re.I):
        return bdate - dt.timedelta(days=1)
    m = re.search(
        r"\b(\d{1,2})(?:\s*[–-]\s*\d{1,2})?\s+"
        r"(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
        sent,
        re.I,
    )
    if m:
        day = int(m.group(1))
        mon = MONTHS[m.group(2).lower()]
        try:
            d = dt.date(bdate.year, mon, day)
            if (d - bdate).days > 30:
                d = dt.date(bdate.year - 1, mon, day)
            return d
        except ValueError:
            pass
    m2 = re.search(r"\b(\d{2})\.(\d{2})\b", sent)
    if m2:
        day, mon = int(m2.group(1)), int(m2.group(2))
        if 1 <= mon <= 12 and 1 <= day <= 31:
            try:
                return dt.date(bdate.year, mon, day)
            except ValueError:
                pass
    return bdate


def extract_phenology(posts: list[tuple[dt.date, str, list[str]]]) -> pd.DataFrame:
    rows = []
    seen: set[tuple[str, int, str]] = set()
    for bd, href, lines in posts:
        for sent in split_sentences(lines):
            if _EXCLUDE_RE.search(sent):
                continue
            for spec in STATION_SPECS:
                if not re.search(spec["pattern"], sent, re.I):
                    continue
                matched_types = []
                if bd.month in (10, 11, 12) and _FREEZEUP_RE.search(sent) and not _FREEZEUP_EXCLUDE_RE.search(sent):
                    matched_types.append("freezeup")
                if bd.month in (4, 5, 6) and _MELT_START_RE.search(sent):
                    matched_types.append("melt_start")
                if bd.month in (4, 5, 6) and _BREAKUP_RE.search(sent):
                    matched_types.append("breakup")
                for etype in matched_types:
                    ed = extract_event_date(sent, bd)
                    # Уточняем гидрологический год по фактической дате события
                    hy = ed.year + 1 if ed.month >= 9 else ed.year
                    key = (spec["point_id"], hy, etype)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "point_id": spec["point_id"],
                        "gauge_code": spec["gauge_code"],
                        "river": spec["river"],
                        "station_name": spec["station_name"],
                        "hydro_year": hy,
                        "event_type": etype,
                        "event_date": ed.isoformat(),
                        "season_day": int(to_season_day(ed.isoformat(), hy)),
                        "bulletin_date": bd.isoformat(),
                        "source_url": href,
                        "quote": sent,
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["point_id", "hydro_year", "event_type"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", type=Path, default=None)
    args = parser.parse_args()
    raw = fetch_html(args.cached)
    posts = parse_posts(raw)
    df = extract_phenology(posts)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    out = EXTERNAL / "ykut_reki_phenology.csv"
    df.to_csv(out, index=False)
    print(f"Бюллетеней разобрано: {len(posts)} ({posts[0][0]} .. {posts[-1][0]})")
    print(f"Извлечено событий: {len(df)} по {df.point_id.nunique()} постам, сезоны {df.hydro_year.min()}..{df.hydro_year.max()}")
    print(df.groupby(["event_type"]).agg(n=("event_date", "count"), stations=("point_id", "nunique")))
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
