"""
Сбор официальных событий зимников Якутии из новостной ленты ЯСИА (тег «автозимник»).

    python scripts/harvest_road_news.py            # → data/external/road_news_ysia.csv, road_events_auto.csv

1. Скачивает все страницы ленты https://ysia.ru/tag/avtozimnik/ (заголовок, дата публикации, ссылка).
2. Размечает заголовок правилами: open (открыт/открыли/разрешено движение), close (закрыт/закрыли/закроют),
   limit_up / limit_down (грузоподъёмность), plan (планируют/ориентировочно/откроют), other.
3. Определяет дорогу по словарю ROADS (ключевые слова в заголовке).

Дата события = дата публикации (новость обычно выходит в день приказа); если в заголовке есть «с 1 апреля»
или «3 декабря», берётся она. Разметка автоматическая — перед выводами её надо просмотреть глазами.
Это данные из СМИ, а не журнал оператора.
"""

import html as html_lib
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "external"
BASE = "https://ysia.ru/tag/avtozimnik/"
UA = {"User-Agent": "okno-zavoza-research/1.0 (open data project)"}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"])}

# дорога → ключевые слова в заголовке (первое совпадение)
ROADS = {
    "Кангалассы — Соттинцы": ["кангалассы"],
    "Мохсоголлох — Качикатцы": ["мохсоголлох"],
    "Хатассы — Павловск": ["хатассы"],
    "Якутск — Нижний Бестях (Лена)": ["нижний бестях", "через реку лену", "через лену"],
    "Вилюй (Тас-Юрях — Верхнемарково)": ["тас-юрях", "таас-юрях", "верхнемарково", "«вилюй»", "а-331"],
    "Алдан (Томпонский р-н)": ["«алдан»", "хандыга", "джебарики"],
    "Анабар": ["«анабар»"],
    "Умнас": ["«умнас»", "умнас"],
    "Себян": ["«себян»", "себян"],
    "Батамай — Сегян-Кюель": ["батамай", "сегян"],
    "Эдьигээн": ["эдьигээн"],
    "Усть-Мая": ["усть-мая", "усть-майск", "белькачи", "болугур"],
    "Сангар": ["«сангар»", "сангар"],
    "Кобяй": ["«кобяй»", "кобяйск"],
    "Амга (переправа)": ["амгу", "амга"],
    "Арктика": ["«арктика»"],
    "Яна": ["«яна»"],
    "Мома / Индигир": ["«мома»", "индигир", "момск"],
    "Покровск — Жемкон": ["жемкон"],
    "Тит-Ары — Синск": ["тит-ары"],
    "Вилюйск — Жиганск": ["жиганск"],
    "Олекминск — Ленск": ["олекминск — ленск", "олекминск-ленск"],
    "Ленский р-н (Нюя)": ["ленском районе", "нюя", "нюю"],
}


def fetch(page):
    url = BASE if page == 1 else f"{BASE}page/{page}/"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"не удалось скачать {url}")


def parse(html):
    """Пары (заголовок, ссылка, «ЧЧ:ММ, ДД месяц [ГГГГ]») из карточек ленты."""
    rows = []
    for m in re.finditer(r'<a[^>]+href="(https://ysia\.ru/[^"#?]+/)"[^>]*>\s*([^<]{25,300}?)\s*</a>(.{0,600}?)(\d{2}:\d{2}),\s*(\d{1,2}) ([а-я]+)(?: (\d{4}))?',
                         html, flags=re.S):
        url, title, _, _, day, month, year = m.groups()
        if "/tag/" in url or "/page/" in url or month not in MONTHS:
            continue
        rows.append(dict(title=html_lib.unescape(re.sub(r"\s+", " ", title).strip()), url=url, day=int(day), month=MONTHS[month],
                         year=int(year) if year else None))
    return rows


def classify(t):
    t = t.lower()
    if re.search(r"планиру|ориентировочн|откро(ют|ется)|назван[аы] .*(дата|срок)|могут открыть|готовят|сроки открытия|рассказали, когда", t):
        return "plan"
    if re.search(r"грузоподъ[её]мност|тонн", t):
        if re.search(r"сниж|до 10 тонн|до 20 тонн.*сниж", t):
            return "limit_down"
        if re.search(r"увелич|повыс|повыш|до \d+ тонн", t):
            return "limit_up"
    if re.search(r"закры|закро|прекращ|запрещ|перекро", t):
        return "close_temp" if re.search(r"временно|из-за (погод|ухудш)|учени", t) else "close"
    if re.search(r"откры|разреш|заработал|восстанов", t):
        return "open"
    return "other"


def road_of(t):
    t = t.lower()
    for road, keys in ROADS.items():
        if any(k in t for k in keys):
            return road
    return ""


def event_date(row, pub):
    """Дата из заголовка («с 1 апреля», «3 декабря»), иначе дата публикации."""
    m = re.search(r"(?:с |^|\s)(\d{1,2}) (" + "|".join(MONTHS) + r")", row["title"].lower())
    if m:
        mon = MONTHS[m.group(2)]
        y = pub.year + (1 if mon < pub.month - 6 else 0) - (1 if mon > pub.month + 6 else 0)
        try:
            return pd.Timestamp(y, mon, int(m.group(1)))
        except ValueError:
            pass
    return pub


def main():
    rows, page = [], 1
    while True:
        html = fetch(page)
        if html is None:
            break
        got = parse(html)
        if not got:
            break
        rows += got
        page += 1
        time.sleep(1)
    df = pd.DataFrame(rows).drop_duplicates("url")
    # у новостей текущего года сайт не пишет год
    df["year"] = df["year"].fillna(pd.Timestamp.now().year)
    df["published"] = [pd.Timestamp(int(y), m, d) for y, m, d in zip(df.year, df.month, df.day)]
    # записи без года до первой датированной относятся к текущему году
    now = pd.Timestamp.now()
    df.loc[df.published > now, "published"] -= pd.DateOffset(years=1)
    df["kind"] = df.title.map(classify)
    df["road"] = df.title.map(road_of)
    df["event_date"] = [event_date(r, p) for r, p in zip(df.to_dict("records"), df.published)]
    df["hydro_year"] = df.event_date.dt.year + (df.event_date.dt.month >= 9).astype(int)
    df = df[["published", "event_date", "hydro_year", "kind", "road", "title", "url"]].sort_values("published")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "road_news_ysia.csv", index=False, date_format="%Y-%m-%d")
    # первое открытие и последнее закрытие по дороге и сезону — кандидаты в road_events (проверить вручную)
    ev = df[df.road.ne("") & df.kind.isin(["open", "close", "plan"])]
    first_open = ev[ev.kind == "open"].groupby(["road", "hydro_year"]).event_date.min().rename("first_open")
    last_close = ev[ev.kind == "close"].groupby(["road", "hydro_year"]).event_date.max().rename("last_close")
    plan = ev[ev.kind == "plan"].groupby(["road", "hydro_year"]).title.first().rename("plan_title")
    summary = pd.concat([first_open, last_close, plan], axis=1).reset_index()
    summary.to_csv(OUT / "road_events_auto.csv", index=False, date_format="%Y-%m-%d")
    print(f"новостей: {len(df)} (страниц {page - 1}); с дорогой: {(df.road != '').sum()}; "
          f"пар дорога×сезон: {len(summary)}")
    print(df.kind.value_counts().to_string())


if __name__ == "__main__":
    main()
