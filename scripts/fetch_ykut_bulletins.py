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
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from okno.ykut_phenology import OUT_PATH, extract_phenology, fetch_html, parse_posts, update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", type=Path, default=None)
    args = parser.parse_args()
    df, posts = update(cached=args.cached, out=OUT_PATH)
    print(f"Бюллетеней разобрано: {len(posts)} ({posts[0][0]} .. {posts[-1][0]})")
    print(f"Извлечено событий: {len(df)} по {df.point_id.nunique()} постам, сезоны {df.hydro_year.min()}..{df.hydro_year.max()}")
    print(df.groupby(["event_type"]).agg(n=("event_date", "count"), stations=("point_id", "nunique")))
    print(f"Сохранено: {OUT_PATH}")


if __name__ == "__main__":
    main()
