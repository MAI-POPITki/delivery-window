"""
Сверка внешних данных с источниками и с архивом постов.

    python scripts/source_audit.py      # → reports/source_audit.md

1. Итог ручной сверки ссылок (data/external/source_audit.csv, 07.11.2026): что на странице действительно
   написано про эту дату / толщину.
2. Правдоподобие дат ледостава и вскрытия после 2012 г. (hydro_post2012_phenology.csv): разброс от года к году
   против архива тех же постов (1993–2012) и связь с погодой (средняя температура апреля–мая для вскрытия).
   Настоящие даты вскрытия меняются от года к году на 4–13 сут и раньше в тёплую весну.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno.ice import load_weather, season_start
from okno.paths import DATA, PROCESSED, REPORTS
from okno.report import md_table


def plausibility():
    ph = pd.read_csv(DATA / "external" / "hydro_post2012_phenology.csv")
    arch = pd.read_csv(PROCESSED / "hydro_ice_seasons.csv")
    rows = []
    for pid, g in ph.groupby("point_id"):
        if not pid.startswith("gauge_"):
            continue
        code = int(pid.split("_")[1])
        a = arch[(arch.gauge_code == code) & arch.valid & (arch.hydro_year >= 1993)].dropna(subset=["drift_start_day"])
        w = load_weather(pid)
        spring = w[w.date.dt.month.isin([4, 5])].groupby("hydro_year").t.mean()
        new_br = pd.Series([(pd.Timestamp(d) - season_start(y)).days for d, y in zip(g.breakup_date, g.hydro_year)],
                           index=g.hydro_year)
        corr = lambda s: float(np.corrcoef(s.values, spring.reindex(s.index).values)[0, 1]) if len(s) >= 4 else np.nan
        rows.append({"пост": pid, "лет (новые)": len(g), "лет (архив)": len(a),
                     "разброс вскрытия, сут (новые)": round(new_br.std(), 1),
                     "разброс вскрытия, сут (архив)": round(a.drift_start_day.std(), 1),
                     "связь с T апреля–мая (новые)": round(corr(new_br), 2),
                     "связь с T апреля–мая (архив)": round(corr(a.set_index("hydro_year").drift_start_day), 2)})
    return pd.DataFrame(rows)


def main():
    a = pd.read_csv(DATA / "external" / "source_audit.csv")
    lines = ["# Сверка внешних данных с источниками", "",
             "Каждая ссылка открыта вручную 07.11.2026; построчно — в "
             "[source_audit.csv](../data/external/source_audit.csv). Строки не удалены: файлы остаются как собраны, "
             "а в выводах используются только подтверждённые.", ""]
    for f, g in a.groupby("file"):
        t = g.verdict.value_counts().rename_axis("итог сверки").reset_index(name="строк")
        lines += [f"## {f}", "", md_table(t), ""]
    lines += ["## Правдоподобие дат после 2012 г. (hydro_post2012_phenology.csv)", "",
              "Настоящие даты вскрытия от года к году гуляют на 4–13 сут и сдвигаются раньше в тёплую весну "
              "(связь с температурой отрицательная). Слишком ровный ряд или связь с обратным знаком — признак "
              "того, что даты не наблюдённые.", "", md_table(plausibility()), "",
              "**Вывод.** Строки по переправе Якутск за 2013–2024 — те же даты, на которых обучена М1 "
              "(новой проверки не дают). Для остальных ссылок даты на страницах не найдены: это главные страницы "
              "сайтов, сводки толщины льда, опубликованные до вскрытия, или новость 2015 г., приведённая для 2021–2025. "
              "Разброс новых дат в 2–8 раз меньше архивного; по Алдану и Яне связь с погодой как у настоящих дат, "
              "по Полую и Пинеге — с обратным знаком. **Для проверки М1 этот файл не используется**, пока даты "
              "не подтверждены конкретными сводками (гидрологические обзоры УГМС за май и ноябрь каждого года).", ""]
    (REPORTS / "source_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
