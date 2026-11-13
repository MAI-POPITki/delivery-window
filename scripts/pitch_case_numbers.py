"""
Цифры для кейсов 2–3 презентации: диспетчер (анонс приказа против факта) и штаб паводка (выход реки).

    python scripts/pitch_case_numbers.py   # → reports/pitch_cases.md

Только сохранённые данные проекта, без новых допущений:
* data/external/road_events.csv — пары «объявленная дата — фактическая дата» приказов зимников;
* reports/flood_day_predictions.csv.gz — архивные выпуски вероятности высокой воды (прогноз только по данным на дату).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "pitch_cases.md"


def dispatcher():
    d = pd.read_csv(ROOT / "data/external/road_events.csv",
                    parse_dates=["announced_date", "actual_date", "actual_published"])
    a = d.dropna(subset=["announced_date", "actual_date"]).copy()
    a["diff"] = (a.actual_date - a.announced_date).dt.days
    restrict = a.kind.isin(["limit30", "limit20", "close"])
    risk = a[(restrict & (a["diff"] < 0)) | ((a.kind == "open") & (a["diff"] > 0))]
    lost = a[(restrict & (a["diff"] > 0)) | ((a.kind == "open") & (a["diff"] < 0))]
    pub = (a.actual_published - a.actual_date).dt.days.dropna()
    cols = ["road", "hydro_year", "kind", "announced_date", "actual_date", "diff"]

    def show(x):
        return x[cols].assign(announced_date=x.announced_date.dt.date, actual_date=x.actual_date.dt.date)
    return dict(pairs=len(a), roads=a.road.nunique(), years=(int(a.hydro_year.min()), int(a.hydro_year.max())),
                diverged=int((a["diff"] != 0).sum()),
                risk_n=len(risk), risk_days=int(risk["diff"].abs().sum()),
                lost_n=len(lost), lost_days=int(lost["diff"].abs().sum()),
                pub_median=float(pub.median()),
                risk_rows=show(risk), lost_rows=show(lost))


def flood(threshold=0.5):
    d = pd.read_csv(ROOT / "reports/flood_day_predictions.csv.gz", parse_dates=["issue", "target_date"])
    e = d[["pid", "target_date", "event"]].drop_duplicates().sort_values(["pid", "target_date"])
    e["prev"] = e.groupby("pid").event.shift(1)
    onsets = e[(e.event == 1) & (e.prev == 0)]
    rows = []
    for r in onsets.itertuples():
        p = d[(d.pid == r.pid) & (d.target_date == r.target_date)].set_index("lead")
        if 1 not in p.index:
            continue  # нет выпуска за сутки — событие вне оценочного окна
        lead = 0
        while (lead + 1) in p.index and p.probability[lead + 1] >= threshold:
            lead += 1
        rows.append(dict(post=r.pid, date=r.target_date.date(), warned_days=lead,
                         p_1day=round(float(p.probability[1]), 2),
                         persistence_1day=round(float(p.persistence_probability[1]), 2)))
    calm = d[(d.lead == 1) & (d.event == 0)]
    return pd.DataFrame(rows), dict(calm=len(calm), fa_service=int((calm.probability >= threshold).sum()),
                                    fa_persistence=int((calm.persistence_probability >= threshold).sum()))


def main():
    a = dispatcher()
    f, fa = flood()
    n = len(f)
    lines = [
        "# Цифры кейсов 2–3 презентации",
        "",
        "Воспроизведение: `python scripts/pitch_case_numbers.py`. Только сохранённые данные проекта.",
        "",
        "## Диспетчер: объявленная дата приказа против фактической",
        "",
        f"Пар «анонс — факт»: {a['pairs']} на {a['roads']} дорогах, {a['years'][0]}–{a['years'][1]} гг. "
        f"Факт не совпал с анонсом: {a['diverged']}.",
        f"* План по анонсу пускает машину туда, где уже действует предел/закрытие или ещё не открыто: "
        f"{a['risk_n']} случаев, {a['risk_days']} суток суммарно.",
        f"* План по анонсу не использует дни, когда дорога реально работала: {a['lost_n']} случаев, "
        f"{a['lost_days']} суток суммарно.",
        f"* Медианная задержка публикации фактического приказа: {a['pub_median']:.0f} сут.",
        "",
        "Реестр собран по официальным сообщениям вручную и скриптом `harvest_road_news.py`; ежедневный сбор не включён.",
        "",
        a["risk_rows"].to_markdown(index=False),
        "",
        a["lost_rows"].to_markdown(index=False),
        "",
        "## Штаб паводка: выход реки выше порога, которого накануне не было",
        "",
        "Порог: Урал — выход на пойму (НСА); Лена и Ишим — 10% самых высоких уровней (q90). "
        "Предупреждение — вероятность ≥50% без перерыва до даты события.",
        "",
        f"Выходов с выпуском за сутки: {n}. Предупреждены за ≥1 сут: {(f.warned_days >= 1).sum()}, "
        f"за ≥2 сут: {(f.warned_days >= 2).sum()}, за ≥3 сут: {(f.warned_days >= 3).sum()}. "
        f"Правило «уровень как сегодня» за сутки: {(f.persistence_1day >= 0.5).sum()} из {n}.",
        f"Ложные тревоги за сутки на {fa['calm']} спокойных днях: сервис {fa['fa_service']}, "
        f"«как сегодня» {fa['fa_persistence']}.",
        "",
        f.to_markdown(index=False),
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
