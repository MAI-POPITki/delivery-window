"""
Расчёт «до/после» для кейса А3 на прошлых сезонах + анализ чувствительности ±30%.

    python scripts/before_after.py

Было («до»):  равномерные отправки до календарной даты закрытия (средняя за прошлые 10 сезонов).
Стало («после»): еженедельный пересчёт графика по прогнозу М1 (планировщик М3).
Оба правила прогоняются на каждом прошлом сезоне с ФАКТИЧЕСКОЙ датой закрытия; М1 видит
только то, что было известно на дату решения (прогнозы из бэктеста leave-one-season-out).

Примеры:
- arctic: арктический зимник Верхоянского района (Батагай); закрытие — по замерам гидропоста
  «Яна — Верхоянск» (3414), начало таяния минус запас;
- yakutsk: переправа Якутск — Нижний Бестях; закрытие — по спутниковому вскрытию Лены.

Выход: reports/before_after_{example}.csv (по сезонам), reports/before_after.md,
       reports/sensitivity.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno.planner import SENSITIVE, Params, simulate_season, warning_lead
from okno.paths import REPORTS
from okno.report import md_table

EXAMPLES = {
    "arctic": {"point_id": "gauge_3414", "name": "Зимник на Батагай (Верхоянский р-н), гидропост Яна — Верхоянск"},
    "yakutsk": {"point_id": "yakutsk_crossing", "name": "Переправа Якутск — Нижний Бестях (спутник)"},
}


def run_example(bt: pd.DataFrame, point_id: str, p: Params) -> pd.DataFrame:
    rows = []
    b = bt[bt.point_id == point_id]
    events = b.groupby("hydro_year").target_day.first()
    for y, g in b.groupby("hydro_year"):
        event = float(g.target_day.iloc[0])
        cal_event = float(g.cal_target.iloc[0])
        actual_close = event - p.close_margin_days
        cal_close = cal_event - p.close_margin_days
        fc = g[["issue_day", "q10", "q50", "q90"]]
        # то же вероятностное планирование М3, но без М1: распределение = календарь ± разброс прошлых лет
        past = events[(events.index < y) & (events.index >= y - 10)]
        sd = float(past.std()) if len(past) >= 3 else 7.0
        fc_clim = pd.DataFrame({"issue_day": g.issue_day, "q50": cal_event,
                                "q10": cal_event - 1.2816 * sd, "q90": cal_event + 1.2816 * sd})
        lead = warning_lead(fc, event, cal_event)
        for rule, rule_fc in (("calendar", None), ("forecast", fc), ("asap", None), ("clim_prob", fc_clim)):
            r = simulate_season(actual_close, rule_fc, cal_close, p, "forecast" if rule == "clim_prob" else rule)
            r["rule"] = rule
            r.update({"hydro_year": y, "season": f"{y - 1}/{str(y)[2:]}", "actual_close_day": actual_close,
                      "cal_close_day": cal_close, "close_error_cal_days": cal_event - event,
                      "warning_lead_days": lead if rule == "forecast" else 0.0})
            rows.append(r)
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, bt_point: pd.DataFrame) -> pd.DataFrame:
    s14 = bt_point[bt_point.lead_days.between(11, 17)]
    err_cal = (s14.cal_target - s14.target_day).abs().mean()
    err_m1 = (s14.q50 - s14.target_day).abs().mean()
    c, f, a, k = (res[res.rule == r] for r in ("calendar", "forecast", "asap", "clim_prob"))
    early = f[f.close_error_cal_days >= 5]  # сезоны, когда путь закрылся раньше календаря
    C, F = "до (календарь)", "после (М1 + М3)"
    K, A = "М3 без М1 (календарь ± разброс)", "контроль: везти сразу на пределе"

    def hold(x):
        return round(x.hold_rub.mean() / x.delivered_t.clip(lower=1).mean())

    return pd.DataFrame([
        {"показатель": "Ошибка в дате закрытия за 14 дней до события", "единица": "сут",
         C: round(err_cal, 1), F: round(err_m1, 1), K: round(err_cal, 1), A: "—"},
        {"показатель": "Доля плана, вывезенная до закрытия", "единица": "%",
         **{col: round(100 * x.share_delivered.mean(), 1) for col, x in ((C, c), (F, f), (K, k), (A, a))}},
        {"показатель": "Недовезённый груз (среднее за сезон)", "единица": "т",
         **{col: round(x.undelivered_t.mean()) for col, x in ((C, c), (F, f), (K, k), (A, a))}},
        {"показатель": "Недовезённый груз в худший сезон", "единица": "т",
         **{col: round(x.undelivered_t.max()) for col, x in ((C, c), (F, f), (K, k), (A, a))}},
        {"показатель": "Привлечено дополнительной техники (среднее)", "единица": "т",
         **{col: round(x.extra_t.mean()) for col, x in ((C, c), (F, f), (K, k), (A, a))}},
        {"показатель": "Расходы на хранение при раннем завозе", "единица": "₽/т",
         **{col: hold(x) for col, x in ((C, c), (F, f), (K, k), (A, a))}},
        {"показатель": "Расходы на тонну с учётом дорогих каналов и хранения", "единица": "₽/т",
         **{col: round(x.cost_rub_per_t.mean()) for col, x in ((C, c), (F, f), (K, k), (A, a))}},
        {"показатель": "Заблаговременность предупреждения о раннем закрытии", "единица": "сут",
         C: 0, F: round(early.warning_lead_days.mean(), 1) if len(early) else np.nan, K: 0, A: "—"},
        {"показатель": "Сезонов в бэктесте / из них с ранним закрытием (≥5 сут)", "единица": "шт",
         C: c.hydro_year.nunique(), F: len(early), K: "—", A: "—"},
    ])


def sensitivity(bt, point_id, base: Params) -> pd.DataFrame:
    rows = []
    for name in SENSITIVE:
        for factor in (0.7, 1.0, 1.3):
            p = base.scaled(name, factor)
            r = run_example(bt, point_id, p)
            c, f, a, k = (r[r.rule == x] for x in ("calendar", "forecast", "asap", "clim_prob"))
            rows.append({"point_id": point_id, "параметр": name, "множитель": factor,
                         "₽/т до": round(c.cost_rub_per_t.mean()), "₽/т после": round(f.cost_rub_per_t.mean()),
                         "₽/т контроль": round(a.cost_rub_per_t.mean()), "₽/т М3 без М1": round(k.cost_rub_per_t.mean()),
                         "недовоз т до": round(c.undelivered_t.mean()), "недовоз т после": round(f.undelivered_t.mean()),
                         "после лучше": bool(f.cost_rub_per_t.mean() <= c.cost_rub_per_t.mean()),
                         "после лучше контроля": bool(f.cost_rub_per_t.mean() <= a.cost_rub_per_t.mean()),
                         "М1 добавляет к М3": bool(f.cost_rub_per_t.mean() <= k.cost_rub_per_t.mean())})
    return pd.DataFrame(rows)


def main():
    bt = pd.read_csv(REPORTS / "m1_backtest_close.csv")
    p = Params()
    lines = ["# «До/после» на прошлых сезонах", "",
             "Было: равномерный завоз до календарной даты закрытия (средняя за прошлые 10 сезонов). "
             "Стало: еженедельный пересчёт графика по прогнозу М1 (М3, Монте-Карло 1 000 сценариев). "
             "Фактические даты закрытия — реальные (гидропост / спутник). Экономические параметры — "
             "синтетика и допущения, см. таблицу чувствительности ±30%.", "",
             "Параметры: " + ", ".join(f"{k} = {v}" for k, v in vars(p).items() if k in SENSITIVE), ""]
    sens_all = []
    for key, ex in EXAMPLES.items():
        res = run_example(bt, ex["point_id"], p)
        res.to_csv(REPORTS / f"before_after_{key}.csv", index=False)
        summ = summarize(res, bt[bt.point_id == ex["point_id"]])
        summ.to_csv(REPORTS / f"before_after_{key}_summary.csv", index=False)
        sens = sensitivity(bt, ex["point_id"], p)
        sens_all.append(sens)
        robust = sens["после лучше"].mean()
        robust_ctrl = sens["после лучше контроля"].mean()
        m1_adds = sens["М1 добавляет к М3"].mean()
        print(ex["name"]); print(summ.to_string(index=False))
        print(f"  после лучше календаря в {robust:.0%}, лучше контроля в {robust_ctrl:.0%}, "
              f"М1 добавляет к М3 в {m1_adds:.0%} вариантов ±30%")
        per = res.pivot_table(index="season", columns="rule", values=["undelivered_t", "cost_rub_per_t"]).round(0)
        per.columns = [f"{a}_{b}" for a, b in per.columns]
        lines += [f"## {ex['name']}", "", md_table(summ), "",
                  f"Вывод «после не хуже, чем до» держится в **{robust:.0%}** вариантов, когда каждое допущение "
                  f"меняется на ±30%; «после не хуже, чем просто везти сразу» — в **{robust_ctrl:.0%}**.", "",
                  f"Вклад частей: прогноз М1 дешевле, чем М3 по одному календарному распределению, только в "
                  f"**{m1_adds:.0%}** вариантов. Для закрытия основной выигрыш даёт вероятностное планирование "
                  "(учёт разброса дат), а не точность М1 — она на закрытии на уровне календаря "
                  "(см. reports/m1_metrics.md).", "",
                  "<details><summary>По сезонам</summary>", "",
                  md_table(per.reset_index()), "", "</details>", ""]
    pd.concat(sens_all).to_csv(REPORTS / "sensitivity.csv", index=False)
    (REPORTS / "before_after.md").write_text("\n".join(lines), encoding="utf-8")
    print("готово:", REPORTS / "before_after.md")


if __name__ == "__main__":
    main()
