"""
Главная проверка оптимизатора (М3) на реальных ценах: перевозка, хранение и цена ожидания получателя.

    python scripts/real_prices_check.py      # → reports/real_prices_check.md, .csv

Сравниваются два способа при ОДНИХ И ТЕХ ЖЕ календарных датах окна переправы Якутск — Нижний Бестях
(подтверждённые окна 2019, 2023, 2024): привычный предзавоз по календарю (эвристика) и экономический
оптимизатор по календарю. Прогноз дат здесь не участвует — проверяется только планирование.

Цены (источники — data/external/prices.csv):
* перевозка: ж/д 11 003 ₽/т (до ст. Беркакит, ТЭК-Трансойл), паром 373 ₽/т (тариф 2025), судно на
  воздушной подушке 25 000 ₽/т (тариф 2026) — tariffs.csv;
* хранение, варианты:
    - «топливо»: нефтебазы АО «Саханефтегазсбыт», 22,30 ₽/т·сут (2024) — на обоих берегах;
    - «сухие грузы»: портовый терминал, крытый склад 80 ₽/т·сут (Якутск), открытая площадка с укрытием
      50 ₽/т·сут (Нижний Бестях); надбавки за долгое хранение не учтены — это нижняя оценка;
    - «допущение команды»: 220 / 450 ₽/т·сут (как в economic_planner);
* цена дня ожидания получателя — сетка: 49 ₽/т·сут (неустойка 44-ФЗ: 1/300 ключевой ставки 14% от цены
  дизельного топлива ≈ 105 тыс. ₽/т), 300, 1 000, 3 000, 12 000 ₽/т·сут (последнее — допущение команды).

Выигрыш честный, только если оптимизатор дешевле И получатель ждёт не дольше. Поэтому показываются оба.
"""

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import corridor_simulator as cs
from okno.corridor import CorridorConfig
from okno.report import md_table

TRANSPORT = dict(rail_cost_per_ton=11003., crossing_cost_normal=373., crossing_cost_alt=25000.)
STORAGE = {"топливо (нефтебазы, 22,3 ₽)": (22.3, 22.3),
           "топливо, прочие клиенты (нефтебазы СНГС 2026, 44,4 ₽)": (44.44, 44.44),
           "сухие грузы (терминал, 80 / 50 ₽)": (80., 50.),
           # Якутск: тёплый склад 25 ₽ (ЦИАН); Нижний Бестях: груз застрял в вагоне —
           # простой вагона 50 ₽ + сбор станции за хранение 10,4 ₽ (prices.csv)
           "склад Якутска / застрявший вагон в Н. Бестяхе (25 / 60 ₽)": (25., 60.4),
           # Якутск: ответственное хранение ЯТА до 120 ₽; Н. Бестях: контейнер на терминале 45,6 ₽
           "ответ. хранение Якутск / контейнер Н. Бестях (120 / 46 ₽)": (120., 45.56),
           "допущение команды (220 / 450 ₽)": (220., 450.)}
# 49 и 67 — неустойка 44-ФЗ по дизелю (цена 2026 г. Росстат / АЗС СНГС); 200 — простой грузовика (ТК, Якутск)
WAIT = [49., 67., 200., 1000., 3000., 12000.]


def main():
    rows = []
    for sname, (yak, nb) in STORAGE.items():
        for pen in WAIT:
            cfg = replace(CorridorConfig(), **TRANSPORT, yak_holding_cost_per_ton_day=yak,
                          nb_holding_cost_per_ton_day=nb, shortage_penalty_per_ton_day=pen)
            h = cs.run_verified_backtest(cfg)
            for y, g in h.groupby("hydro_year"):
                g = g.set_index("policy")
                rows.append({"хранение": sname, "ожидание, ₽/т·сут": pen, "сезон": int(y),
                             "эвристика, млн ₽": g.loc["calendar", "total_cost_rub"] / 1e6,
                             "оптимизатор, млн ₽": g.loc["economic_calendar", "total_cost_rub"] / 1e6,
                             "ожидание эвристика, т·сут": g.loc["calendar", "shortage_ton_days"],
                             "ожидание оптимизатор, т·сут": g.loc["economic_calendar", "shortage_ton_days"],
                             "резерв эвристика, т": g.loc["calendar", "tons_alt_transport"],
                             "резерв оптимизатор, т": g.loc["economic_calendar", "tons_alt_transport"]})
    d = pd.DataFrame(rows)
    d["экономия, %"] = 100 * (1 - d["оптимизатор, млн ₽"] / d["эвристика, млн ₽"])
    d["ожидание не дольше"] = d["ожидание оптимизатор, т·сут"] <= d["ожидание эвристика, т·сут"] + 1e-6
    d.to_csv(ROOT / "reports" / "real_prices_check.csv", index=False)
    s = d.groupby(["хранение", "ожидание, ₽/т·сут"]).agg(
        **{"экономия, % (мин–макс по сезонам)": ("экономия, %", lambda x: f"{x.min():.1f} … {x.max():.1f}"),
           "ожидание эвристика, т·сут": ("ожидание эвристика, т·сут", "mean"),
           "ожидание оптимизатор, т·сут": ("ожидание оптимизатор, т·сут", "mean"),
           "ожидание не дольше (сезонов из 3)": ("ожидание не дольше", "sum")}).reset_index()
    s["ожидание эвристика, т·сут"] = s["ожидание эвристика, т·сут"].round(0)
    s["ожидание оптимизатор, т·сут"] = s["ожидание оптимизатор, т·сут"].round(0)
    honest = d[d["ожидание не дольше"]]
    lines = ["# Оптимизатор на реальных ценах: перевозка, хранение, ожидание", "",
             "Одни и те же календарные даты окна переправы Якутск (подтверждённые окна 2019, 2023, 2024); "
             "сравниваются привычный предзавоз (эвристика) и экономический оптимизатор. Цены и источники — "
             "в шапке `scripts/real_prices_check.py` и в [prices.csv](../data/external/prices.csv). "
             "Экономия > 0 — оптимизатор дешевле. Честный выигрыш — когда он дешевле и получатель ждёт не дольше.", "",
             md_table(s), "",
             f"**Честных выигрышей (дешевле и ожидание не дольше): {len(honest)} из {len(d)} сочетаний; "
             f"экономия в них — от {honest['экономия, %'].min():.1f}% до {honest['экономия, %'].max():.1f}%.**"
             if len(honest) else "**Честных выигрышей нет.**", ""]
    # режим «сервис»: минимум расходов оператора при заданном ожидании (цена ожидания в расходы не входит)
    from okno.corridor import service_frontier, service_plan
    from okno.corridor_data import case_inputs
    srows = []
    for sname, (yak, nb) in ((k, v) for k, v in STORAGE.items() if not k.startswith("допущение")):
        cfg = replace(CorridorConfig(), **TRANSPORT, yak_holding_cost_per_ton_day=yak, nb_holding_cost_per_ton_day=nb)
        for y in (2019, 2023, 2024):
            c = case_inputs(y)
            fr, base = service_frontier(cfg, c)
            row = {"хранение": sname, "сезон": y, "ожидание привычной схемы, т·сут": base["shortage_ton_days"]}
            for share, label in ((1.0, "при том же ожидании"), (0.5, "при вдвое меньшем ожидании")):
                plan, _ = service_plan(cfg, c, base["shortage_ton_days"] * share, frontier=fr)
                row[f"расходы {label}, %"] = round(100 * (plan["operating_cost_rub"] / base["operating_cost_rub"] - 1), 1)
            srows.append(row)
    sv = pd.DataFrame(srows)
    lines += ["## Режим «сервис»: без допущения о цене ожидания", "",
              "Оптимизатор ищет минимум расходов на перевозку и хранение при ожидании получателя не больше, чем у "
              "привычной схемы (или вдвое меньше). Внутренняя цена ожидания — только рычаг, в расходы не входит. "
              "Отрицательное — дешевле привычной схемы.", "", md_table(sv), ""]
    (ROOT / "reports" / "real_prices_check.md").write_text("\n".join(lines), encoding="utf-8")
    print(s.to_string(index=False))
    print(lines[-2])


if __name__ == "__main__":
    main()
