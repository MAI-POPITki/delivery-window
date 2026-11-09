"""
Польза модулей ветки ZXC на реальных данных.

    python scripts/zxc_value_check.py      # → reports/zxc_value.md, .csv

1. Узкое место коридора: на подтверждённых сезонах переправы Якутск (2019, 2023, 2024) и реальных ценах каждая
   мощность по очереди увеличивается на 10% (okno/bottlenecks.py); правило отправок — привычный календарный
   предзавоз. Сколько ожидания получателя (т·сут) снимает каждое звено.
2. Проверка массы машины: сколько дорог реестра (data/external/road_events.csv) дают проверяемый предел массы.
3. Обратная загрузка: только искусственные контрольные примеры (reports/backhaul_validation.md) — без реальных заявок.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from okno.bottlenecks import diagnose_corridor
from okno.corridor import CorridorConfig, simulate_corridor
from okno.corridor_data import case_inputs
from okno.paths import DATA, REPORTS
from okno.report import md_table

REAL = dict(rail_cost_per_ton=11003., crossing_cost_normal=373., crossing_cost_alt=25000.,
            yak_holding_cost_per_ton_day=22.3, nb_holding_cost_per_ton_day=22.3)


def main():
    cfg = replace(CorridorConfig(), **REAL)
    rows = []
    for y in (2019, 2023, 2024):
        c = case_inputs(y)
        base = simulate_corridor(cfg, c["ac"], c["ao"], c["cc"], c["co"])["shortage_ton_days"]
        for r in diagnose_corridor(cfg, c).itertuples():
            rows.append({"сезон": y, "звено": r.Участок, "мощность, т/сут": r.Мощность,
                         "ожидание без изменений, т·сут": round(base),
                         "сокращение ожидания при +10%, т·сут": round(r.Сокращение_ожидания_т_сут, 1),
                         "сокращение, %": round(100 * r.Сокращение_ожидания_т_сут / base, 1) if base else 0.0})
    d = pd.DataFrame(rows)
    d.to_csv(REPORTS / "zxc_value.csv", index=False)
    s = d.groupby("звено", sort=False)["сокращение, %"].agg(["min", "max"]).reset_index()
    s.columns = ["звено", "сокращение ожидания, % (мин)", "сокращение ожидания, % (макс)"]

    e = pd.read_csv(DATA / "external" / "road_events.csv")
    lim = e[e.kind.isin(["limit20", "limit30"])]
    reg = pd.DataFrame([{"событий в реестре": len(e), "дорог": e.road.nunique(),
                         "записей предела массы (20/30 т)": len(lim), "дорог с пределом массы": lim.road.nunique(),
                         "сезонов с пределом": lim.hydro_year.nunique()}])

    lines = ["# Польза модулей ZXC на реальных данных", "",
             "Метод — в шапке `scripts/zxc_value_check.py`. Цены реальные (ж/д 11 003, паром 373, СВП 25 000 ₽/т, "
             "хранение 22,3 ₽/т·сут); даты распутицы — подтверждённые 2019, 2023, 2024.", "",
             "## 1. Где узкое место доставки в распутицу", "", md_table(s), "", md_table(d), "",
             "**Вывод.** На всех трёх сезонах ожидание получателя снимает резервный канал через Лену (СВП) и штатная "
             "переправа, а не железная дорога: +10% мощности ж/д не меняет ничего. Стоимость приобретения мощности здесь "
             "не задана — это указание, куда смотреть, а не расчёт окупаемости.", "",
             "## 2. Проверка массы машины по реестру ограничений", "", md_table(reg), "",
             "Проверка «превышает ли машина опубликованный предел» работает только там, где предел записан в реестре. "
             "На остальных дорогах сервис честно пишет «предел неизвестен — уточните у владельца дороги» и не подставляет "
             "догадку. Пополнение реестра (приказы о 30/20 т) напрямую расширяет покрытие.", "",
             "## 3. Обратная загрузка", "",
             "Оптимизатор совмещения завоза и вывоза проверен на 7 искусственных примерах ([backhaul_validation]"
             "(backhaul_validation.md)): в контрольном примере пробег 4L → 2L, без обратного груза или при несовместимых "
             "кузовах выгоды нет (как и должно быть). Реальных пар заявок нет — коммерческая польза не доказана; "
             "модуль сохранён в коде, в интерфейс не выведен.", ""]
    (REPORTS / "zxc_value.md").write_text("\n".join(lines), encoding="utf-8")
    print(s.to_string(index=False))
    print(reg.to_string(index=False))


if __name__ == "__main__":
    main()
