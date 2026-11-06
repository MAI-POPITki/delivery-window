"""Разбор заказа на 45 т: python scripts/nizhneyansk_batagay.py.

Ожидаемые расходы по распределению М1, а не факт исполнения заказа.
Расчёт с дедлайном использует только полностью завершённые недельные интервалы.
"""

import datetime as dt
from dataclasses import fields
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.examples import NIZHNEYANSK_BATAGAY as SETTINGS
from okno import forecast as FC
from okno.ice import from_season_day, to_season_day
from okno.paths import REPORTS
from okno.planner import Params, SENSITIVE, WEEK, calendar_plan, plan_week_volumes, sample_quantiles
from okno.report import md_table


def inputs():
    p = Params(**{f.name: SETTINGS[f.name] for f in fields(Params) if f.name in SETTINGS})
    hy = SETTINGS["hydro_year"]
    issue = int(to_season_day(SETTINGS["decision_date"], hy))
    q = FC.forecast(SETTINGS["point_id"], hy, issue, "close", use_weather=False)
    if not q:
        raise ValueError("Нет прогноза для примера")
    return p, hy, issue, q


def compare(p, issue, q, deadline_day=None):
    weeks = np.arange(max(issue, p.start_day), 300, WEEK, dtype=float)
    if deadline_day is not None:
        weeks = weeks[weeks + WEEK <= deadline_day]
    if not len(weeks):
        raise ValueError("До срока заказа нет полностью завершённой недели")
    sc = sample_quantiles(q["q10"], q["q50"], q["q90"], p.n_scenarios,
                          np.random.default_rng(p.seed)) - p.close_margin_days
    sc = np.maximum(sc, weeks[0])
    risk = np.array([(sc < w + WEEK).mean() for w in weeks])
    cal_close = q["calendar"] - p.close_margin_days
    if deadline_day is not None:
        cal_close = min(cal_close, deadline_day)
    # Та же опорная дата хранения, что в дашборде. Не фактический срок хранения.
    reference = q["calendar"] - p.close_margin_days + 14
    plans = {
        "calendar": calendar_plan(weeks, cal_close, p),
        "forecast": plan_week_volumes(weeks, risk, p.Q_t, p, reference),
    }
    rows, schedules = [], []
    for rule, (base, extra) in plans.items():
        volume = base + extra
        success = 1 - risk
        shortage = float((risk * volume).sum() + max(0.0, p.Q_t - volume.sum()))
        road = float((success * volume).sum() * p.c_road)
        surcharge = float((success * extra).sum() * p.c_extra)
        holding = float((success * volume * np.maximum(reference - weeks, 0) / WEEK).sum()
                        * p.c_hold_week)
        expensive = shortage * p.c_exp
        rows.append({"rule": rule, "base_t": float(base.sum()), "extra_t": float(extra.sum()),
                     "expected_road_t": p.Q_t - shortage, "expected_shortage_t": shortage,
                     "road_rub": road, "extra_rub": surcharge, "holding_rub": holding,
                     "expensive_rub": expensive, "total_rub": road + surcharge + holding + expensive})
        for week, b, e, prob in zip(weeks, base, extra, risk):
            if b + e:
                schedules.append({"rule": rule, "week_day": week, "completion_day": week + WEEK,
                                  "base_t": b, "extra_t": e, "p_closed": prob})
    return pd.DataFrame(rows), pd.DataFrame(schedules)


def main():
    p, hy, issue, q = inputs()
    deadline = int(to_season_day(dt.date(2024, 3, 31), hy))
    summaries, schedules, sensitivity = [], [], []
    for case, cutoff in (("dashboard", None), ("order_deadline", deadline)):
        result, schedule = compare(p, issue, q, cutoff)
        result.insert(0, "case", case)
        schedule.insert(0, "case", case)
        for col in ("week_day", "completion_day"):
            schedule[col.replace("_day", "_date")] = schedule[col].map(
                lambda d: from_season_day(d, hy).date())
        summaries.append(result)
        schedules.append(schedule)
        for name in SENSITIVE:
            for factor in (0.7, 1.0, 1.3):
                res, _ = compare(p.scaled(name, factor), issue, q, cutoff)
                costs = res.set_index("rule").total_rub
                sensitivity.append({"case": case, "parameter": name, "factor": factor,
                                    "calendar_rub": costs.calendar, "forecast_rub": costs.forecast,
                                    "saving_rub": costs.calendar - costs.forecast})
    totals = pd.concat(summaries, ignore_index=True)
    schedule = pd.concat(schedules, ignore_index=True)
    prefix = REPORTS / "nizhneyansk_batagay_45t"
    totals.to_csv(f"{prefix}.csv", index=False)
    schedule.to_csv(f"{prefix}_schedule.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(f"{prefix}_sensitivity.csv", index=False)

    table = totals.rename(columns={"case": "расчёт", "rule": "план", "base_t": "своя техника, т",
                                   "extra_t": "дополнительная, т", "expected_road_t": "ожидаемо зимником, т",
                                   "expected_shortage_t": "ожидаемо дорогим каналом, т",
                                   "road_rub": "зимник, ₽", "extra_rub": "надбавка, ₽",
                                   "holding_rub": "хранение, ₽", "expensive_rub": "дорогой канал, ₽",
                                   "total_rub": "всего, ₽"}).round(2)
    text = """# Нижнеянск → Батагай: проверка заказа на 45 тонн

Реальная экономия на этом заказе не доказана. При одинаковом сроке до 31 марта
недельная модель даёт одинаковые планы по календарю и прогнозу: 28 т своей техникой,
17 т дополнительной. Расчётная разница расходов — 0 ₽.

## Исходные данные

Регион — Якутия, сезон 2023/24, решение на 22.03.2024. Точка gauge_3414 —
гидропост «Яна — Верхоянск», приближение для направления, а не весь маршрут.
45 т считаем ещё не вывезенными: фактический остаток неизвестен.

| Параметр | Значение | Основание |
| --- | --- | --- |
| План заказа | 45 т | Закупка №1089698, не факт доставки |
| Зимник | 22 447,50 ₽/т | 750 км × 29,93 ₽/т·км, начальная цена |
| Хранение | 149,80 ₽/т·нед | Официальный тариф 2024 г. 21,40 ₽/т·сут × 7; применение к заказу не подтверждено |
| Своя техника | 28 т/нед | Допущение: 3 × 8 т × 7 / 6 суток |
| Дополнительная техника | 18,7 т/нед | Допущение: 2 × 8 т × 7 / 6 суток, округлено |
| Надбавка | 4 489,50 ₽/т | Допущение: +20% к тарифу зимника |
| Авиация | 500 000 ₽/т | Сценарий, не котировка маршрута и не подтверждение возможности перевозки топлива |
| Запас до таяния | 21 сутки | Допущение, не официальная дата закрытия |

Все цены без НДС. Минимум три автомобиля в документации — требование к участнику,
а не подтверждение фактической производительности. Тип и загрузка машин, оборот 6 суток,
реальный график, даты закрытия маршрута и итоговые расходы неизвестны.

## Что считает приложение

Календарная оценка закрытия с запасом — 19 апреля; медиана модели — 17 апреля,
диапазон q10–q90 — 11–27 апреля. Это модельные оценки, не разрешение движения.
При 1 000 сценариев и seed=42 календарь назначает по 11,25 т на 22 и 29 марта,
5 и 12 апреля. Модель назначает 17 т на 22 марта и 28 т на 29 марта.
Недельные интервалы заканчиваются через семь суток после этих дат.

В приложении календарный план рискует не довезти 7,245 т, модельный — 0 т.
Нулевой риск означает отсутствие неудач в выбранном распределении, а не гарантию.
Разница ожидаемых затрат получается 3 450 417,80 ₽ (76,72%).
Однако календарный план нарушает срок заказа, а модельный интервал 29 марта — 5 апреля
тоже заканчивается после 31 марта. Эти 76,72% нельзя выдавать за выгоду по заказу.

## Сравнение с одинаковым сроком

Оставляем только полностью завершённые недельные интервалы до 31 марта:
доступна неделя 22–29 марта. Частичная неделя не моделируется.
Оба плана используют 28 т своей и 17 т дополнительной техники:

- Базовая перевозка: 45 × 22 447,50 = 1 010 137,50 ₽.
- Надбавка: 17 × 4 489,50 = 76 321,50 ₽.
- Вместе без хранения: 1 086 459 ₽ в обоих планах.
- Хранение по условной опорной дате приложения 3 мая: 41 120,10 ₽ в обоих планах.
- Итого с этим хранением: 1 127 579,10 ₽ в обоих планах.

Опорная дата хранения не подтверждена договором. Дополнительная надбавка также
не предусмотрена подтверждённой ценой заказа: это параметр сценария.
Авиатариф не влияет на базовый вариант со сроком: весь объём помещается в зимник.
При отдельном изменении каждого из восьми параметров на ±30% разница двух планов
с одинаковым сроком остаётся нулевой (24 расчёта, включая повторения базового варианта).
Это проверка данного сценария, не статистическое доказательство для других заказов.

## Что это говорит об идее

Маленький заказ, который помещается в одну неделю при привлечении техники, не
демонстрирует преимущества прогноза сезонного закрытия. Здесь результат определяет
срок договора и доступность машин. Потенциальная польза прогноза — для большого
сезонного плана или отправок вблизи закрытия, когда дата окончания окна меняет решение.
Для подтверждения нужны реальные недельные отправки, остатки, производительность,
официальные даты ограничений по участкам и фактические затраты.

Это оценка ожидаемой стоимости фиксированных планов на дату решения, не динамический
бэктест с еженедельным обновлением. Фактического закрытия gauge_3414 за 2023/24 в данных
проекта нет. Проверить реальную доставку и экономию по этому сезону пока нельзя.

## Источники и воспроизведение

- [Закупка №1089698](https://market.otc.ru/ProductRequestGroup/Index/1089698): план и документация, не подтверждение исполнения.
- [План перевозки XLSX](https://tender.otc.ru/main/Sites/FileServiceSite/FileDownloadHandler.ashx?FileGuid=9c9c79e4-94e8-4fec-869d-f59970ed0c41).
- [Техническое задание DOCX](https://tender.otc.ru/main/Sites/FileServiceSite/FileDownloadHandler.ashx?FileGuid=7db06590-d3c0-4402-b06a-2fbb9b4f22e5).
- [Тарифы хранения с 01.01.2024, Саханефтегазсбыт](https://aosngs.ru/upload/Klientam/2023/31_%D0%A6%D0%B5%D0%BD%D1%8B%20%D0%BD%D0%B0%20%D1%83%D1%81%D0%BB%D1%83%D0%B3%D0%B8%20%D0%9F%D0%A1%D0%9E%20%D0%B8%20%D1%85%D1%80%D0%B0%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%82%D0%BE%D0%BF%D0%BB%D0%B8%D0%B2%D0%B0%20%2801.01.2024%D0%B3.%29.pdf).

Запуск: `.venv/bin/python scripts/nizhneyansk_batagay.py`.
Кнопка «Пример: Нижнеянск → Батагай, 45 т» загружает те же значения в дашборд.
CSV рядом с отчётом содержат полный расчёт, расписания и чувствительность.

## Численные результаты

"""
    Path(f"{prefix}.md").write_text(text + md_table(table) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print("Отчёт:", f"{prefix}.md")


if __name__ == "__main__":
    main()
