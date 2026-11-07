"""Reviewable scenario replay, separate from the ordinary weekly planner."""

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from okno.yakutsk_replay import RULE_LABELS as NAMES, Scenario, capacity_table, simulate_all

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "scenarios"
PREFIX = "yakutsk_spring_2026"


@st.cache_data(show_spinner=False)
def run_comparison(config, version):
    f = pd.read_csv(INPUTS / f"{PREFIX}_forecasts.csv")
    o = pd.read_csv(INPUTS / f"{PREFIX}_observations.csv")
    return simulate_all(f, o, Scenario(**config))


def render_experiment():
    st.markdown("**Нижний Бестях → Якутск · 16 марта — 15 апреля 2026 года**")
    st.info("Сценарный эксперимент: ограничения движения и спутниковые наблюдения реальные; "
            "заказ, парк, оборот машин и расходы — согласованные допущения. "
            "Результат не является подтверждённой экономией перевозчика.")
    c1, c2, c3 = st.columns(3)
    values = {
        "Q_t": c1.number_input("Остаток заказа, т", 0.1, 100000., 5000., 100., key="replay_Q"),
        "base_trucks": c1.number_input("Основной парк, машин", 1, 100, 10, key="replay_base"),
        "extra_trucks": c1.number_input("Дополнительный парк, машин", 0, 100, 5, key="replay_extra"),
        "tare_t": c2.number_input("Масса пустой машины, т", 0., 39., 12., 1., key="replay_tare"),
        "max_payload_t": c2.number_input("Максимальная полезная загрузка, т", .1, 100., 28., 1., key="replay_payload"),
        "trips_per_workday": c2.number_input("Рейсов на машину за рабочий день", .1, 10., 1., .1, key="replay_trips"),
        "c_road": c3.number_input("Обычная доставка, ₽/т", 0., 200000., 4000., 100., key="replay_road"),
        "c_extra": c3.number_input("Надбавка за дополнительную машину, ₽/т", 0., 200000., 1500., 100., key="replay_extra_cost"),
        "c_backup": c3.number_input("Резервная доставка и издержки, ₽/т", 1., 1000000., 20000., 100., key="replay_backup"),
        "c_hold_week": c3.number_input("Хранение, ₽/т·неделю", 0., 10000., 100., 10., key="replay_hold"),
    }
    st.caption("Работа: понедельник — пятница; рейс завершается в тот же рабочий день. "
               "Груз нужен к 16 апреля. Резерв приходит 26 мая (допущение). "
               "Хранение считается за календарные дни; ставка резерва включает его доставку и связанные издержки.")
    s = Scenario(**values)
    cap = capacity_table(s)
    st.dataframe(pd.DataFrame({
        "Событие": ["Полная масса ≤30 т", "Полная масса ≤20 т", "Последний день проезда"],
        "Объявлено в начале марта": ["01.04.2026", "08.04.2026", "15.04.2026"],
        "Фактически": ["27.03.2026", "08.04.2026", "15.04.2026, до вечернего закрытия"],
    }), hide_index=True, width="stretch", key="replay_restrictions")
    st.caption("В предварительном объявлении есть дата закрытия 15 апреля без точного часа. "
               "В расчёте этот день включён в предварительное окно; фактическое вечернее закрытие подтверждено.")
    st.caption(f"Полезная загрузка при лимитах 40/30/20 т: "
               f"{min(s.max_payload_t, max(0, 40-s.tare_t)):g} / "
               f"{min(s.max_payload_t, max(0, 30-s.tare_t)):g} / "
               f"{min(s.max_payload_t, max(0, 20-s.tare_t)):g} т. "
               f"Вместимость основного парка за весь период: {cap.base_capacity_t.sum():,.0f} т.")
    paths = [INPUTS / f"{PREFIX}_{name}.csv" for name in ("forecasts", "observations")]
    version = tuple(p.stat().st_mtime_ns for p in paths + [ROOT / "okno/yakutsk_replay.py"])
    with st.spinner("Пересчитываем отправки по дням…"):
        results = run_comparison(asdict(s), version)
    summary = pd.DataFrame([r.summary for r in results.values()]).set_index("rule")
    view = summary[["delivered_t", "backup_t", "extra_t", "cost_rub", "cost_rub_per_t", "completion_date"]].copy()
    view.index = view.index.map(NAMES)
    view.index.name = "Способ"
    view.columns = ["По зимней дороге, т", "Резерв, т", "Доп. техникой, т", "Расходы, ₽", "Расходы, ₽/т", "Заказ доставлен"]
    st.dataframe(view.round(2), width="stretch", key="replay_summary", column_config={
        col: st.column_config.NumberColumn(format="localized") for col in view.columns[:-1]})
    difference = summary.loc["forecast_m2", "cost_rub"] - summary.loc["calendar", "cost_rub"]
    percent = difference / summary.loc["calendar", "cost_rub"] * 100 if summary.loc["calendar", "cost_rub"] else 0
    st.metric("Разница расходов: наш метод относительно календаря", f"{difference:+,.0f} ₽".replace(",", " "),
              delta=f"{percent:+.2f}%", delta_color="inverse")
    if difference > .01:
        st.info("При этих параметрах календарь дешевле. Прогноз раньше привлекает дополнительные машины "
                "и увеличивает хранение; смотрите отдельно доставленный объём и расходы.")
    elif difference < -.01:
        st.success("В этом условном сценарии наш метод дешевле календаря. Проверьте устойчивость результата к параметрам.")
    else:
        st.info("Расходы методов в этом сценарии совпадают.")
    model = summary.loc["forecast_m2"]
    matching_controls = [NAMES[rule] for rule in ("climatology", "asap")
                         if abs(summary.loc[rule, "cost_rub"] - model.cost_rub) < .01
                         and abs(summary.loc[rule, "delivered_t"] - model.delivered_t) < .001]
    if matching_controls:
        st.caption("Те же объём и расходы получились без М1 у способов: "
                   + "; ".join(matching_controls) + ". Отдельный выигрыш прогноза здесь не доказан.")
    if results["forecast"].schedule.equals(results["forecast_m2"].schedule.assign(rule="forecast")):
        st.caption("Спутниковая поправка М2 в этом расчёте не изменила отправки и расходы.")
    st.markdown("Календарный метод каждый день пересчитывает остаток и потребность в дополнительных машинах. "
                "В день изменения лимита все методы видят новое ограничение. Будущие фактические даты им неизвестны.")
    fig = go.Figure()
    for rule, result in results.items():
        fig.add_scatter(x=result.schedule.date, y=result.schedule.delivered_t.cumsum(), name=NAMES[rule],
                        mode="lines", line=dict(dash="dash" if rule in ("forecast_m2", "asap") else "solid"))
    fig.add_hline(y=s.Q_t, line_dash="dot", annotation_text="заказ")
    fig.update_layout(title="Накопленная доставка по ледовой дороге", yaxis_title="Тонны", height=420,
                      legend=dict(orientation="h", y=-.25))
    st.plotly_chart(fig, width="stretch", key="replay_deliveries")
    costs = go.Figure()
    for col, name in (("road_rub", "Доставка"), ("extra_rub", "Дополнительная техника"),
                      ("hold_rub", "Хранение"), ("backup_rub", "Резерв")):
        costs.add_bar(x=[NAMES[r] for r in summary.index], y=summary[col]/1e6, name=name)
    costs.update_layout(barmode="stack", title="Из чего складываются расходы", yaxis_title="млн ₽", height=400)
    st.plotly_chart(costs, width="stretch", key="replay_costs")
    with st.expander("Спутниковые снимки и доступность данных"):
        st.write("Это обзоры целых радарных сцен, цвета условные. Оценка «лёд/вода» относится к точке "
                 "переправы и не определяет толщину льда или разрешённую массу автомобиля.")
        obs = pd.read_csv(paths[1])
        st.dataframe(obs[["date", "available_date", "p_ice"]].rename(columns={
            "date": "Съёмка", "available_date": "Первое решение со снимком", "p_ice": "P(лёд)"}),
            hide_index=True, width="stretch")
        cols = st.columns(2)
        for col, day, available in zip(cols, ("20260318", "20260411"), ("20 марта", "13 апреля")):
            image_path = ROOT / "docs/replay_yakutsk_2026" / f"{day}_scene_preview.png"
            if image_path.exists():
                col.image(str(image_path), caption=f"Sentinel‑1 · {day[6:]}.{day[4:6]}.2026 · доступен с {available}")
        st.caption("Источник: ESA/Copernicus Sentinel‑1; обработка RTC — Catalyst, каталог и обзорные изображения — "
                   "Microsoft Planetary Computer. Исходные VV/VH вырезки не загружены: используются сохранённые "
                   "признаки и повторно проверенные оценки М2. Задержка публикации 24 часа и свежесть 14 дней — допущения.")
        st.write("Погода — архивный реанализ ERA5 с задержкой пять суток. Модели обучены на целях до 2024 года; "
                 "это реконструкция текущим кодом, а не прогноз, реально выпущенный в марте 2026 года.")
        st.write("М1 предсказывает физическое вскрытие. Запас 21 сутки и переходы к ограничениям за 14/7 суток "
                 "до закрытия — гипотезы, проверенные отдельно в чувствительности отчёта.")
    schedules = pd.concat([r.schedule for r in results.values()], ignore_index=True)
    st.download_button("Скачать графики текущего расчёта (CSV)", schedules.to_csv(index=False).encode("utf-8-sig"),
                       "yakutsk-spring-2026-schedules.csv", "text/csv", key="replay_download_schedule")
    report = ROOT / "reports/yakutsk_spring_2026.md"
    if report.exists():
        st.download_button("Скачать полный отчёт для исходных параметров", report.read_bytes(),
                           "yakutsk-spring-2026-report.md", "text/markdown", key="replay_download_report")
        st.caption("Таблица выше пересчитывается при изменении параметров. Полный отчёт и чувствительность "
                   "относятся к исходному согласованному сценарию: 5 000 т, 10 + 5 машин.")
