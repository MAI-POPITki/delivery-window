"""
Дашборд «Окно завоза».

    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from okno import forecast as FC
from okno.ice import from_season_day, to_season_day
from okno.paths import PROCESSED, REPORTS
from okno.planner import Params, calendar_plan, plan_week_volumes, sample_quantiles, simulate_season

st.set_page_config(page_title="Окно завоза", page_icon="🧊", layout="wide")

# ------------------------------------------------------------------ данные
WEATHER_POINTS = ["yakutsk_crossing"] + [
    f"gauge_{c}" for c in pd.read_csv(PROCESSED / "gauges.csv").query("yakutia").gauge_code
    if (Path(PROCESSED).parent / "weather" / "gauges" / f"gauge_{c}.csv").exists()
]
LABELS = {"yakutsk_crossing": "Переправа Якутск — Нижний Бестях (Лена)",
          "gauge_3414": "Зимник на Батагай · Яна — Верхоянск"}


def label(pid):
    if pid in LABELS:
        return LABELS[pid]
    return "Гидропост · " + FC.point_meta(pid)["name"].replace(" At ", " — ")


def season_name(hy):
    return f"{hy - 1}/{str(hy)[2:]}"


@st.cache_data(show_spinner=False)
def fc_cached(pid, hy, day, kind):
    try:
        return FC.forecast(pid, hy, day, kind)
    except Exception as e:  # noqa: BLE001 — в дашборде показываем, а не падаем
        return {"error": str(e)}


@st.cache_data(show_spinner=False)
def fc_path(pid, hy, kind, days):
    rows = [fc_cached(pid, hy, d, kind) for d in days]
    return pd.DataFrame([r for r in rows if r and "error" not in r])


def p_by(q, t):
    """P(событие ≤ t) из квантилей М1 (кусочно-линейно)."""
    if q is None or "error" in q or np.isnan(q.get("q50", np.nan)):
        return np.nan
    s = sample_quantiles(q["q10"], q["q50"], q["q90"], 4000, np.random.default_rng(0))
    return float((s <= t).mean())


# ------------------------------------------------------------------ боковая панель
st.sidebar.title("🧊 Окно завоза")
st.sidebar.caption("Прогноз окон сезонных путей Якутии и график завоза. Открытые данные, без датчиков.")
today = dt.date.today()
cur_hy = today.year + (1 if today.month >= 9 else 0)
hy = st.sidebar.selectbox("Сезон", list(range(cur_hy, 1985, -1)), format_func=season_name, index=0)
season_start = dt.date(hy - 1, 9, 1)
default_date = min(today, dt.date(hy, 5, 31)) if hy == cur_hy else dt.date(hy, 3, 1)
issue_date = st.sidebar.date_input("Дата решения", value=default_date,
                                   min_value=dt.date(hy - 1, 10, 1), max_value=dt.date(hy, 6, 15))
issue_day = int(to_season_day(issue_date, hy))
pid = st.sidebar.selectbox("Путь", WEATHER_POINTS, format_func=label,
                           index=WEATHER_POINTS.index("gauge_3414") if "gauge_3414" in WEATHER_POINTS else 0)
horizon = st.sidebar.slider("Горизонт N, дней", 7, 60, 14, 7)
margin = st.sidebar.slider("Запас до таяния при закрытии, сут", 0, 35, 21,
                           help="Путь закрывают раньше физического начала таяния: ограничения массы, "
                                "дорожный режим. Допущение, проверяется на чувствительность.")
last = FC.last_weather_day(pid, hy)
if last is not None and last < issue_day:
    st.sidebar.warning(f"Погода есть до {from_season_day(last, hy).date():%d.%m.%Y}: прогноз выпускается на эту дату.")

tabs = st.tabs(["Карта путей", "Карточка пути", "Планировщик", "Рейтинг риска", "До / после", "О данных"])

# ------------------------------------------------------------------ 1. карта
with tabs[0]:
    st.subheader(f"Вероятность, что путь открыт через {horizon} дн. — {season_name(hy)}")
    t = issue_day + horizon
    rows = []
    for p in WEATHER_POINTS:
        meta = FC.point_meta(p)
        qo, qc = fc_cached(p, hy, issue_day, "open"), fc_cached(p, hy, issue_day, "close")
        p_open = p_by(qo, t)
        p_open = 1.0 if np.isnan(p_open) and issue_day > 120 else p_open
        p_closed = p_by(qc, t + margin)
        p_ok = p_open * (1 - p_closed) if not (np.isnan(p_open) or np.isnan(p_closed)) else np.nan
        color = ("#9e9e9e" if np.isnan(p_ok) else
                 "#2e7d32" if p_ok >= 0.8 else "#ef8c00" if p_ok >= 0.5 else "#c62828")
        txt = f"<b>{label(p)}</b><br>P(открыт через {horizon} дн.) = " + ("н/д" if np.isnan(p_ok) else f"{p_ok:.0%}")
        if qc and "error" not in qc:
            txt += f"<br>Таяние (М1, 50%): {qc['q50_date']:%d.%m}, диапазон {qc['q10_date']:%d.%m}–{qc['q90_date']:%d.%m}"
        rows.append({"lat": meta["lat"], "lon": meta["lon"], "color": color, "txt": txt})
    pts = pd.DataFrame(rows)
    # Plotly + MapLibre GL (открытый движок, рисует на видеокарте); подложка CARTO по данным OSM
    fmap = go.Figure(go.Scattermap(
        lat=pts.lat, lon=pts.lon, mode="markers", text=pts.txt, hoverinfo="text",
        marker=dict(size=16, color=pts.color, opacity=0.9)))
    fmap.update_layout(map=dict(style="carto-voyager", center=dict(lat=65, lon=132), zoom=2.6),
                       margin=dict(l=0, r=0, t=0, b=0), height=520, showlegend=False)
    st.plotly_chart(fmap, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    st.caption("Цвет: зелёный ≥ 80%, оранжевый 50–80%, красный < 50%, серый — нет данных для прогноза. "
               "Вероятность = P(лёд уже держит 20 т) × P(путь ещё не закрыт).")

# ------------------------------------------------------------------ 2. карточка пути
with tabs[1]:
    st.subheader(label(pid))
    c1, c2 = st.columns(2)
    for kind, col, title in (("open", c1, "Открытие: лёд держит 20 т"), ("close", c2, "Закрытие: начало таяния")):
        q = fc_cached(pid, hy, issue_day, kind)
        with col:
            st.markdown(f"**{title}**")
            if not q or "error" in q:
                st.info("Нет прогноза для этой точки и вида события"
                        + (" (у переправы нет замеров толщины — только спутниковые даты вскрытия)."
                           if pid == "yakutsk_crossing" and kind == "open" else "."))
                continue
            close_note = f" (закрытие пути ≈ на {margin} сут раньше)" if kind == "close" else ""
            st.metric("Медиана прогноза М1", f"{q['q50_date']:%d.%m.%Y}",
                      delta=f"{q['q50'] - q['calendar']:+.0f} сут к календарю", delta_color="inverse")
            st.write(f"80%-диапазон: **{q['q10_date']:%d.%m} – {q['q90_date']:%d.%m}**{close_note}")
            st.write(f"Календарь (средняя за прошлые сезоны): {q['calendar_date']:%d.%m}; "
                     f"физика: {q['physics_date']:%d.%m}")
            st.write(f"Расчётная толщина льда на дату решения: **{q['h_stefan_now_cm']:.0f} см**")
            if q["actual_date"]:
                st.success(f"Факт: {q['actual_date']:%d.%m.%Y}")
    # как менялся прогноз закрытия в течение сезона
    days = list(range(150, 271, 7))
    path = fc_path(pid, hy, "close", tuple(days))
    if not path.empty:
        x = [from_season_day(d, hy) for d in path.issue_day]
        f = go.Figure()
        f.add_trace(go.Scatter(x=x + x[::-1], y=[from_season_day(v, hy) for v in list(path.q90) + list(path.q10[::-1])],
                               fill="toself", line=dict(width=0), name="80%-диапазон М1", opacity=0.3))
        f.add_trace(go.Scatter(x=x, y=[from_season_day(v, hy) for v in path.q50], name="М1, медиана"))
        f.add_trace(go.Scatter(x=x, y=[from_season_day(v, hy) for v in path.calendar], name="Календарь",
                               line=dict(dash="dash")))
        if not np.isnan(path.actual.iloc[0]):
            f.add_hline(y=from_season_day(path.actual.iloc[0], hy), line_color="black", annotation_text="факт")
        f.update_layout(title="Как менялся прогноз начала таяния в течение сезона", xaxis_title="дата решения",
                        yaxis_title="прогноз даты", height=420)
        st.plotly_chart(f, width="stretch")
    hist = FC.history(pid)
    if hist["close_day"].notna().any():
        h = hist.dropna(subset=["close_day"])
        f2 = go.Figure(go.Bar(x=h.index, y=h.close_day, name="начало таяния (день сезона)"))
        f2.update_layout(title="История прошлых сезонов: начало таяния, дней от 1 сентября", height=300)
        st.plotly_chart(f2, width="stretch")

# ------------------------------------------------------------------ 3. планировщик
with tabs[2]:
    st.subheader("Планировщик отправок (М3)")
    st.caption("Экономические параметры — синтетика и допущения; вывод проверен на ±30% (вкладка «До / после»).")
    c1, c2, c3 = st.columns(3)
    P = Params(Q_t=c1.number_input("Осталось вывезти на дату решения, т", 1000, 100000, 12000, 500),
               K_t_week=c1.number_input("Своя техника, т/нед", 100, 20000, 1000, 100),
               K_extra_t_week=c1.number_input("Доп. техника максимум, т/нед", 0, 20000, 600, 100),
               c_road=c2.number_input("Зимник, ₽/т", 1000, 200000, 20000, 1000),
               c_extra=c2.number_input("Надбавка за доп. технику, ₽/т", 0, 200000, 10000, 1000),
               c_exp=c3.number_input("Дорогой канал (авиа/СВП), ₽/т", 10000, 1000000, 120000, 5000),
               c_hold_week=c3.number_input("Хранение, ₽/т·нед", 0, 10000, 400, 50),
               close_margin_days=margin)
    q = fc_cached(pid, hy, max(issue_day, 150), "close")
    if not q or "error" in q:
        st.info("Нет прогноза закрытия для этой точки.")
    else:
        weeks = np.arange(max(issue_day, P.start_day), 300, 7, dtype=float)
        sc = sample_quantiles(q["q10"], q["q50"], q["q90"], P.n_scenarios, np.random.default_rng(P.seed)) - margin
        sc = np.maximum(sc, weeks[0])
        p_closed = np.array([(sc < w + 7).mean() for w in weeks])
        R = q["calendar"] - margin + 14
        xb, xe = plan_week_volumes(weeks, p_closed, P.Q_t, P, R)
        x = xb + xe
        cal_close = q["calendar"] - margin
        cb, ce = calendar_plan(weeks, cal_close, P)
        x_cal = cb + ce
        st.write(f"Рекомендация: своей техникой **{xb.sum():,.0f} т**, привлечь дополнительную на "
                 f"**{xe.sum():,.0f} т**; календарный план привлекает {ce.sum():,.0f} т.".replace(",", " "))
        dates = [from_season_day(w, hy).date() for w in weeks]
        keep = (x > 0) | (x_cal > 0) | (p_closed < 0.99)
        f = go.Figure()
        f.add_trace(go.Bar(x=np.array(dates)[keep], y=x_cal[keep], name="Календарный план"))
        f.add_trace(go.Bar(x=np.array(dates)[keep], y=x[keep], name="Рекомендация по прогнозу"))
        f.add_trace(go.Scatter(x=np.array(dates)[keep], y=(p_closed[keep] * P.K_t_week), name="P(путь закрыт) × K",
                               yaxis="y", line=dict(color="red", dash="dot")))
        f.update_layout(barmode="group", title="Отправки по неделям, т", height=420)
        st.plotly_chart(f, width="stretch")
        risk_cal = float(np.sum(x_cal * p_closed))
        risk_fc = float(np.sum(x * p_closed)) + (P.Q_t - x.sum())
        a, b = st.columns(2)
        a.metric("Ожидаемый недовоз — календарь", f"{risk_cal:,.0f} т".replace(",", " "))
        b.metric("Ожидаемый недовоз — по прогнозу", f"{risk_fc:,.0f} т".replace(",", " "),
                 delta=f"{risk_fc - risk_cal:,.0f} т".replace(",", " "), delta_color="inverse")
        if not np.isnan(q["actual"]):
            st.markdown("**Проверка на факте этого сезона** (оба правила, фактическая дата закрытия):")
            path = fc_path(pid, hy, "close", tuple(range(150, 271, 7)))
            res = [simulate_season(q["actual"] - margin, path[["issue_day", "q10", "q50", "q90"]] if rule == "forecast"
                                   else None, cal_close, P, rule) for rule in ("calendar", "forecast")]
            st.dataframe(pd.DataFrame(res).set_index("rule")[["share_delivered", "undelivered_t", "cost_rub_per_t"]]
                         .rename(index={"calendar": "до (календарь)", "forecast": "после (прогноз)"},
                                 columns={"share_delivered": "доля плана", "undelivered_t": "недовоз, т",
                                          "cost_rub_per_t": "₽/т"}).round(2))

# ------------------------------------------------------------------ 4. рейтинг риска
with tabs[3]:
    st.subheader(f"Рейтинг риска на {issue_date:%d.%m.%Y}")
    st.caption("Ожидаемый недовоз = P(путь закроется до выполнения плана) × остаток плана. "
               "Сортировка — с каких путей начинать пересчёт графика.")
    P0 = Params(close_margin_days=margin)
    rows = []
    for p in WEATHER_POINTS:
        q = fc_cached(p, hy, max(issue_day, 150), "close")
        if not q or "error" in q:
            continue
        need_weeks = P0.Q_t / P0.K_t_week
        done_day = max(issue_day, P0.start_day) + 7 * need_weeks
        p_early = p_by(q, done_day + margin)
        rows.append({"путь": label(p), "закрытие (50%)": from_season_day(q["q50"] - margin, hy).date(),
                     "диапазон 10–90%": f"{from_season_day(q['q10'] - margin, hy):%d.%m} – "
                                        f"{from_season_day(q['q90'] - margin, hy):%d.%m}",
                     "сдвиг к календарю, сут": round(q["q50"] - q["calendar"]),
                     "P(закроется до конца плана)": round(p_early, 2),
                     "ожидаемый недовоз, т": round(p_early * P0.Q_t * 0.5)})
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("ожидаемый недовоз, т", ascending=False), width="stretch")

# ------------------------------------------------------------------ 5. до/после
with tabs[4]:
    st.subheader("Бэктест на прошлых сезонах")
    for name, f in (("М1: точность прогноза", REPORTS / "m1_metrics.md"),
                    ("«До/после» для А3", REPORTS / "before_after.md")):
        if f.exists():
            with st.expander(name, expanded=True):
                st.markdown(f.read_text(encoding="utf-8"), unsafe_allow_html=True)
        else:
            st.info(f"Нет файла {f.name}: запустите scripts/train_m1.py и scripts/before_after.py")

# ------------------------------------------------------------------ 6. о данных
with tabs[5]:
    f = REPORTS / "data_checks.md"
    st.markdown(f.read_text(encoding="utf-8") if f.exists() else "Запустите scripts/validate_data.py")
