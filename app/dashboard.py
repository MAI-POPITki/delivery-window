"""
Дашборд «Окно завоза».

    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.welcome import render_welcome, show_welcome
from app.examples import load_nizhneyansk_batagay

st.set_page_config(page_title="Окно завоза", page_icon="🧊", layout="wide")

# Главная страница не загружает данные и модели до перехода в дашборд.
if not st.session_state.get("dashboard_open", False):
    render_welcome()
    st.stop()

import datetime as dt
import importlib
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Streamlit перезапускает этот скрипт при каждом изменении, но уже импортированные модули okno
# остаются в памяти процесса. Сначала обновляем зависимости, затем импортируем зависимые
# модули: иначе новый weather_forecast падает ещё до reload из-за старого paths в памяти.
_MODULE_NAMES = (
    "okno.paths", "okno.regions", "okno.report", "okno.ice", "okno.calib", "okno.features",
    "okno.planner", "okno.m1", "okno.weather_forecast", "okno.forecast",
)
_cached_modules = [sys.modules.get(name) for name in _MODULE_NAMES]
_needs_reload = any(module is None or getattr(module, "_loaded_mtime_ns", None)
                    != os.stat(module.__file__).st_mtime_ns for module in _cached_modules)
for _name in _MODULE_NAMES:
    _mod = sys.modules.get(_name)
    if _mod is None:
        _mod = importlib.import_module(_name)
    elif _needs_reload:
        _mod = importlib.reload(_mod)
    _mod._loaded_mtime_ns = os.stat(_mod.__file__).st_mtime_ns

from okno import forecast as FC
from okno.ice import from_season_day, to_season_day
from okno.paths import PROCESSED, REPORTS
from okno.planner import Params, calendar_plan, plan_week_volumes, sample_quantiles, simulate_season

# ------------------------------------------------------------------ данные
from okno.regions import REGIONS, label  # noqa: E402


def region_points(key):
    """Точки региона, для которых есть погода (иначе прогноз не посчитать)."""
    r = REGIONS[key]
    pts = list(r["extra_points"]) + [f"gauge_{c}" for c in r["gauges"]]
    return [p for p in pts if FC._weather_path(p).exists()]


def season_name(hy):
    return f"{hy - 1}/{str(hy)[2:]}"


# версия данных входит в ключ кэша: после ежедневного обновления погоды (GitHub Actions)
# дашборд пересчитает прогнозы сам, без перезапуска
DATA_VER = FC.data_version()


@st.cache_data(show_spinner=False, max_entries=5000, ttl=3600)
def _fc_cached(pid, hy, day, kind, ver, use_weather):
    try:
        return FC.forecast(pid, hy, day, kind, use_weather=use_weather)
    except Exception as e:  # noqa: BLE001 — в дашборде показываем, а не падаем
        return {"error": str(e)}


def fc_cached(pid, hy, day, kind):
    return _fc_cached(pid, hy, day, kind, DATA_VER, use_weather_close)


@st.cache_data(show_spinner=False, max_entries=500, ttl=3600)
def _fc_path(pid, hy, kind, days, ver, use_weather):
    rows = [_fc_cached(pid, hy, d, kind, ver, use_weather) for d in days]
    return pd.DataFrame([r for r in rows if r and "error" not in r])


def fc_path(pid, hy, kind, days):
    return _fc_path(pid, hy, kind, days, DATA_VER, use_weather_close)


@st.cache_data(show_spinner=False)
def weather_until(ver):
    """(конец архива ERA5, конец оперативных наблюдений, конец прогноза погоды)."""
    from okno.paths import RECENT, WEATHER
    era5 = pd.read_csv(WEATHER / "yakutsk_crossing.csv", usecols=["date"])["date"].max()
    rp = RECENT / "yakutsk_crossing.csv"
    if not rp.exists():
        return era5, None, None
    r = pd.read_csv(rp, usecols=["date", "is_forecast"])
    return era5, r[~r.is_forecast]["date"].max(), r[r.is_forecast]["date"].max()


def p_by(q, t):
    """P(событие ≤ t) из квантилей М1 (кусочно-линейно)."""
    if q is None or "error" in q or np.isnan(q.get("q50", np.nan)):
        return np.nan
    s = sample_quantiles(q["q10"], q["q50"], q["q90"], 4000, np.random.default_rng(0))
    return float((s <= t).mean())


# ------------------------------------------------------------------ боковая панель
st.sidebar.title("🧊 Окно завоза")
st.sidebar.button("О проекте", key="show_welcome", on_click=show_welcome,
                  icon=":material/arrow_back:", width="stretch")
st.sidebar.caption("Прогноз окон сезонных путей и график завоза. Открытые данные, без датчиков.")
st.sidebar.button("Пример: Нижнеянск → Батагай, 45 т", key="load_nizhneyansk_batagay",
                  on_click=load_nizhneyansk_batagay)
region = st.sidebar.selectbox("Регион", list(REGIONS), format_func=lambda k: REGIONS[k]["name"], key="region")
st.sidebar.caption(REGIONS[region]["note"])
WEATHER_POINTS = region_points(region)
_era5, _obs, _fut = weather_until(DATA_VER)
st.sidebar.caption(f"Погода: архив ERA5 по {pd.Timestamp(_era5):%d.%m}"
                   + (f", оперативные данные по {pd.Timestamp(_obs):%d.%m}, прогноз до {pd.Timestamp(_fut):%d.%m.%Y}"
                      if _obs else "") + ". Обновляется ежедневно.")
today = dt.date.today()
cur_hy = today.year + (1 if today.month >= 9 else 0)
hy = st.sidebar.selectbox("Сезон", list(range(cur_hy, 1985, -1)), format_func=season_name, index=0,
                         key="hydro_year")
season_start = dt.date(hy - 1, 9, 1)
default_date = min(today, dt.date(hy, 5, 31)) if hy == cur_hy else dt.date(hy, 3, 1)
if "decision_date" in st.session_state and not (
        dt.date(hy - 1, 10, 1) <= st.session_state.decision_date <= dt.date(hy, 6, 15)):
    st.session_state.decision_date = default_date
issue_date = st.sidebar.date_input("Дата решения", value=None if "decision_date" in st.session_state else default_date,
                                   min_value=dt.date(hy - 1, 10, 1), max_value=dt.date(hy, 6, 15),
                                   key="decision_date")
issue_day = int(to_season_day(issue_date, hy))
_dflt = REGIONS[region]["default"]
if st.session_state.get("point_id", _dflt) not in WEATHER_POINTS:
    st.session_state.point_id = _dflt if _dflt in WEATHER_POINTS else WEATHER_POINTS[0]
pid = st.sidebar.selectbox("Путь", WEATHER_POINTS, format_func=label,
                           index=WEATHER_POINTS.index(_dflt) if _dflt in WEATHER_POINTS else 0, key="point_id")
horizon = st.sidebar.slider("Горизонт N, дней", 7, 60, 14, 7, key="horizon")
margin = st.sidebar.slider("Запас до таяния при закрытии, сут", 0, 35, 21,
                           key="close_margin_days",
                           help="Путь закрывают раньше физического начала таяния: ограничения массы, "
                                "дорожный режим. Допущение, проверяется на чувствительность.")
use_weather_close = st.sidebar.checkbox(
    # Новый ключ включает эксперимент и в уже открытых сессиях после обновления.
    "Погодная модель закрытия (эксперимент)", value=True, key="weather_close_enabled_v2",
    help="Модель обучена на архиве прогнозов GFS. Пилот — переправа Якутск — Нижний Бестях. "
         "При отсутствии данных используется обычная М1. Прогноз открытия не меняется.",
)
if use_weather_close:
    st.sidebar.caption("Эксперимент: прогноз GFS на 6 дней. Выигрыш за 7/14 дней не подтверждён, "
                       "диапазон пока ненадёжен. Результаты — во вкладке «До / после».")
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
    _c = REGIONS[region]["center"]
    fmap.update_layout(map=dict(style="carto-voyager", center=dict(lat=_c[0], lon=_c[1]), zoom=REGIONS[region]["zoom"]),
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
            range_name = "Экспериментальный диапазон" if q.get("weather_model_applied") else "80%-диапазон"
            st.write(f"{range_name}: **{q['q10_date']:%d.%m} – {q['q90_date']:%d.%m}**{close_note}")
            st.write(f"Календарь (средняя за прошлые сезоны): {q['calendar_date']:%d.%m}; "
                     f"физика: {q['physics_date']:%d.%m}")
            st.write(f"Расчётная толщина льда на дату решения: **{q['h_stefan_now_cm']:.0f} см**")
            if kind == "close" and use_weather_close:
                st.caption(q["weather_model_message"])
                if q["weather_model_applied"]:
                    st.write(f"Сдвиг к обычной М1: **{q['weather_delta_days']:+.1f} сут**. "
                             f"Прогноз погоды учтён на {q['forecast_days_used']} дней.")
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
    if pid == "gauge_3414" and hy == 2024:
        st.info("Для примера Нижнеянск → Батагай: план заказа — 45 т, срок 22–31 марта 2024 г. "
                "Планировщик ниже учитывает прогноз закрытия, но не ограничивает отправки 31 марта. "
                "Гидропост — приближение для направления, не модель всего маршрута. "
                "Провозная способность, надбавка и авиационный тариф заданы как допущения.")
    c1, c2, c3 = st.columns(3)
    P = Params(Q_t=c1.number_input("Осталось вывезти на дату решения, т", 0.1, 100000.0, 12000.0, 1.0,
                                  format="%.1f", key="Q_t"),
               K_t_week=c1.number_input("Своя техника, т/нед", 0.1, 20000.0, 1000.0, 1.0, format="%.1f",
                                        key="K_t_week"),
               K_extra_t_week=c1.number_input("Доп. техника максимум, т/нед", 0.0, 20000.0, 600.0, 1.0,
                                             format="%.1f", key="K_extra_t_week"),
               c_road=c2.number_input("Зимник, ₽/т", 1000.0, 200000.0, 20000.0, 1000.0, format="%.2f",
                                      key="c_road"),
               c_extra=c2.number_input("Надбавка за доп. технику, ₽/т", 0.0, 200000.0, 10000.0, 1000.0,
                                       format="%.2f", key="c_extra"),
               c_exp=c3.number_input("Дорогой канал (авиа/СВП), ₽/т", 10000.0, 1000000.0, 120000.0, 5000.0,
                                     format="%.2f", key="c_exp"),
               c_hold_week=c3.number_input("Хранение, ₽/т·нед", 0.0, 10000.0, 400.0, 50.0, format="%.2f",
                                           key="c_hold_week"),
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
    for name, f in (("Нижнеянск → Батагай, 45 т: разбор фиксированного примера",
                     REPORTS / "nizhneyansk_batagay_45t.md"),
                    ("М1: точность прогноза", REPORTS / "m1_metrics.md"),
                    ("Погодная модель закрытия: эксперимент", REPORTS / "weather_close.md"),
                    ("«До/после» для А3", REPORTS / "before_after.md"),
                    ("Стресс-тест допущений", REPORTS / "stress_test.md"),
                    ("Журнал прогнозов: проверка вперёд", REPORTS / "forecast_journal.md")):
        if f.exists():
            with st.expander(name, expanded=True):
                st.markdown(f.read_text(encoding="utf-8"), unsafe_allow_html=True)
        else:
            st.info(f"Нет файла {f.name}: запустите scripts/train_m1.py и scripts/before_after.py")

# ------------------------------------------------------------------ 6. о данных
with tabs[5]:
    f = REPORTS / "data_checks.md"
    st.markdown(f.read_text(encoding="utf-8") if f.exists() else "Запустите scripts/validate_data.py")
