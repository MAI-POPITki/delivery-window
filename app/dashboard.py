"""
Дашборд «Окно завоза».

    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.welcome import render_welcome, show_welcome

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
    "okno.planner", "okno.corridor", "okno.m1", "okno.weather_forecast", "okno.forecast",
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
from okno.corridor import CorridorConfig, compute_prestock_schedule, run_sensitivity_check, simulate_corridor
from okno.ice import from_season_day, to_season_day
from okno.paths import DOCS, PROCESSED, REPORTS
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
region = st.sidebar.selectbox("Регион", list(REGIONS), format_func=lambda k: REGIONS[k]["name"])
st.sidebar.caption(REGIONS[region]["note"])
WEATHER_POINTS = region_points(region)
_era5, _obs, _fut = weather_until(DATA_VER)
st.sidebar.caption(f"Погода: архив ERA5 по {pd.Timestamp(_era5):%d.%m}"
                   + (f", оперативные данные по {pd.Timestamp(_obs):%d.%m}, прогноз до {pd.Timestamp(_fut):%d.%m.%Y}"
                      if _obs else "") + ". Обновляется ежедневно.")
today = dt.date.today()
cur_hy = today.year + (1 if today.month >= 9 else 0)
hy = st.sidebar.selectbox("Сезон", list(range(cur_hy, 1985, -1)), format_func=season_name, index=0)
season_start = dt.date(hy - 1, 9, 1)
default_date = min(today, dt.date(hy, 5, 31)) if hy == cur_hy else dt.date(hy, 3, 1)
issue_date = st.sidebar.date_input("Дата решения", value=default_date,
                                   min_value=dt.date(hy - 1, 10, 1), max_value=dt.date(hy, 6, 15))
issue_day = int(to_season_day(issue_date, hy))
_dflt = REGIONS[region]["default"]
pid = st.sidebar.selectbox("Путь", WEATHER_POINTS, format_func=label,
                           index=WEATHER_POINTS.index(_dflt) if _dflt in WEATHER_POINTS else 0)
horizon = st.sidebar.slider("Горизонт N, дней", 7, 60, 14, 7)
margin = st.sidebar.slider("Запас до таяния при закрытии, сут", 0, 35, 21,
                           help="Путь закрывают раньше физического начала таяния: ограничения массы, "
                                "дорожный режим. Допущение, проверяется на чувствительность.")
use_weather_close = st.sidebar.checkbox(
    # Выключено по умолчанию: на проверке выигрыш 1,3%, на горизонтах 7 и 14 дней — без улучшения.
    "Погодная модель закрытия (эксперимент)", value=False, key="weather_close_enabled_v3",
    help="Модель обучена на архиве прогнозов GFS. Пилот — переправа Якутск — Нижний Бестях. "
         "При отсутствии данных используется обычная М1. Прогноз открытия не меняется.",
)
if use_weather_close:
    st.sidebar.caption("Эксперимент: прогноз GFS на 6 дней. Выигрыш за 7/14 дней не подтверждён, "
                       "диапазон пока ненадёжен. Результаты — во вкладке «До / после».")
last = FC.last_weather_day(pid, hy)
if last is not None and last < issue_day:
    st.sidebar.warning(f"Погода есть до {from_season_day(last, hy).date():%d.%m.%Y}: прогноз выпускается на эту дату.")

tabs = st.tabs([
    "Карта путей",
    "Карточка пути",
    "Сквозной коридор (А3)",
    "Планировщик",
    "Рейтинг риска",
    "До / после",
    "О данных",
])

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
        rows.append({"lat": meta["lat"], "lon": meta["lon"], "color": color, "txt": txt, "p_ok": p_ok})
    pts = pd.DataFrame(rows)
    _l1, _l2 = st.columns(2)
    show_roads = _l1.checkbox("Зимники и ледовые переправы (OpenStreetMap)", value=True, key="layer_roads")
    show_s1 = _l2.checkbox("Лена по радарным снимкам (М2)", value=region == "yakutia", key="layer_s1")
    fmap = go.Figure()
    # 1) реальные зимники из OpenStreetMap (scripts/osm_roads.py) — одной линией с разрывами, так быстрее
    _roads_file = Path(PROCESSED).parent / "osm" / "winter_roads.json"
    if show_roads and _roads_file.exists():
        import json as _json
        _lat, _lon, _txt = [], [], []
        for r in _json.loads(_roads_file.read_text(encoding="utf-8")):
            if r.get("region", region) != region:  # только выбранный регион — карта легче
                continue
            nm = r["name"] or ("ледовая переправа" if r["kind"] == "ice_road" else "зимник")
            _lat += r["lat"] + [None]
            _lon += r["lon"] + [None]
            _txt += [nm] * len(r["lat"]) + [None]
        fmap.add_trace(go.Scattermap(lat=_lat, lon=_lon, mode="lines", text=_txt, hoverinfo="text",
                                     line=dict(width=2.5, color="#2b6cb0"), name="зимники OSM", opacity=0.75))
    # 2) Лена по снимкам Sentinel-1 (М2): белое кольцо + цвет состояния
    _pp = Path(PROCESSED).parent / "s1" / "predictions.csv"
    if show_s1 and _pp.exists():
        from okno.s1_points import LENA_NAMES
        _p = pd.read_csv(_pp, parse_dates=["date"]).dropna(subset=["p_ice"])
        _idx = pd.read_csv(Path(PROCESSED).parent / "s1" / "index.csv", usecols=["item_id", "lat", "lon"])
        _s = _p.merge(_idx, on="item_id").sort_values("date").groupby("point").tail(1)
        _state = np.where(_s.p_ice >= 0.8, "лёд", np.where(_s.p_ice <= 0.2, "вода", "неясно"))
        _col = np.where(_s.p_ice >= 0.8, "#cfeeff", np.where(_s.p_ice <= 0.2, "#0b3d91", "#8aa5c8"))
        _t = [f"<b>{LENA_NAMES.get(a, a)}</b><br>Sentinel-1, {d:%d.%m.%Y}: <b>{st_}</b> (P(лёд) = {pi:.2f})"
              for a, d, st_, pi in zip(_s.point, _s.date, _state, _s.p_ice)]
        fmap.add_trace(go.Scattermap(lat=_s.lat, lon=_s.lon, mode="markers", hoverinfo="skip",
                                     marker=dict(size=17, color="#ffffff"), showlegend=False))
        fmap.add_trace(go.Scattermap(lat=_s.lat, lon=_s.lon, mode="markers", text=_t, hoverinfo="text",
                                     marker=dict(size=12, color=list(_col)), name="Лена по снимкам: лёд / вода"))
    # 3) пути: цвет — вероятность «путь открыт через N дней» по непрерывной шкале
    ok = pts[pts.p_ok.notna()]
    na = pts[pts.p_ok.isna()]
    if len(na):
        fmap.add_trace(go.Scattermap(lat=na.lat, lon=na.lon, mode="markers", text=na.txt, hoverinfo="text",
                                     marker=dict(size=15, color="#9e9e9e"), name="нет данных для прогноза"))
    if len(ok):
        fmap.add_trace(go.Scattermap(lat=ok.lat, lon=ok.lon, mode="markers", hoverinfo="skip",
                                     marker=dict(size=22, color="#ffffff"), showlegend=False))
        fmap.add_trace(go.Scattermap(
            lat=ok.lat, lon=ok.lon, mode="markers", text=ok.txt, hoverinfo="text", name="пути: прогноз",
            marker=dict(size=17, color=ok.p_ok, cmin=0, cmax=1, colorscale="RdYlGn",
                        colorbar=dict(title=dict(text="P(путь<br>открыт)", font=dict(size=11)), tickformat=".0%",
                                      thickness=12, len=0.6, x=0.99, xanchor="right", y=0.5,
                                      bgcolor="rgba(255,255,255,0.75)"))))
    _c = REGIONS[region]["center"]
    fmap.update_layout(map=dict(style="carto-voyager", center=dict(lat=_c[0], lon=_c[1]), zoom=REGIONS[region]["zoom"]),
                       margin=dict(l=0, r=0, t=0, b=0), height=560, showlegend=True,
                       legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)", font=dict(size=11, color="#222")))
    st.plotly_chart(fmap, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    st.caption("Цвет точки пути — вероятность, что путь открыт через N дней: P(лёд уже держит 20 т) × P(путь ещё не "
               "закрыт); серый — нет данных. Синие линии — зимники и ледовые переправы региона из OpenStreetMap. Светлые точки на Лене — лёд, тёмные — вода по последнему снимку Sentinel-1 (М2).")

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

    # состояние Лены по последним радарным снимкам (М2)
    if region == "yakutia":
        _pp = Path(PROCESSED).parent / "s1" / "predictions.csv"
        if _pp.exists():
            from okno.s1_points import LENA_NAMES
            _p = pd.read_csv(_pp, parse_dates=["date"]).dropna(subset=["p_ice"])
            _last = _p.sort_values("date").groupby("point").tail(1).copy()
            _last["точка на Лене"] = _last.point.map(LENA_NAMES)
            _last["снимок"] = _last.date.dt.strftime("%d.%m.%Y")
            _last["состояние"] = np.where(_last.p_ice >= 0.8, "лёд", np.where(_last.p_ice <= 0.2, "вода", "неясно"))
            _last["P(лёд)"] = _last.p_ice.round(2)
            st.markdown("**Лена по радарным снимкам Sentinel-1 (М2)** — что на реке по последнему снимку")
            st.dataframe(_last.sort_values("date", ascending=False)[["точка на Лене", "снимок", "состояние", "P(лёд)"]],
                         hide_index=True, width="stretch")
            st.caption("Классификатор лёд/вода: точность 0,95 на годах вне обучения. Снимок примерно раз в 12 дней; "
                       "даты вскрытия по ряду снимков — с точностью ≈ 9 сут.")

# ------------------------------------------------------------------ 3. сквозной коридор (А3)
with tabs[2]:
    st.subheader("Сквозной логистический коридор из 4 звеньев (Кейс А3)")
    st.caption(
        "Маршрут: Ж/д хаб Сибири → ст. Нижний Бестях (правый берег) → переправа через р. Лену → "
        "целевой склад ТЛЦ г. Якутска (левый берег). Показывает весь путь целиком: как ранний буфер "
        "увеличивает затраты на хранение в Якутске, но кратно снижает полную стоимость цепочки."
    )

    _bt_corridor_file = REPORTS / "window_backtest_yakutsk_2001_2024.csv"
    _bt_corr_df = pd.read_csv(_bt_corridor_file) if _bt_corridor_file.exists() else pd.DataFrame()

    _scen_options = ["Эталонный сценарий (распутица 27 сут, раннее закрытие −6 сут — аналог мая 2024)"]
    if not _bt_corr_df.empty:
        _scen_options += [
            f"Сезон {r.season} ({r.regime}, вскрытие: день {r.actual_sat_breakup_day:.0f})"
            for r in _bt_corr_df.sort_values("hydro_year", ascending=False).itertuples()
        ]
    _selected_scen = st.selectbox("Сценарий / исторический сезон (2001–2024)", _scen_options, index=0, key="corr_scenario")

    with st.expander("Параметры и тарифы 4 звеньев маршрута (ползунки для проверки чувствительности)", expanded=False):
        _k1, _k2, _k3 = st.columns(3)
        _d_tons = _k1.slider("Суточная потребность ТЛЦ Якутска d, т/сут", 10.0, 50.0, 25.0, 1.0, key="corr_d")
        _q_norm = _k1.slider("Пропускная способность парома / льда Q_max, т/сут", 35.0, 120.0, 60.0, 5.0, key="corr_qmax")
        _q_alt = _k1.slider("Лимит СВП / Ми-8 в распутицу Q_alt, т/сут", 2.0, 20.0, 8.0, 1.0, key="corr_qalt")
        _v_max = _k1.slider("Свободная ёмкость склада Якутска V_max, т", 300.0, 1200.0, 650.0, 50.0, key="corr_vmax")

        _c_rail = _k2.slider("1. Тариф ж/д до Н. Бестяха, ₽/т", 4000.0, 15000.0, 8500.0, 500.0, key="corr_crail")
        _h_nb = _k2.slider("2. Простой / хранение в Н. Бестяхе, ₽/т·сут", 150.0, 1000.0, 450.0, 25.0, key="corr_hnb")
        _c_norm = _k2.slider("3а. Штатный тариф переправы, ₽/т", 800.0, 4000.0, 1800.0, 100.0, key="corr_cnorm")
        _c_alt = _k2.slider("3б. Аварийный тариф СВП / Ми-8, ₽/т", 15000.0, 60000.0, 35000.0, 1000.0, key="corr_calt")

        _h_yak = _k3.slider("4а. Плановое хранение на ТЛЦ Якутска, ₽/т·сут", 100.0, 500.0, 220.0, 10.0, key="corr_hyak")
        _h_over = _k3.slider("4а. Аренда при переполнении ТЛЦ, ₽/т·сут", 300.0, 1200.0, 600.0, 25.0, key="corr_hover")
        _p_def = _k3.slider("4б. Штраф за дефицит / срыв сроков, ₽/т·сут", 5000.0, 25000.0, 12000.0, 500.0, key="corr_pdef")
        _rasputitsa_len = _k3.slider("Длительность грузовой распутицы, сут", 15, 40, 27, 1, key="corr_rasp_len")

    _cfg_corr = CorridorConfig(
        daily_demand_tons=_d_tons,
        rail_cost_per_ton=_c_rail,
        nb_holding_cost_per_ton_day=_h_nb,
        crossing_cap_normal_tons=_q_norm,
        crossing_cap_alt_tons=_q_alt,
        crossing_cost_normal=_c_norm,
        crossing_cost_alt=_c_alt,
        yak_warehouse_cap_tons=_v_max,
        yak_holding_cost_per_ton_day=_h_yak,
        yak_overflow_cost_per_ton_day=_h_over,
        shortage_penalty_per_ton_day=_p_def,
    )
    if _q_norm <= _d_tons:
        st.warning(
            f"Пропускная способность переправы Q_max ({_q_norm:.0f} т/сут) не превышает суточное потребление d "
            f"({_d_tons:.0f} т/сут): свободного резерва для предварительного накопления буфера нет."
        )

    if _selected_scen.startswith("Эталонный") or _bt_corr_df.empty:
        _ac, _ao = 30, 30 + _rasputitsa_len
        _cc, _co = 36, 36 + _rasputitsa_len
        _mc, _mo = 29, 29 + _rasputitsa_len + 2
        _scen_caption = (
            f"Эталонный режим ранней весны: река закрывается на день {_ac} (на 6 сут раньше календарной даты {_cc}), "
            f"распутица длится {_rasputitsa_len} сут."
        )
    else:
        _sel_season_str = _selected_scen.split()[1]
        _srow = _bt_corr_df[_bt_corr_df["season"] == _sel_season_str].iloc[0]
        _base_off = 35
        _anom_act = int(round(float(_srow["actual_sat_breakup_day"] - _srow["cal_median_breakup_day"])))
        _anom_q20 = int(round(float(_srow["q20_breakup_day"] - _srow["cal_median_breakup_day"])))
        _anom_q50 = int(round(float(_srow["q50_breakup_day"] - _srow["cal_median_breakup_day"])))
        _ac = _base_off + _anom_act
        _ao = _ac + _rasputitsa_len
        _cc = _base_off
        _co = _cc + _rasputitsa_len
        _mc = _base_off + _anom_q20
        _mo = max(_mc + _rasputitsa_len, _base_off + _anom_q50 + _rasputitsa_len - 3)
        _scen_caption = (
            f"Сезон **{_srow['season']}** ({_srow['regime']}): дата выпуска прогноза `{_srow['issue_date']}` "
            f"(за {_srow['lead_days_to_breakup']:.0f} сут до вскрытия). Отклонение факта от календаря: **{_anom_act:+d} сут**, "
            f"нижний квантиль модели `q20`: **{_anom_q20:+d} сут** к календарю."
        )
    st.info(_scen_caption)

    _sched_info = compute_prestock_schedule(_cfg_corr, _mc, _mo)
    _res_before, _daily_before = simulate_corridor(_cfg_corr, _ac, _ao, _cc, _co, use_prestock=False, return_daily=True)
    _res_after, _daily_after = simulate_corridor(_cfg_corr, _ac, _ao, _mc, _mo, use_prestock=True, return_daily=True)

    _diff_rub = _res_after["total_cost_rub"] - _res_before["total_cost_rub"]
    _diff_pct = 100.0 * _diff_rub / _res_before["total_cost_rub"] if _res_before["total_cost_rub"] else 0.0
    _yak_delta = _res_after["cost_holding_yak_rub"] - _res_before["cost_holding_yak_rub"]

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric(
        "Полная стоимость цепочки (После)",
        f"{_res_after['total_cost_rub'] / 1e6:.2f} млн ₽",
        delta=f"{_diff_rub / 1e6:+.2f} млн ₽ ({_diff_pct:+.1f}%)",
        delta_color="inverse",
    )
    _m2.metric(
        "Дни дефицита в Якутске",
        f"{_res_after['days_with_deficit']} дн.",
        delta=f"{_res_after['days_with_deficit'] - _res_before['days_with_deficit']:+d} дн. (было {_res_before['days_with_deficit']})",
        delta_color="inverse",
    )
    _m3.metric(
        "Затор в Нижнем Бестяхе",
        f"{_res_after['tons_stuck_nb_ton_days']:.0f} т·сут",
        delta=f"{_res_after['tons_stuck_nb_ton_days'] - _res_before['tons_stuck_nb_ton_days']:+.0f} т·сут",
        delta_color="inverse",
    )
    _m4.metric(
        "Хранение на ТЛЦ Якутска (рост звена 4)",
        f"{_res_after['cost_holding_yak_rub'] / 1e6:.2f} млн ₽",
        delta=f"{_yak_delta / 1e6:+.2f} млн ₽",
        delta_color="off",
    )
    st.caption(
        f"Учёт «бутылочного горлышка» переправы: резерв провозной способности `Q_max − d = {_sched_info['spare_crossing_cap_tons']:.0f} т/сут` → "
        f"окно предзавоза `N_pre = {_sched_info['prestock_duration_days']} сут` (+`{_cfg_corr.rail_lead_time_days}` сут хода по ж/д = "
        f"старт отгрузки с ж/д хаба за `{_sched_info['prestock_duration_days'] + _cfg_corr.rail_lead_time_days}` сут до `q20`)."
    )

    _col_chart1, _col_chart2 = st.columns(2)
    with _col_chart1:
        _seg_names = [
            "1. Ж/д плечо",
            "2. Простой Н.Бестях",
            "3а. Паром / лёд",
            "3б. СВП / Ми-8",
            "4а. Хранение Якутск",
            "4б. Штраф дефицита",
        ]
        _vals_b = [
            _res_before["cost_rail_rub"] / 1e6,
            _res_before["cost_holding_nb_rub"] / 1e6,
            _res_before["cost_crossing_normal_rub"] / 1e6,
            _res_before["cost_crossing_alt_rub"] / 1e6,
            _res_before["cost_holding_yak_rub"] / 1e6,
            _res_before["cost_shortage_rub"] / 1e6,
        ]
        _vals_a = [
            _res_after["cost_rail_rub"] / 1e6,
            _res_after["cost_holding_nb_rub"] / 1e6,
            _res_after["cost_crossing_normal_rub"] / 1e6,
            _res_after["cost_crossing_alt_rub"] / 1e6,
            _res_after["cost_holding_yak_rub"] / 1e6,
            _res_after["cost_shortage_rub"] / 1e6,
        ]
        _fig_seg = go.Figure()
        _fig_seg.add_trace(go.Bar(x=_seg_names, y=_vals_b, name="Сейчас (по календарю)", marker_color="#c62828"))
        _fig_seg.add_trace(go.Bar(x=_seg_names, y=_vals_a, name="После («Окно завоза»)", marker_color="#2e7d32"))
        _fig_seg.update_layout(
            barmode="group",
            title="Декомпозиция затрат по 4 звеньям пути, млн ₽",
            yaxis_title="млн ₽",
            height=380,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(_fig_seg, width="stretch")

    with _col_chart2:
        _fig_inv = go.Figure()
        _fig_inv.add_vrect(x0=_ac, x1=_ao, fillcolor="#ffebee", opacity=0.5, line_width=0, annotation_text="распутица")
        _fig_inv.add_trace(go.Scatter(
            x=_daily_before["day"], y=_daily_before["inv_nb_t"],
            name="До: затор в Н. Бестяхе, т", line=dict(color="#c62828", dash="dot"),
        ))
        _fig_inv.add_trace(go.Scatter(
            x=_daily_before["day"], y=_daily_before["inv_yak_t"],
            name="До: запас ТЛЦ Якутск, т", line=dict(color="#ef6c00"),
        ))
        _fig_inv.add_trace(go.Scatter(
            x=_daily_after["day"], y=_daily_after["inv_yak_t"],
            name="После: запас ТЛЦ Якутск, т", line=dict(color="#2e7d32", width=2.5),
        ))
        _fig_inv.add_hline(y=_cfg_corr.yak_warehouse_cap_tons, line_dash="dash", line_color="#546e7a",
                           annotation_text="ёмкость ТЛЦ V_max")
        _fig_inv.update_layout(
            title="Посуточная динамика запасов по берегам Лены (t = 0…89 сут)",
            xaxis_title="день квартального горизонта",
            yaxis_title="тонн",
            height=380,
            legend=dict(orientation="h", y=1.14),
        )
        st.plotly_chart(_fig_inv, width="stretch")

    if not _bt_corr_df.empty:
        _fig_q = go.Figure()
        _x_s = _bt_corr_df["season"].tolist()
        _fig_q.add_trace(go.Scatter(
            x=_x_s + _x_s[::-1],
            y=list(_bt_corr_df["q90_breakup_day"]) + list(_bt_corr_df["q10_breakup_day"][::-1]),
            fill="toself", line=dict(width=0), fillcolor="rgba(43,108,176,0.20)", name="Диапазон q10–q90",
        ))
        _fig_q.add_trace(go.Scatter(
            x=_x_s, y=_bt_corr_df["q20_breakup_day"],
            name="Квантиль предзавоза q20", line=dict(color="#2e7d32", width=2),
        ))
        _fig_q.add_trace(go.Scatter(
            x=_x_s, y=_bt_corr_df["cal_median_breakup_day"],
            name="Календарная медиана", line=dict(color="#757575", dash="dash"),
        ))
        _fig_q.add_trace(go.Scatter(
            x=_x_s, y=_bt_corr_df["actual_sat_breakup_day"],
            mode="markers+lines", name="Факт (спутник)", line=dict(color="#111111", width=1.5),
        ))
        _fig_q.update_layout(
            title="Квантили вскрытия Лены за 14–21 день до события (LOSO-бэктест CatBoost, 2001–2024 гг.)",
            xaxis_title="сезон",
            yaxis_title="день сезона (0 = 1 сентября)",
            height=340,
        )
        st.plotly_chart(_fig_q, width="stretch")

    with st.expander("Интерактивный стресс-тест чувствительности (±30% от выбранных ползунков)", expanded=False):
        _live_sens = run_sensitivity_check(_cfg_corr, _ac, _ao, _cc, _co, _mc, _mo)
        st.dataframe(
            _live_sens.rename(columns={
                "label": "Параметр",
                "variation": "Отклонение",
                "param_value": "Значение",
                "before_mln_rub": "До, млн ₽",
                "after_mln_rub": "После, млн ₽",
                "saving_mln_rub": "Экономия, млн ₽",
                "saving_pct": "Экономия, %",
                "yak_storage_delta_mln_rub": "Рост хранения в Якутске, млн ₽",
                "effect_positive": "Эффект > 0",
            })[[
                "Параметр", "Отклонение", "Значение", "До, млн ₽", "После, млн ₽",
                "Экономия, млн ₽", "Экономия, %", "Рост хранения в Якутске, млн ₽", "Эффект > 0",
            ]],
            hide_index=True,
            width="stretch",
        )

# ------------------------------------------------------------------ 4. планировщик
with tabs[3]:
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

# ------------------------------------------------------------------ 5. рейтинг риска
with tabs[4]:
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

# ------------------------------------------------------------------ 6. до/после
with tabs[5]:
    st.subheader("Бэктест на прошлых сезонах")
    for name, f in (("Сквозной коридор 4 звеньев: «До и После» (Кейс А3)", REPORTS / "corridor_before_after.md"),
                    ("Квантильная модель окна CatBoost (2001–2024)", REPORTS / "window_model_report.md"),
                    ("М1: точность прогноза", REPORTS / "m1_metrics.md"),
                    ("Погодная модель закрытия: эксперимент", REPORTS / "weather_close.md"),
                    ("«До/после» по складскому М3", REPORTS / "before_after.md"),
                    ("Стресс-тест допущений", REPORTS / "stress_test.md"),
                    ("Журнал прогнозов: проверка вперёд", REPORTS / "forecast_journal.md")):
        if f.exists():
            with st.expander(name, expanded=True):
                st.markdown(f.read_text(encoding="utf-8"), unsafe_allow_html=True)
        else:
            st.info(f"Нет файла {f.name}: запустите python src/corridor_simulator.py и scripts/train_m1.py")

# ------------------------------------------------------------------ 7. о данных
with tabs[6]:
    _val_a3 = DOCS / "validation_report.md"
    if _val_a3.exists():
        st.markdown(_val_a3.read_text(encoding="utf-8"))
        st.divider()
    f = REPORTS / "data_checks.md"
    st.markdown(f.read_text(encoding="utf-8") if f.exists() else "Запустите scripts/validate_data.py")

