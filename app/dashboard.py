"""
Дашборд «Окно завоза».

    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.welcome import render_welcome, show_welcome

def _px(lat, lon, zoom):
    """Координаты в пикселях веб-меркатора при стартовом масштабе карты."""
    import math
    k = 512 * 2 ** zoom / 360  # плитка maplibre — 512 px
    return lon * k, -math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))) * k


def place_labels(df, zoom, obstacles=()):
    """Подписи путей без наложений (карта сама прячет подпись, которая задевает другую).
    1) Точки вплотную (< 34 px) получают одну общую подпись — имена столбиком.
    2) Для каждой подписи пробуем места «сверху», «снизу», «справа», «слева» и берём то, где она меньше всего
       задевает чужие точки, точки и подписи Лены и другие подписи; несколько проходов, пока расстановка меняется.
    obstacles — прямоугольники (x0, y0, x1, y1) в пикселях стартового масштаба."""
    gap, th = 14, 16  # отступ от центра точки и высота строки, px
    df = df.sort_values("lat", ascending=False).copy()
    xy = {i: _px(r.lat, r.lon, zoom) for i, r in zip(df.index, df.itertuples())}
    heads, names = [], {}
    for i, r in zip(df.index, df.itertuples()):
        x, y = xy[i]
        h = next((j for j, hx, hy in heads if (x - hx) ** 2 + (y - hy) ** 2 < 34 ** 2), None)
        if h is None:
            heads.append((i, x, y))
            names[i] = [r.short]
        else:
            names[h].append(r.short)
    # неразрывные пробелы и по одному имени в строке: короткую строку карта не переносит сама
    df["label_txt"] = ["<br>".join(n.replace(" ", " ") for n in names[i]) if i in names else "" for i in df.index]

    def cands(i):
        x, y = xy[i]
        lines = names[i]
        w, hh = 9 * max(map(len, lines)) + 4, th * len(lines)
        c = {"top center": (x - w / 2, y - gap - hh, x + w / 2, y - gap),
             "bottom center": (x - w / 2, y + gap, x + w / 2, y + gap + hh),
             "middle right": (x + gap, y - hh / 2, x + gap + w, y + hh / 2),
             "middle left": (x - gap - w, y - hh / 2, x - gap, y + hh / 2)}
        if len(lines) > 1:  # несколько строк карта ровно ставит только сбоку: сверху/снизу текст ложится на кружок
            c = {k: v for k, v in c.items() if k.startswith("middle")}
        return c

    def area(b, o):
        return max(0, min(b[2], o[2]) - max(b[0], o[0])) * max(0, min(b[3], o[3]) - max(b[1], o[1]))

    fixed = [tuple(b) for b in obstacles]
    dots = {i: (x - 11, y - 11, x + 11, y + 11) for i, (x, y) in xy.items()}
    order = sorted(names, key=lambda i: -len(names[i]))  # сначала многострочные: у них меньше вариантов
    C = {i: cands(i) for i in order}
    pos = {i: next(iter(C[i])) for i in order}

    def cost(i, k):
        b = C[i][k]
        return (sum(area(b, o) for o in fixed) + sum(area(b, d) for j, d in dots.items() if j != i)
                + sum(area(b, C[j][pos[j]]) for j in order if j != i))

    placed = set()
    for i in order:  # жадно, потом улучшение
        pos[i] = min(C[i], key=lambda k: sum(area(C[i][k], o) for o in fixed)
                     + sum(area(C[i][k], d) for j, d in dots.items() if j != i)
                     + sum(area(C[i][k], C[j][pos[j]]) for j in placed))
        placed.add(i)
    for _ in range(4):
        moved = False
        for i in order:
            k = min(C[i], key=lambda k: cost(i, k))
            if cost(i, k) < cost(i, pos[i]):
                pos[i], moved = k, True
        if not moved:
            break
    df["textpos"] = [pos.get(i, "top center") for i in df.index]
    return df[df.label_txt != ""]


LEGEND_HTML = '\n<style>.okno-legend{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 10px}\n.okno-legend div{flex:1 1 220px;border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:8px 10px;font-size:13px;line-height:1.35}\n.okno-legend b{font-size:14px}.okno-sw{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:middle;margin:0 3px 0 0;border:2px solid #fff;box-shadow:0 0 0 1px #999}\n.okno-bar{display:inline-block;width:70px;height:10px;border-radius:5px;vertical-align:middle;background:linear-gradient(90deg,#a50026,#fdae61,#ffffbf,#a6d96a,#006837)}</style>\n<div class="okno-legend">\n<div>🚚 <b>Пути с прогнозом</b><br>крупные точки с подписью; цвет — вероятность, что путь открыт через N дней<br>\n<span class="okno-bar"></span> 0% → 100% &nbsp; <span class="okno-sw" style="background:#9e9e9e"></span>нет данных</div>\n<div>🛰️ <b>Лена со спутника (М2)</b><br>что на реке по последнему радарному снимку<br>\n<span class="okno-sw" style="background:#bfe9ff"></span>лёд &nbsp; <span class="okno-sw" style="background:#0b3d91"></span>вода &nbsp;\n<span class="okno-sw" style="background:#0b3d91;opacity:.35"></span>снимок старше 14 дней</div>\n<div>🌊 <b>Притоки Лены</b><br>мелкие бирюзовые точки — прогноз даты вскрытия (отдельная модель)<br>\n<span class="okno-sw" style="background:#0f9d8a"></span>точка на реке</div>\n<div>🛣️ <b>Зимники и переправы</b><br>линии из OpenStreetMap: где проходят сезонные пути региона</div>\n</div>'

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
    "okno.planner", "app.planner_view", "okno.economic_planner", "okno.corridor", "okno.corridor_data", "app.corridor_view", "okno.m1", "okno.weather_forecast", "okno.forecast",
    "okno.yakutsk_replay", "app.yakutsk_experiment",
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
from okno.planner import Params, sample_quantiles, simulate_season

# ------------------------------------------------------------------ данные
from okno.regions import REGIONS, label  # noqa: E402


def region_points(key):
    """Точки региона, для которых есть погода (иначе прогноз не посчитать)."""
    from okno.regions import points_of
    return [p for p in points_of(key) if FC._weather_path(p).exists()]


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
def _activate_replay():
    st.session_state.update(replay_active=True, filter_region="yakutia", filter_season=2026,
                            filter_date=dt.date(2026, 3, 16), filter_point="yakutsk_crossing")


def _leave_replay():
    st.session_state["replay_active"] = False


st.sidebar.button("Эксперимент: весна 2026", key="activate_replay", on_click=_activate_replay,
                  help="Общий объём за сезон и остаток на дату решения задаются отдельно; сравнение пяти способов завоза.")
replay_active = st.session_state.get("replay_active", False)
if replay_active:
    st.sidebar.caption("Открыт сценарный эксперимент весны 2026 года.")
    st.sidebar.button("Вернуться к обычному планировщику", key="leave_replay", on_click=_leave_replay)
if st.session_state.get("filter_region") not in REGIONS:
    st.session_state.filter_region = next(iter(REGIONS))
region = st.sidebar.selectbox("Регион", list(REGIONS), format_func=lambda k: REGIONS[k]["name"],
                             index=None, key="filter_region", disabled=replay_active)
st.sidebar.caption(REGIONS[region]["note"])
WEATHER_POINTS = region_points(region)
if not replay_active:
    _era5, _obs, _fut = weather_until(DATA_VER)
    st.sidebar.caption(f"Погода: архив ERA5 по {pd.Timestamp(_era5):%d.%m}"
                       + (f", оперативные данные по {pd.Timestamp(_obs):%d.%m}, прогноз до {pd.Timestamp(_fut):%d.%m.%Y}"
                          if _obs else "") + ". Обновляется ежедневно.")
today = dt.date.today()
cur_hy = today.year + (1 if today.month >= 9 else 0)
if st.session_state.get("filter_season") not in range(cur_hy, 1985, -1):
    st.session_state.filter_season = cur_hy
hy = st.sidebar.selectbox("Сезон", list(range(cur_hy, 1985, -1)), format_func=season_name, index=None,
                         key="filter_season", disabled=replay_active)
season_start = dt.date(hy - 1, 9, 1)
default_date = min(today, dt.date(hy, 5, 31)) if hy == cur_hy else dt.date(hy, 3, 1)
if st.session_state.get("filter_date") is None or not dt.date(hy - 1, 10, 1) <= st.session_state.filter_date <= dt.date(hy, 6, 15):
    st.session_state.filter_date = default_date
issue_date = st.sidebar.date_input("Дата решения", value=None,
                                   min_value=dt.date(hy - 1, 10, 1), max_value=dt.date(hy, 6, 15),
                                   key="filter_date", disabled=replay_active)
issue_day = int(to_season_day(issue_date, hy))
_dflt = REGIONS[region]["default"]
if st.session_state.get("filter_point") not in WEATHER_POINTS:
    st.session_state.filter_point = _dflt if _dflt in WEATHER_POINTS else WEATHER_POINTS[0]
pid = st.sidebar.selectbox("Путь", WEATHER_POINTS, format_func=label,
                           index=None,
                           key="filter_point", disabled=replay_active)
if replay_active:
    st.sidebar.caption("Первое решение: погода ERA5 только по 11.03.2026. "
                       "Далее — ежедневный пересчёт с задержкой погоды пять суток.")
    st.title("Сценарная проверка завоза")
    from app.yakutsk_experiment import render_experiment
    render_experiment()
    st.stop()

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
    "Расширение применения",
])

# ------------------------------------------------------------------ 1. карта
with tabs[0]:
    st.subheader(f"Вероятность, что путь открыт через {horizon} дн. — {season_name(hy)}")
    t = issue_day + horizon
    rows = []
    basin_rows = []
    for p in WEATHER_POINTS:
        meta = FC.point_meta(p)
        if FC.is_basin(p):  # притоки Лены: только прогноз вскрытия — отдельный слой
            qc = fc_cached(p, hy, issue_day, "close")
            if qc and "error" not in qc:
                basin_rows.append({"lat": meta["lat"], "lon": meta["lon"],
                                   "txt": f"<b>{label(p)}</b><br>Вскрытие (М1 для притоков, 50%): {qc['q50_date']:%d.%m}, "
                                          f"диапазон {qc['q10_date']:%d.%m}–{qc['q90_date']:%d.%m}"})
            continue
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
    if len(pts):
        def _short(p):
            if p == "yakutsk_crossing":
                return "Переправа Якутск"
            parts = label(p).split(" (")[0].split(" — ")
            return parts[-1] if parts[-1][:1].isupper() else parts[0]  # «Алгий — устье» → «Алгий»
        pts["short"] = [_short(p) for p in WEATHER_POINTS if not FC.is_basin(p)]

    # ---- легенда с иконками: значки рисует браузер, на самой карте — цвет, размер и подписи
    st.markdown(LEGEND_HTML, unsafe_allow_html=True)
    _l1, _l2, _l3 = st.columns(3)
    show_roads = _l1.checkbox("🛣️ Зимники", value=True, key="layer_roads")
    show_s1 = _l2.checkbox("🛰️ Лена со спутника", value=region == "yakutia", key="layer_s1")
    show_basin = _l3.checkbox("🌊 Притоки Лены", value=True, key="layer_basin")
    fmap = go.Figure()
    # 1) зимники OSM — фоном, приглушённо, чтобы не спорить с точками
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
                                     line=dict(width=2, color="#4f7cac"), name="🛣️ зимники и переправы", opacity=0.6))
    # 2) притоки Лены — мелкие бирюзовые точки без подписей
    if show_basin and basin_rows:
        _b = pd.DataFrame(basin_rows)
        fmap.add_trace(go.Scattermap(lat=_b.lat, lon=_b.lon, mode="markers", text=_b.txt, hoverinfo="text",
                                     marker=dict(size=9, color="#0f9d8a", opacity=0.85),
                                     name=f"🌊 притоки Лены ({len(_b)})"))
    # 3) Лена со спутника (М2): белое кольцо + цвет; старше 14 дней — полупрозрачно, дата снимка в подписи
    _pp = Path(PROCESSED).parent / "s1" / "predictions.csv"
    if show_s1 and _pp.exists():
        from okno.s1_points import LENA_NAMES
        _p = pd.read_csv(_pp, parse_dates=["date"]).dropna(subset=["p_ice"])
        _idx = pd.read_csv(Path(PROCESSED).parent / "s1" / "index.csv", usecols=["item_id", "lat", "lon"])
        _s = _p.merge(_idx, on="item_id").sort_values("date").groupby("point").tail(1).copy()
        _s["age"] = (pd.Timestamp(issue_date) - _s.date).dt.days
        _s["state"] = np.where(_s.p_ice >= 0.8, "лёд", np.where(_s.p_ice <= 0.2, "вода", "неясно"))
        _s["col"] = np.where(_s.p_ice >= 0.8, "#bfe9ff", np.where(_s.p_ice <= 0.2, "#0b3d91", "#8aa5c8"))
        _s["hover"] = [f"<b>{LENA_NAMES.get(a, a)}</b><br>Sentinel-1, снимок {d:%d.%m.%Y} ({ag} дн. назад): <b>{st_}</b>"
                       f"<br>P(лёд) = {pi:.2f}" for a, d, ag, st_, pi in zip(_s.point, _s.date, _s.age, _s.state, _s.p_ice)]
        _s["lbl"] = [f"\u00a0\u00a0\u00a0{st_} · {d:%d.%m}" for st_, d in zip(_s.state, _s.date)]
        for fresh, g in _s.groupby(_s.age <= 14):
            op = 1.0 if fresh else 0.35
            fmap.add_trace(go.Scattermap(lat=g.lat, lon=g.lon, mode="markers", hoverinfo="skip", showlegend=False,
                                         marker=dict(size=17, color="#ffffff", opacity=op)))
            fmap.add_trace(go.Scattermap(
                lat=g.lat, lon=g.lon, mode="markers+text", text=g.lbl, hovertext=g.hover, hoverinfo="text",
                textposition="middle right", textfont=dict(size=11, color="#0b3d91" if fresh else "#7a8aa0"),
                marker=dict(size=12, color=list(g.col), opacity=op),
                name="🛰️ Лена со спутника" + ("" if fresh else " (снимок старше 14 дней)")))
    # 4) пути — крупные точки с подписями, поверх всего
    ok = pts[pts.p_ok.notna()] if len(pts) else pts
    na = pts[pts.p_ok.isna()] if len(pts) else pts
    if len(na):
        fmap.add_trace(go.Scattermap(lat=na.lat, lon=na.lon, mode="markers", hovertext=na.txt, hoverinfo="text",
                                     marker=dict(size=15, color="#9e9e9e"), name="🚚 путь: нет данных"))
    if len(ok):
        fmap.add_trace(go.Scattermap(lat=ok.lat, lon=ok.lon, mode="markers", hoverinfo="skip", showlegend=False,
                                     marker=dict(size=24, color="#ffffff")))
        fmap.add_trace(go.Scattermap(
            lat=ok.lat, lon=ok.lon, mode="markers", hovertext=ok.txt, hoverinfo="text", name="🚚 пути: прогноз",
            marker=dict(size=19, color=ok.p_ok, cmin=0, cmax=1, colorscale="RdYlGn",
                        colorbar=dict(title=dict(text="P(путь открыт<br>через N дн.)", font=dict(size=11)),
                                      tickformat=".0%", thickness=12, len=0.55, x=0.99, xanchor="right", y=0.5,
                                      bgcolor="rgba(255,255,255,0.8)"))))
    # подписи путей — отдельными текстовыми слоями «над» и «под» точкой (слой не принимает разные позиции)
    if len(pts):
        _z = REGIONS[region]["zoom"]
        _obst = []
        if show_basin and basin_rows:  # мелкие точки притоков
            for la, lo in zip(_b.lat, _b.lon):
                x, y = _px(la, lo, _z)
                _obst.append((x - 5, y - 5, x + 5, y + 5))
        if show_s1 and _pp.exists():  # точки Лены и их подписи справа
            for la, lo, t in zip(_s.lat, _s.lon, _s.lbl):
                x, y = _px(la, lo, _z)
                _obst += [(x - 8, y - 8, x + 8, y + 8), (x + 8, y - 8, x + 8 + 7 * len(t), y + 8)]
        _lab = place_labels(pts, _z, _obst)
        for where, g in _lab.groupby("textpos"):
            fmap.add_trace(go.Scattermap(lat=g.lat, lon=g.lon, mode="markers+text", text=g.label_txt, textposition=where,
                                         marker=dict(size=26, color="rgba(0,0,0,0)"),  # прозрачная точка задаёт отступ
                                         textfont=dict(size=13, color="#14243a", weight="bold"),
                                         hoverinfo="skip", showlegend=False))
    _c = REGIONS[region]["center"]
    fmap.update_layout(map=dict(style="carto-voyager", center=dict(lat=_c[0], lon=_c[1]), zoom=REGIONS[region]["zoom"]),
                       margin=dict(l=0, r=0, t=0, b=0), height=600, showlegend=True,
                       legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)", font=dict(size=12, color="#222"),
                                   bordercolor="rgba(0,0,0,0.15)", borderwidth=1))
    st.plotly_chart(fmap, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    st.caption(f"Прогноз выпущен на {issue_date:%d.%m.%Y}; вероятность — на "
               f"{(issue_date + dt.timedelta(days=horizon)):%d.%m.%Y}. Состояние Лены — на дату последнего снимка "
               "в каждой точке (подписано рядом с точкой).")

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
                        + (" (здесь нет замеров толщины льда — только спутниковые даты вскрытия)."
                           if (pid == "yakutsk_crossing" or FC.is_basin(pid)) and kind == "open" else "."))
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
            if kind == "close" and pid == "yakutsk_crossing":
                _sp = Path(PROCESSED).parent / "s1" / "predictions.csv"
                if _sp.exists():
                    from okno.m2 import check_against_m1
                    _sn = pd.read_csv(_sp, parse_dates=["date"]).dropna(subset=["p_ice"])
                    _sn = _sn[(_sn.point == pid) & (_sn.date <= pd.Timestamp(issue_date))
                              & (_sn.date >= pd.Timestamp(issue_date) - pd.Timedelta(days=14))]
                    if len(_sn):
                        _o = _sn.sort_values("date").iloc[-1]
                        _st, _txt = check_against_m1(q["q10"], q["q50"], q["q90"],
                                                     to_season_day(_o.date, hy), float(_o.p_ice))
                        _msg = f"М1 и снимок {_o.date:%d.%m}: {_txt}"
                        {"conflict": st.error, "warn": st.warning}.get(_st, st.caption)(_msg)
                    else:
                        st.caption("М1 и снимок: свежего снимка (≤ 14 сут) нет — сверка невозможна")
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
    from app.corridor_view import render_corridor
    render_corridor()

# ------------------------------------------------------------------ 4. планировщик
with tabs[3]:
    from app.planner_view import render_planner
    q = fc_cached(pid, hy, issue_day, "close")
    P = render_planner(q, hy, issue_day, margin, label(pid))
    if q and 'error' not in q and np.isfinite(q.get('actual', np.nan)):
        with st.expander('Прежний ретроспективный тест М3: календарь и пересчёт по прогнозу'):
            st.caption('Отдельная проверка на фактическом закрытии. Здесь введённый объём считается заказом '
                       'на начало сезона, отправки начинаются с января. Это не проверка двух текущих графиков '
                       'для остатка на выбранную дату решения.')
            cal_close = q['calendar'] - margin
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
                    ("Экономический планировщик: сравнение подходов", REPORTS / "economic_planner.md"),
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


with tabs[7]:
    from app.supply_view import render_supply
    render_supply()
