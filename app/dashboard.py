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

# Старые ссылки открывают общую рабочую вкладку, не отдельные страницы.
_requested_plan = st.query_params.get("module") in ("routes", "backhaul", "journey")
if _requested_plan and not st.session_state.get("journey_link_processed"):
    st.session_state["dashboard_open"] = True
    st.session_state["main_tab"] = "Риски и рекомендации"
    st.session_state["journey_link_processed"] = True

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
    "okno.paths", "okno.northeast", "okno.regions", "okno.report", "okno.ice", "okno.calib", "okno.features", "okno.snow",
    "okno.planner", "okno.decision_support", "app.decision_view", "okno.truck_load", "okno.backhaul", "okno.backhaul_io", "okno.routing", "okno.journey", "okno.risk_advice", "app.truck_load_view", "app.planner_view", "okno.economic_planner", "okno.corridor", "okno.corridor_data", "okno.bottlenecks", "app.corridor_view", "okno.m1", "okno.weather_forecast", "okno.forecast",
    "okno.yakutsk_replay", "app.yakutsk_experiment", "app.backhaul_view", "app.journey_view",
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
def _fc_cached(pid, hy, day, kind, ver, use_weather, use_snow):
    try:
        return FC.forecast(pid, hy, day, kind, use_weather=use_weather,use_snow=use_snow)
    except Exception as e:  # noqa: BLE001 — в дашборде показываем, а не падаем
        return {"error": str(e)}


def fc_cached(pid, hy, day, kind):
    return _fc_cached(pid, hy, day, kind, DATA_VER, use_weather_close,use_snow_depth)


@st.cache_data(show_spinner=False, max_entries=500, ttl=3600)
def _fc_path(pid, hy, kind, days, ver, use_weather,use_snow):
    rows = [_fc_cached(pid, hy, d, kind, ver, use_weather,use_snow) for d in days]
    return pd.DataFrame([r for r in rows if r and "error" not in r])


def fc_path(pid, hy, kind, days):
    return _fc_path(pid, hy, kind, days, DATA_VER, use_weather_close,use_snow_depth)


@st.cache_data(show_spinner=False)
def _journal(ver):
    """Готовые прогнозы из ежедневного журнала (scripts/forecast_journal.py, GitHub Actions):
    ключ (point_id, hydro_year, kind, issue_date) → строка; только строки с текущими моделями М1."""
    import hashlib
    from okno.paths import DATA, MODELS
    p = DATA / "forecast_log" / "forecasts.csv"
    if not p.exists():
        return {}
    j = pd.read_csv(p, dtype=str)
    if "model_file" not in j.columns:
        j["model_file"] = ""
    j["model_file"] = j["model_file"].fillna("")
    files = j["model_file"].where(j["model_file"] != "", "m1_" + j["kind"] + ".cbm")
    tags = {f: hashlib.sha1((MODELS / f).read_bytes()).hexdigest()[:8]
            for f in files.unique() if (MODELS / f).exists()}
    j = j[j.model == files.map(tags)]
    if "snow_applied" in j.columns:  # карта по умолчанию без снежной поправки
        j = j[j["snow_applied"].astype(str) != "True"]
    j = j.sort_values("run_date")
    return {(r.point_id, int(r.hydro_year), r.kind, r.issue_date): r._asdict()
            for r in j.itertuples(index=False)}


def fc_map(pid, hy, day, kind):
    """Прогноз для карты: если на эту дату он уже есть в журнале (те же модели и данные) — берём готовый,
    иначе считаем. Совпадение даты выпуска гарантирует, что прогноз построен по тем же наблюдениям."""
    if not (kind == "close" and use_weather_close) and not use_snow_depth:
        last = FC.last_weather_day(pid, hy)  # прогноз выпускается не позже последнего дня наблюдений точки
        eff = from_season_day(min(day, last) if last is not None else day, hy).date()
        r = _journal(DATA_VER).get((pid, hy, kind, eff.isoformat()))
        if r is not None:
            q = {"point_id": pid, "hydro_year": hy, "kind": kind, "from_journal": True,
                 "issue_day": int(to_season_day(eff, hy))}
            for k in ("q10", "q50", "q90", "calendar", "physics"):
                v = r.get(f"{k}_day")  # точное значение; в старых строках журнала — только дата
                q[k] = float(v) if isinstance(v, str) and v not in ("", "nan") else float(to_season_day(pd.Timestamp(r[f"{k}_date"]), hy))
                q[f"{k}_date"] = from_season_day(q[k], hy).date()
            return q
    return fc_cached(pid, hy, day, kind)


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


@st.cache_data(show_spinner=False)
def measured_ice_snow(code,ver):
    path=PROCESSED/'hydro_ice_thickness.csv'
    d=pd.read_csv(path,usecols=['gauge_code','hydro_year','date','ice_cm','snow_on_ice_cm','author_interpolated'])
    return d[d.gauge_code.eq(code)].sort_values('date')


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
use_snow_depth=st.sidebar.checkbox('Учитывать толщину снега (эксперимент)',value=False,key='snow_depth_enabled',
    help='Для прогноза М1 на карте, в карточке и планировщике М3. '
         'Сквозной коридор А3 использует отдельный прогноз. Различает снег на льду и снег на суше из ERA5-Land. '
         'Использует только данные до даты решения; с погодным экспериментом GFS поправки не складываются.')
if use_weather_close:
    st.sidebar.caption("Эксперимент: прогноз GFS на 6 дней. Выигрыш за 7/14 дней не подтверждён, "
                       "диапазон пока ненадёжен. Результаты — во вкладке «До / после».")
last = FC.last_weather_day(pid, hy)
if last is not None and last < issue_day:
    st.sidebar.warning(f"Погода есть до {from_season_day(last, hy).date():%d.%m.%Y}: прогноз выпускается на эту дату.")

if st.session_state.get("main_tab") in ("Рейтинг риска", "Маршрут и план завоза"):  # прежние названия
    st.session_state["main_tab"] = "Риски и рекомендации"

tabs = st.tabs([
    "Карта путей",
    "Карточка пути",
    "Сквозной коридор (А3)",
    "Планировщик",
    "Риски и рекомендации",
    "До / после",
    "О данных",
], key="main_tab", on_change="rerun")
# Streamlit выполняет код всех вкладок сразу; тяжёлые (коридор, рейтинг риска) считаем только когда
# вкладка открыта — первое открытие дашборда не ждёт расчётов, которые пользователь может и не смотреть.
def _tab_open(i):
    return tabs[i].open is not False

# ------------------------------------------------------------------ 1. карта
with tabs[0]:
    st.subheader(f"Вероятность, что путь открыт через {horizon} дн. — {season_name(hy)}")
    t = issue_day + horizon
    rows = []
    basin_rows = []
    for p in WEATHER_POINTS:
        meta = FC.point_meta(p)
        if FC.is_basin(p):  # притоки Лены: только прогноз вскрытия — отдельный слой
            qc = fc_map(p, hy, issue_day, "close")
            if qc and "error" not in qc:
                basin_rows.append({"lat": meta["lat"], "lon": meta["lon"],
                                   "txt": f"<b>{label(p)}</b><br>Вскрытие (М1 для притоков, 50%): {qc['q50_date']:%d.%m}, "
                                          f"диапазон {qc['q10_date']:%d.%m}–{qc['q90_date']:%d.%m}"})
            continue
        qo, qc = fc_map(p, hy, issue_day, "open"), fc_map(p, hy, issue_day, "close")
        p_open = p_by(qo, t)
        p_open = 1.0 if np.isnan(p_open) and issue_day > 120 else p_open
        p_closed = p_by(qc, t + margin)
        p_ok = p_open * (1 - p_closed) if not (np.isnan(p_open) or np.isnan(p_closed)) else np.nan
        color = ("#9e9e9e" if np.isnan(p_ok) else
                 "#2e7d32" if p_ok >= 0.8 else "#ef8c00" if p_ok >= 0.5 else "#c62828")
        txt = f"<b>{label(p)}</b><br>P(открыт через {horizon} дн.) = " + ("н/д" if np.isnan(p_ok) else f"{p_ok:.0%}")
        if qc and "error" not in qc:
            txt += f"<br>Таяние (М1, 50%): {qc['q50_date']:%d.%m}, диапазон {qc['q10_date']:%d.%m}–{qc['q90_date']:%d.%m}"
        rows.append({"lat": meta["lat"], "lon": meta["lon"], "color": color, "txt": txt, "p_ok": p_ok, "pid": p})
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
    show_s1 = _l2.checkbox("🛰️ Лена со спутника", value=False, key="layer_s1")
    show_basin = _l3.checkbox("🌊 Притоки Лены", value=False, key="layer_basin")
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
        # Подписи всегда тёмные и непрозрачные: давность передаётся цветом точки, а не бледным текстом.
        for fresh, g in _s.groupby(_s.age <= 14):
            op = 1.0 if fresh else 0.35
            fmap.add_trace(go.Scattermap(lat=g.lat, lon=g.lon, mode="markers", hoverinfo="skip", showlegend=False,
                                         marker=dict(size=17, color="#ffffff", opacity=op)))
            fmap.add_trace(go.Scattermap(
                lat=g.lat, lon=g.lon, mode="markers+text", text=g.lbl, hovertext=g.hover, hoverinfo="text",
                textposition="middle right", textfont=dict(size=13, color="#17212b", weight="bold"),
                marker=dict(size=12, color=list(g.col), opacity=op),
                name="🛰️ Лена со спутника" + ("" if fresh else " (снимок старше 14 дней)")))
    # 4) пути — крупные точки с подписями, поверх всего
    ok = pts[pts.p_ok.notna()] if len(pts) else pts
    na = pts[pts.p_ok.isna()] if len(pts) else pts
    if len(na):
        fmap.add_trace(go.Scattermap(lat=na.lat, lon=na.lon, mode="markers", hovertext=na.txt, hoverinfo="text",
                                     customdata=na[["pid"]], marker=dict(size=15, color="#9e9e9e"), name="🚚 путь: нет данных"))
    if len(ok):
        fmap.add_trace(go.Scattermap(lat=ok.lat, lon=ok.lon, mode="markers", hoverinfo="skip", showlegend=False,
                                     marker=dict(size=24, color="#ffffff")))
        fmap.add_trace(go.Scattermap(
            lat=ok.lat, lon=ok.lon, customdata=ok[["pid"]], mode="markers", hovertext=ok.txt, hoverinfo="text", name="🚚 пути: прогноз",
            marker=dict(size=19, color=ok.p_ok, cmin=0, cmax=1, colorscale="RdYlGn",
                        colorbar=dict(title=dict(text="P(путь открыт<br>через N дн.)", font=dict(size=11, color="#17212b")),
                                      tickfont=dict(color="#17212b"), tickformat=".0%", thickness=12, len=0.55, x=0.99, xanchor="right", y=0.5,
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
            fmap.add_trace(go.Scattermap(lat=g.lat, lon=g.lon, mode="markers+text", customdata=g[["pid"]], text=g.label_txt, textposition=where,
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
    from okno.northeast import STATIONS as _new_stations
    is_new_gauge=pid.startswith('gauge_') and int(pid.split('_')[1]) in _new_stations
    if is_new_gauge:
        st.caption('Это гидропост — точка измерения льда и снега, а не трасса зимника. '
                   'Достижение расчётной толщины льда не означает разрешение на проезд.')
    c1, c2 = st.columns(2)
    for kind, col, title in (("open", c1, "Открытие: лёд держит 20 т"), ("close", c2, "Закрытие: начало таяния")):
        q = fc_cached(pid, hy, issue_day, kind)
        with col:
            st.markdown(f"**{title}**")
            if not q or "error" in q:
                if is_new_gauge and kind=='close':
                    st.info('В новом источнике нет дат начала таяния. Для прогноза закрытия нужен дополнительный ряд ледовых фаз.')
                    continue
                st.info("Нет прогноза для этой точки и вида события"
                        + (" (здесь нет замеров толщины льда — только спутниковые даты вскрытия)."
                           if (pid == "yakutsk_crossing" or FC.is_basin(pid)) and kind == "open" else "."))
                continue
            close_note = f" (закрытие пути ≈ на {margin} сут раньше)" if kind == "close" else ""
            st.metric("Медиана прогноза М1", f"{q['q50_date']:%d.%m.%Y}",
                      delta=f"{q['q50'] - q['calendar']:+.0f} сут к календарю", delta_color="inverse")
            range_name = "Экспериментальный диапазон" if q.get("weather_model_applied") or q.get('snow_model_applied') or q.get('updated_model_applied') else "80%-диапазон"
            st.write(f"{range_name}: **{q['q10_date']:%d.%m} – {q['q90_date']:%d.%m}**{close_note}")
            st.write(f"Календарь (средняя за прошлые сезоны): {q['calendar_date']:%d.%m}; "
                     f"физика: {q['physics_date']:%d.%m}")
            st.write(f"Расчётная толщина льда на дату решения: **{q['h_stefan_now_cm']:.0f} см**")
            if use_snow_depth:
                st.caption(q.get('snow_model_message',''))
                if q.get('snow_model_applied'):
                    sf=q['snow_features']
                    st.write(f"Сдвиг при учёте снега: **{q['snow_delta_days']:+.1f} сут** к обычной М1.")
                    if np.isfinite(sf.get('snow_ground_cm',np.nan)):
                        st.caption(f"Снег на суше по ERA5-Land: {sf['snow_ground_cm']:.1f} см (не замер на реке).")
                    if sf.get('snow_on_ice_observed'):
                        st.caption(f"Последний замер снега на льду: {sf['snow_on_ice_cm']:.1f} см, {sf['snow_on_ice_age_days']:.0f} суток назад.")
            if kind == "close" and use_weather_close:
                st.caption(q["weather_model_message"])
                if q["weather_model_applied"]:
                    st.write(f"Сдвиг к обычной М1: **{q['weather_delta_days']:+.1f} сут**. "
                             f"Прогноз погоды учтён на {q['forecast_days_used']} дней.")
            if q["actual_date"]:
                fact_label='Оценка достижения толщины по замерам' if is_new_gauge else 'Факт'
                st.success(f"{fact_label}: {q['actual_date']:%d.%m.%Y}")
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
    if is_new_gauge:
        with st.expander('Архив измерений льда и снега на посту'):
            observed=measured_ice_snow(int(pid.split('_')[1]),DATA_VER)
            selected=observed[observed.hydro_year.eq(hy)]
            if selected.empty:
                archive_year=int(observed.hydro_year.max())
                selected=observed[observed.hydro_year.eq(archive_year)]
                st.caption(f'За выбранный сезон замеров нет. Показан последний архивный сезон {season_name(archive_year)}; '
                           'эти старые значения не подставляются как текущий снег.')
            st.caption('PANGAEA: лёд и снег на льду, сантиметры. Интерполированные авторами значения льда исключены из обучения.')
            view=selected[['date','ice_cm','snow_on_ice_cm']].rename(columns={
                'date':'Дата замера','ice_cm':'Лёд, см','snow_on_ice_cm':'Снег на льду, см'})
            st.dataframe(view,hide_index=True,width='stretch',key='snow_measurements_archive')
            st.markdown('[Источник измерений и лицензия](https://doi.org/10.1594/PANGAEA.925460)')
    # как менялся прогноз закрытия в течение сезона
    days = list(range(150, 271, 7))
    path = fc_path(pid, hy, "close", tuple(days))
    if not path.empty:
        x = [from_season_day(d, hy) for d in path.issue_day]
        f = go.Figure()
        experimental=path[['weather_model_applied','snow_model_applied','updated_model_applied']].any().any()
        band_name='Экспериментальный диапазон М1' if experimental else '80%-диапазон М1'
        f.add_trace(go.Scatter(x=x + x[::-1], y=[from_season_day(v, hy) for v in list(path.q90) + list(path.q10[::-1])],
                               fill="toself", line=dict(width=0), name=band_name, opacity=0.3))
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
    if _tab_open(2):
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

# ------------------------------------------------------------------ 5. риски и рекомендации (ZXC)
with tabs[4]:
    if _tab_open(4):
        from app.journey_view import render_journey
        render_journey(pid, hy, issue_date, issue_day, margin, WEATHER_POINTS, fc_cached)

    # ------------------------------------------------------------------ 6. до/после
with tabs[5]:
    st.subheader("Бэктест на прошлых сезонах")
    for name, f in (("Сквозной коридор 4 звеньев: «До и После» (Кейс А3)", REPORTS / "corridor_before_after.md"),
                    ("Экономический планировщик: сравнение подходов", REPORTS / "economic_planner.md"),
                    ("Толщина снега: проверка новой модели", REPORTS / "snow_model.md"),
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
