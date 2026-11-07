"""
Проверки «как сделать прогноз точнее на открытых данных» (без утечки будущего).

    python scripts/open_data_improvements.py   # → reports/open_data_improvements.md, .csv

1. Снег в законе Стефана. Снег ERA5 (snow_depth_mean, Open-Meteo archive) утепляет лёд:
   h_{t+1} = h_t + a² · FDD_t / (2 (h_t + r · s_t)),   где s_t — снег, см; r = k_лёд / k_снег.
   При r = 0 это обычный закон Стефана h = a·√FDD. a и r подбираются по замерам постов
   (5 блоков сезонов: сезоны проверки в подборе не участвуют).
2. Прогноз сразу официальной даты открытия («погода и лёд → приказ»): ridge-регрессия аномалии
   «приказ − календарь дороги» по признакам М1 и снегу на дату выпуска; обучение только на сезонах
   hydro_year < целевого. Соперники — «календарь дороги», «как в прошлом году», «М1 + сдвиг».
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests

from okno.ice import load_weather
from okno.paths import DATA, PROCESSED, REPORTS
from okno.report import md_table

SNOW = DATA / "weather_snow"
API = "https://archive-api.open-meteo.com/v1/archive"


def coords(pid):
    if pid == "yakutsk_crossing":
        return 62.0, 129.82
    g = pd.read_csv(PROCESSED / "gauges.csv").set_index("gauge_code")
    r = g.loc[int(pid.split("_")[1])]
    return float(r.lat), float(r.lon)


def snow(pid):
    """Суточная высота снега ERA5, см (кэш в data/weather_snow)."""
    SNOW.mkdir(parents=True, exist_ok=True)
    f = SNOW / f"{pid}.csv"
    if not f.exists():
        la, lo = coords(pid)
        for a in range(4):
            r = requests.get(API, params={"latitude": la, "longitude": lo, "start_date": "1979-01-01",
                                          "end_date": "2026-10-31", "daily": "snow_depth_mean", "timezone": "GMT"},
                             timeout=120)
            if r.status_code == 200:
                break
            time.sleep(5 * (a + 1))
        r.raise_for_status()
        d = r.json()["daily"]
        pd.DataFrame({"date": d["time"], "snow_cm": np.array(d["snow_depth_mean"], dtype=float) * 100}).to_csv(f, index=False)
    s = pd.read_csv(f, parse_dates=["date"])
    return s.set_index("date").snow_cm.fillna(0.0)


def grow(fdd, s, a, r):
    h = np.zeros(len(fdd))
    cur = 0.0
    for i in range(len(fdd)):
        cur += a * a * fdd[i] / (2.0 * max(cur + r * s[i], 1.0))
        h[i] = cur
    return h


# ---------- 1. снег в законе Стефана ----------
def stefan_pairs_snow():
    seasons = pd.read_csv(PROCESSED / "hydro_ice_seasons.csv")
    thick = pd.read_csv(PROCESSED / "hydro_ice_thickness.csv", parse_dates=["date"])
    gauges = pd.read_csv(PROCESSED / "gauges.csv")
    out = []
    for code in gauges.gauge_code:
        pid = f"gauge_{code}"
        try:
            w = load_weather(pid).set_index("date").sort_index()
            sn = snow(pid)
        except Exception as e:  # нет погоды/снега — пропускаем пост
            print(pid, "пропуск:", type(e).__name__)
            continue
        s = seasons[(seasons.gauge_code == code) & seasons.valid & seasons.freezeup_date.notna()]
        th = thick[(thick.gauge_code == code) & thick.ice_cm.notna() & thick.date.dt.month.isin([10, 11, 12, 1, 2, 3])]
        for r in s.itertuples():
            fz = pd.Timestamp(r.freezeup_date)
            m = th[(th.hydro_year == r.hydro_year) & (th.date > fz)]
            if m.empty or fz < w.index.min() or m.date.max() > w.index.max():
                continue
            days = pd.date_range(fz, m.date.max())
            out.append({"gauge": code, "hy": r.hydro_year, "days": days,
                        "fdd": np.maximum(0, -w.t.reindex(days).fillna(0).to_numpy()),
                        "snow": sn.reindex(days).fillna(0).to_numpy(),
                        "obs_idx": ((m.date - fz).dt.days - 1).clip(lower=0).to_numpy(),
                        "obs": m.ice_cm.to_numpy()})
    return out


def pack(seas):
    L = max(len(x["fdd"]) for x in seas)
    F = np.zeros((len(seas), L)); S = np.zeros((len(seas), L))
    for i, x in enumerate(seas):
        F[i, :len(x["fdd"])] = x["fdd"]; S[i, :len(x["snow"])] = x["snow"]
    rows = np.concatenate([[i] * len(x["obs"]) for i, x in enumerate(seas)]).astype(int)
    cols = np.concatenate([x["obs_idx"] for x in seas]).astype(int)
    obs = np.concatenate([x["obs"] for x in seas])
    return F, S, rows, cols, obs


def grow_all(F, S, a, r):
    H = np.zeros_like(F); cur = np.zeros(F.shape[0])
    for t in range(F.shape[1]):
        cur = cur + a * a * F[:, t] / (2.0 * np.maximum(cur + r * S[:, t], 1.0))
        H[:, t] = cur
    return H


def fit(seas, rs=(0.0, 1.0, 2.5, 5.0, 7.5, 10.0)):
    F, S, ri, ci, obs = pack(seas)
    best = None
    for r in rs:
        for a in np.arange(1.0, 4.01, 0.1):
            mae = np.abs(grow_all(F, S, a, r)[ri, ci] - obs).mean()
            if best is None or mae < best[2]:
                best = (a, r, mae)
    return best


def snow_test():
    seas = stefan_pairs_snow()
    years = np.array(sorted({x["hy"] for x in seas}))
    folds = np.array_split(years, 5)
    rows = []
    for fy in folds:
        fy = set(fy)
        tr, te = [x for x in seas if x["hy"] not in fy], [x for x in seas if x["hy"] in fy]
        a0, _, _ = fit(tr, rs=(0.0,))
        a1, r1, _ = fit(tr)
        F, S, ri, ci, obs = pack(te)
        p0, p1 = grow_all(F, S, a0, 0.0)[ri, ci], grow_all(F, S, a1, r1)[ri, ci]
        for k in range(len(obs)):
            x = te[ri[k]]
            rows.append({"gauge": x["gauge"], "hydro_year": x["hy"], "obs": obs[k], "plain": p0[k], "snow": p1[k],
                         "snow_cm": S[ri[k], ci[k]], "r": r1})
    return pd.DataFrame(rows)


# ---------- 2. прямой прогноз официальной даты ----------
def official_direct():
    df = pd.read_csv(REPORTS / "official_dates.csv")
    df = df[(df.kind == "open") & df.road_cal.notna()].copy()
    snow_feat = []
    for r in df.itertuples():
        try:
            sn = snow(r.point_id)
            d = pd.Timestamp(int(r.hydro_year) - 1, 9, 1) + pd.Timedelta(days=int(r.issue_day))
            snow_feat.append(float(sn.get(d, np.nan)))
        except Exception:
            snow_feat.append(np.nan)
    df["snow_now"] = snow_feat
    df["m1_anom"] = df.m1 - df.post_cal
    feats = ["m1_anom", "snow_now", "issue_day"]
    pred = []
    for r in df.itertuples():
        tr = df[(df.hydro_year < r.hydro_year) & (df.issue == r.issue)].dropna(subset=feats)
        if len(tr) < 8 or not np.isfinite(r.snow_now):
            pred.append(np.nan)
            continue
        X = tr[feats].to_numpy(float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = np.c_[np.ones(len(tr)), (X - mu) / sd]
        y = (tr.actual - tr.road_cal).to_numpy()
        lam = np.diag([0] + [5.0] * len(feats))
        b = np.linalg.solve(Xs.T @ Xs + lam, Xs.T @ y)
        x = np.r_[1, (np.array([getattr(r, f) for f in feats], float) - mu) / sd]
        pred.append(r.road_cal + x @ b)
    df["direct"] = pred
    return df


def boot(d, a, b, n=2000):
    rng = np.random.default_rng(0)
    yrs = d.hydro_year.unique()
    g = {y: d[d.hydro_year == y] for y in yrs}
    res = []
    for _ in range(n):
        s = pd.concat([g[y] for y in rng.choice(yrs, len(yrs))])
        res.append((s[b] - s.actual).abs().mean() - (s[a] - s.actual).abs().mean())
    return np.percentile(res, [5, 95])


def main():
    out = []
    st = snow_test()
    st.to_csv(REPORTS / "open_data_snow.csv", index=False)
    e0, e1 = (st.plain - st.obs).abs(), (st.snow - st.obs).abs()
    deep = st.snow_cm >= 20
    out.append({"проверка": "снег в законе Стефана (толщина, см)", "n": len(st), "сезонов": st.hydro_year.nunique(),
                "MAE было": round(e0.mean(), 1), "MAE стало": round(e1.mean(), 1),
                "выигрыш, %": round(100 * (1 - e1.mean() / e0.mean()), 1),
                "комментарий": f"при снеге ≥ 20 см: {e0[deep].mean():.1f} → {e1[deep].mean():.1f} см (n={int(deep.sum())}); "
                               f"r = k_лёд/k_снег (медиана) {st.r.median():.1f}"})
    od = official_direct()
    od.to_csv(REPORTS / "open_data_official_direct.csv", index=False)
    d = od.dropna(subset=["direct", "last_year", "m1_adj"])
    for base, name in [("road_cal", "календарь дороги"), ("last_year", "как в прошлом году"), ("m1_adj", "М1 + сдвиг")]:
        lo, hi = boot(d, "direct", base)
        eb, ed = (d[base] - d.actual).abs().mean(), (d.direct - d.actual).abs().mean()
        out.append({"проверка": f"официальное открытие: «погода и лёд → приказ» против «{name}» (сут)", "n": len(d),
                    "сезонов": d.hydro_year.nunique(), "MAE было": round(eb, 1), "MAE стало": round(ed, 1),
                    "выигрыш, %": round(100 * (1 - ed / eb), 1),
                    "комментарий": f"90%-интервал выигрыша {lo:+.1f} … {hi:+.1f} сут"})
    res = pd.DataFrame(out)
    res.to_csv(REPORTS / "open_data_improvements.csv", index=False)
    lines = ["# Прогноз точнее на открытых данных: что проверено", "",
             "Метод и правила честности — в шапке `scripts/open_data_improvements.py`. «Было» — текущий способ, "
             "«стало» — с новыми открытыми данными; выигрыш > 0 — стало точнее.", "", md_table(res), "",
             "## Доступность остальных открытых данных (проверено 07.11.2026)", "",
             "- **Свежие данные гидропостов:** gmvo.skniivh.ru открыт, но сообщает «ресурс выведен из эксплуатации… "
             "ГИС ЦП Вода закрытого контура» (доступ по заявке и с криптозащитой). Открыто скачать замеры после 2012 г. нельзя.",
             "- **ECMWF до 15 суток:** архив Open-Meteo Previous Runs (`ecmwf_ifs025`) отдаёт заблаговременность до 14 сут "
             "с 2024 г. (в марте 2024 — только 1 сут). Для проверки «за месяц до открытия» хватает 1–2 сезонов — пока мало.",
             "- **Сезонный C3S / SEAS5:** Open-Meteo Seasonal (`ecmwf_seas5`, 51 член ансамбля) доступен только с "
             "2025-09-01 — один сезон, честной проверки на нём не сделать; подключить можно в живом режиме зимы 2026/27.",
             "- **AMSR2 ежедневно:** Бременский университет открыто отдаёт суточные яркостные температуры AMSR2 "
             "(сетка 6,25 и 12,5 км, 2012–2026, HDF4; читается без спецбиблиотек). Пиксель 6–12 км шире Лены у Якутска "
             "(~3–10 км с протоками), поэтому сигнал «лёд/вода» есть только по крупным плёсам; против Sentinel-1 "
             "раз в 12 дней это сокращает задержку сторожа М2 с ~6 до ~1 сут в среднем, но с грубым разрешением. "
             "Модель по AMSR2 ещё не обучена.", ""]
    (REPORTS / "open_data_improvements.md").write_text("\n".join(lines), encoding="utf-8")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
