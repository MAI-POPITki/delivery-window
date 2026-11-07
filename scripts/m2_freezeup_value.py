"""
Ценность даты ледостава для прогноза открытия (М1) и роль М2 как её источника.

    python scripts/m2_freezeup_value.py      # → reports/m2_freezeup_value.md, .csv

Зачем. В признаках М1 есть наблюдённая дата ледостава (freezeup_known). В обучении и в бэктесте она берётся
из записей гидропостов, а в живом сервисе её нет (записи постов заканчиваются в 2012 г.) — вместо неё
оценка по погоде. Значит, опубликованная точность может быть выше той, что сервис даёт на деле.
М2 (Sentinel-1) может видеть ледостав в реальном времени — проверяем, сколько это возвращает.

Варианты (одна и та же модель, обученная как в сервисе — с наблюдённым ледоставом; проверка
«сезон целиком вне обучения» на гидропостах Якутии):
  A  «как в отчёте»      — на проверке известен точный ледостав (как в m1_metrics);
  B  «как в сервисе»     — ледостав неизвестен, оценка по погоде;
  C  «сервис + М2»       — ледостав со спутника: истинная дата + ошибка (MAE 7 или 14 сут, 14 — как у
                           автодат М2 на Лене), доступна только через 12 сут после события (снимок раз в ~12 сут);
  D  «без ледостава»     — модель обучена и проверена без наблюдённого ледостава (честная модель сервиса сейчас).
Ошибка спутника моделируется, а не измеряется: на гидропостах нет лет, общих с Sentinel-1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from okno import features as F
from okno import m1
from okno.paths import REPORTS
from okno.report import md_table

LEADS = {"≈14 дней": (11, 17), "≈28 дней": (25, 31), "все": (0, 400)}
REVISIT = 12
_orig = F.season_features


def make_dataset(mode, mae=0.0, seed=0):
    """mode: obs | nan | sat. Подменяет наблюдённый ледостав в признаках."""
    sigma = mae / 0.7979 if mae else 0.0  # MAE нормального распределения = 0,798 σ
    noise = {}

    def patched(T, S, y, d, clim, alpha, fz_obs, theta=np.nan, t_ahead=None):
        if mode == "nan":
            fz_obs = np.nan
        elif mode == "sat" and not np.isnan(fz_obs):
            key = (id(T), y)
            if key not in noise:
                noise[key] = np.random.default_rng(hash((seed, y, T.shape)) % 2**32).normal(0, sigma)
            fz_obs = fz_obs + noise[key] if fz_obs + REVISIT <= d else np.nan
        return _orig(T, S, y, d, clim, alpha, fz_obs, theta, t_ahead)

    F.season_features = patched
    try:
        return F.build_m1_dataset("open")
    finally:
        F.season_features = _orig


def cross_backtest(train_df, test_df):
    """LOSO: обучение на train_df без сезона y, прогноз строк test_df сезона y."""
    parts = []
    for y in sorted(test_df.hydro_year.unique()):
        model = m1.fit(train_df[train_df.hydro_year != y], "open")
        t = test_df[test_df.hydro_year == y]
        parts.append(pd.concat([t, m1.predict(model, t, "open")], axis=1))
    return pd.concat(parts)


def summarize(bt, name):
    rows = []
    for lead, (lo, hi) in LEADS.items():
        s = bt[bt.lead_days.between(lo, hi)]
        cal = (s.cal_target - s.target_day).abs().mean()
        mod = (s.q50 - s.target_day).abs().mean()
        rows.append({"вариант": name, "за сколько": lead, "прогнозов": len(s), "MAE календарь": round(cal, 2),
                     "MAE М1": round(mod, 2), "выигрыш к календарю, %": round(100 * (1 - mod / cal), 1)})
    return rows


def main():
    obs = make_dataset("obs")
    nan = make_dataset("nan")
    key = ["point_id", "hydro_year", "issue_day"]
    assert (obs[key].to_numpy() == nan[key].to_numpy()).all()
    rows = []
    rows += summarize(cross_backtest(obs, obs), "A. как в отчёте: точный ледостав")
    rows += summarize(cross_backtest(obs, nan), "B. как в сервисе: ледостав неизвестен")
    for mae in (7, 14):
        sat = make_dataset("sat", mae=mae)
        rows += summarize(cross_backtest(obs, sat), f"C. сервис + М2: ледостав со спутника, ошибка {mae} сут")
    rows += summarize(cross_backtest(nan, nan), "D. модель без ледостава (обучение и проверка)")
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "m2_freezeup_value.csv", index=False)
    lines = ["# Дата ледостава для прогноза открытия: сколько стоит и может ли её дать М2", "",
             "Прогноз открытия (лёд держит 20 т), гидропосты Якутии, проверка «сезон целиком вне обучения». "
             "Модель обучена как в сервисе (с наблюдённым ледоставом), кроме варианта D. Ошибка спутника в C "
             "смоделирована (на постах нет лет, общих с Sentinel-1); ледостав со спутника доступен через 12 сут "
             "после события.", "", md_table(res), "",
             "**Как читать.** A — точность, опубликованная в m1_metrics. B — что сервис даёт сейчас, когда "
             "записей постов о ледоставе за текущий сезон нет. Разница A − B — сколько точности держится "
             "на дате ледостава. C — сколько из этого вернёт спутник. D — модель, обученная без этого признака "
             "(если она не хуже B, сервис стоит перевести на неё).", ""]
    (REPORTS / "m2_freezeup_value.md").write_text("\n".join(lines), encoding="utf-8")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
