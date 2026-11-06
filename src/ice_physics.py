"""
Физическая модель нарастания и разрушения речного льда (`src/ice_physics.py`, Кейс А3).

Реализует расчёт накопленных отрицательных градусо-дней мороза (FDD — Freezing Degree Days),
положительных градусо-дней весеннего таяния (TDD — Thawing Degree Days), оценку толщины
льда по уравнению Стефана:
    h(t) = alpha * sqrt( sum_{tau = t_freezeup}^{t} max(0, -T_tau) )
и калибровку коэффициента alpha по реальным замерам толщины льда на гидропостах Росгидромета.

Запуск:
    python src/ice_physics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from okno.calib import calibrate_all, fit_alpha, stefan_pairs
from okno.ice import (
    freezeup_from_weather,
    from_season_day,
    load_weather,
    required_thickness_cm,
    season_weather,
    stefan_thickness,
)
from okno.paths import PROCESSED, REPORTS

SPRING_START_DAY = 181  # 1 марта (0 = 1 сентября)


def compute_degree_days(season_df: pd.DataFrame, freezeup_day: int | None = None,
                        spring_start_day: int = SPRING_START_DAY) -> pd.DataFrame:
    """Рассчитывает посуточные ряды FDD, TDD и толщины льда Стефана для одного сезона."""
    df = season_df.sort_values("season_day").reset_index(drop=True).copy()
    t = df["t"].to_numpy(dtype=float)
    days = df["season_day"].to_numpy(dtype=int)

    if freezeup_day is None:
        fz_est = freezeup_from_weather(df)
        freezeup_day = fz_est if fz_est is not None else 45

    frost = np.where(days >= freezeup_day, np.maximum(0.0, -t), 0.0)
    thaw = np.where(days >= spring_start_day, np.maximum(0.0, t), 0.0)

    df["fdd_cum"] = np.cumsum(frost)
    df["tdd_cum"] = np.cumsum(thaw)
    df["freezeup_day_used"] = int(freezeup_day)
    return df


def stefan_ice_curve(season_df: pd.DataFrame, alpha: float,
                     freezeup_day: int | None = None) -> pd.DataFrame:
    """Добавляет расчётную толщину льда h_stefan_cm по формуле Стефана h(t) = alpha * sqrt(FDD(t))."""
    df = compute_degree_days(season_df, freezeup_day=freezeup_day)
    df["h_stefan_cm"] = alpha * np.sqrt(np.maximum(df["fdd_cum"].to_numpy(dtype=float), 0.0))
    # В фазе активного весеннего таяния эффективная несущая толщина снижается пропорционально TDD
    # (эмпирический коэффициент стаивания ~0.45 см на 1 °C·сут положительной температуры)
    df["h_effective_cm"] = np.maximum(0.0, df["h_stefan_cm"] - 0.45 * df["tdd_cum"])
    return df


def calibrate_stefan_alpha(gauge_codes: list[int] | None = None) -> tuple[dict, pd.DataFrame]:
    """Калибрует коэффициент Стефана alpha (см / (°C·сут)^0.5) по архиву замеров Росгидромета."""
    pooled, per_gauge, _ = calibrate_all(gauge_codes=gauge_codes)
    if pooled is None:
        raise RuntimeError("Не удалось откалибровать alpha: отсутствуют данные погоды или замеров гидропостов.")
    return pooled, per_gauge


def summarize_yakutsk_physics(start_year: int = 2001, end_year: int = 2024) -> pd.DataFrame:
    """Считает физические показатели зимне-весеннего режима переправы Якутск — Нижний Бестях."""
    pooled, _ = calibrate_stefan_alpha()
    alpha = float(pooled["alpha"])
    w = load_weather("yakutsk_crossing")
    sat = pd.read_csv(PROCESSED / "yakutsk_sat_seasons.csv").set_index("hydro_year")
    h_req_20t = required_thickness_cm(20.0)
    h_req_30t = required_thickness_cm(30.0)

    rows = []
    for hy in range(start_year, end_year + 1):
        sw = season_weather(w, hy)
        if len(sw) < 300:
            continue
        fz_day = int(round(sat.loc[hy, "sat_freezeup_day"])) if hy in sat.index else None
        curve = stefan_ice_curve(sw, alpha=alpha, freezeup_day=fz_day)

        idx_20t = curve[curve["h_stefan_cm"] >= h_req_20t]
        idx_30t = curve[curve["h_stefan_cm"] >= h_req_30t]
        open_20t_day = int(idx_20t["season_day"].iloc[0]) if not idx_20t.empty else np.nan
        open_30t_day = int(idx_30t["season_day"].iloc[0]) if not idx_30t.empty else np.nan
        max_fdd = float(curve["fdd_cum"].max())
        max_h = float(curve["h_stefan_cm"].max())
        may1_tdd = float(curve.loc[curve["season_day"] <= 242, "tdd_cum"].max())

        rows.append({
            "hydro_year": hy,
            "alpha_used": round(alpha, 3),
            "freezeup_day": curve["freezeup_day_used"].iloc[0],
            "max_fdd_deg_days": round(max_fdd, 0),
            "max_stefan_ice_cm": round(max_h, 1),
            "phys_open_20t_date": from_season_day(open_20t_day, hy).date().isoformat() if pd.notna(open_20t_day) else None,
            "phys_open_30t_date": from_season_day(open_30t_day, hy).date().isoformat() if pd.notna(open_30t_day) else None,
            "tdd_by_may1": round(may1_tdd, 1),
        })
    out = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / "yakutsk_ice_physics_2001_2024.csv", index=False)
    return out


def main() -> None:
    pooled, per_gauge = calibrate_stefan_alpha()
    yak_phys = summarize_yakutsk_physics(2001, 2024)
    print("=== Калибровка уравнения Стефана по гидропостам Росгидромета (Якутия) ===")
    print(
        f"Общий коэффициент alpha = {pooled['alpha']:.3f} см/(°C·сут)^0.5 | "
        f"MAE = {pooled['mae_cm']:.1f} см | R² = {pooled['r2']:.3f} | "
        f"замеров = {pooled['n']} ({len(per_gauge)} гидропостов)"
    )
    print("\n=== Физический режим переправы Якутск — Нижний Бестях (последние 6 сезонов) ===")
    print(yak_phys.tail(6).to_string(index=False))
    print(f"\nСохранено: {REPORTS / 'yakutsk_ice_physics_2001_2024.csv'}")


# Экспорт совместимых функций ядра
__all__ = [
    "compute_degree_days",
    "stefan_ice_curve",
    "calibrate_stefan_alpha",
    "summarize_yakutsk_physics",
    "stefan_thickness",
    "required_thickness_cm",
    "stefan_pairs",
    "fit_alpha",
]


if __name__ == "__main__":
    main()
