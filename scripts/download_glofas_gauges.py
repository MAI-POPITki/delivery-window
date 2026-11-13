"""Загрузка актуальных и исторических данных расхода воды по гидропостам
из Open-Meteo Global Flood API (Copernicus GloFAS v4) с пакетной выгрузкой.

Использование:
    .venv/bin/python scripts/download_glofas_gauges.py
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAUGES_CSV = ROOT / "data" / "processed" / "gauges.csv"
OUT_DAILY = ROOT / "data" / "external" / "glofas_gauge_discharge.csv"
OUT_SUMMARY = ROOT / "data" / "external" / "glofas_gauges_summary.csv"
OUT_REPORT = ROOT / "reports" / "glofas_gauges_summary.md"

API_BASE = "https://flood-api.open-meteo.com/v1/flood"
START_DATE = "2015-01-01"
ISSUE_DATE = date(2026, 11, 12)
FORECAST_DAYS = 14
END_DATE = (ISSUE_DATE + timedelta(days=FORECAST_DAYS)).isoformat()
BATCH_SIZE = 8


def fetch_batch(lats: list[float], lons: list[float]) -> tuple[list[dict], str]:
    params = {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "daily": ",".join([
            "river_discharge",
            "river_discharge_mean",
            "river_discharge_median",
            "river_discharge_p25",
            "river_discharge_p75",
        ]),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "models": "seamless_v4",
    }
    url = f"{API_BASE}?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "delivery-window/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data if isinstance(data, list) else [data]), url
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 4:
                wait = 5 * (attempt + 1)
                print(f"[429] Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded retries")


def main() -> None:
    gauges = pd.read_csv(GAUGES_CSV)
    extra = pd.DataFrame([{
        "gauge_code": "yakutsk_crossing",
        "name": "Lena At Yakutsk Crossing",
        "lat": 62.10,
        "lon": 129.80,
        "drainage_km2": 897000,
        "region": "Respublika Sakha (Yakutiya)",
        "basin": "Lena",
        "yakutia": True,
        "source_type": "Open-Meteo GloFAS",
        "source": API_BASE,
    }])
    gauges = pd.concat([extra, gauges], ignore_index=True)

    daily_rows: list[dict] = []
    summary_rows: list[dict] = []

    for start in range(0, len(gauges), BATCH_SIZE):
        chunk = gauges.iloc[start : start + BATCH_SIZE]
        lats = [float(x) for x in chunk["lat"]]
        lons = [float(x) for x in chunk["lon"]]
        items, src_url = fetch_batch(lats, lons)

        for (_, row), item in zip(chunk.iterrows(), items):
            code = str(row["gauge_code"])
            pid = code if code == "yakutsk_crossing" else f"gauge_{code}"
            block = item.get("daily", {})
            times = block.get("time", [])
            q = block.get("river_discharge", [])
            q_mean = block.get("river_discharge_mean", [])
            q_med = block.get("river_discharge_median", [])
            q_p25 = block.get("river_discharge_p25", [])
            q_p75 = block.get("river_discharge_p75", [])

            grid_lat = round(float(item.get("latitude", row["lat"])), 4)
            grid_lon = round(float(item.get("longitude", row["lon"])), 4)

            gauge_records: list[dict] = []
            for i, d_str in enumerate(times):
                is_fc = 1 if d_str > ISSUE_DATE.isoformat() else 0
                rec = {
                    "point_id": pid,
                    "gauge_code": code,
                    "name": row["name"],
                    "region": row["region"],
                    "basin": row["basin"],
                    "lat": grid_lat,
                    "lon": grid_lon,
                    "date": d_str,
                    "is_forecast": is_fc,
                    "river_discharge_m3s": q[i] if i < len(q) else None,
                    "river_discharge_mean_m3s": q_mean[i] if i < len(q_mean) else None,
                    "river_discharge_median_m3s": q_med[i] if i < len(q_med) else None,
                    "river_discharge_p25_m3s": q_p25[i] if i < len(q_p25) else None,
                    "river_discharge_p75_m3s": q_p75[i] if i < len(q_p75) else None,
                    "source_url": src_url,
                }
                gauge_records.append(rec)

            daily_rows.extend(gauge_records)
            df_g = pd.DataFrame(gauge_records)
            hist_vals = df_g.loc[df_g["is_forecast"] == 0, "river_discharge_m3s"].dropna()
            fc_vals = df_g.loc[df_g["is_forecast"] == 1, "river_discharge_m3s"].dropna()
            latest_row = df_g.loc[(df_g["is_forecast"] == 0) & df_g["river_discharge_m3s"].notna()].tail(1)
            latest_date = latest_row["date"].iloc[0] if not latest_row.empty else ""
            latest_val = float(latest_row["river_discharge_m3s"].iloc[0]) if not latest_row.empty else None

            summary_rows.append({
                "point_id": pid,
                "name": row["name"],
                "region": row["region"],
                "basin": row["basin"],
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
                "obs_days": int(len(hist_vals)),
                "latest_date": latest_date,
                "latest_discharge_m3s": round(latest_val, 2) if latest_val is not None else None,
                "fc_14d_mean_m3s": round(float(fc_vals.mean()), 2) if len(fc_vals) else None,
                "hist_mean_m3s": round(float(hist_vals.mean()), 2) if len(hist_vals) else None,
                "hist_q90_m3s": round(float(hist_vals.quantile(0.90)), 2) if len(hist_vals) else None,
                "hist_max_m3s": round(float(hist_vals.max()), 2) if len(hist_vals) else None,
                "source_url": src_url,
            })
            print(f"{pid}: obs={len(hist_vals)}, latest={latest_date} ({latest_val} m3/s)")
        time.sleep(2.0)

    df_daily = pd.DataFrame(daily_rows)
    df_sum = pd.DataFrame(summary_rows)
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(OUT_DAILY, index=False, compression="gzip")
    df_sum.to_csv(OUT_SUMMARY, index=False)

    lines = [
        "# Актуальные данные по гидропостам (Open-Meteo Global Flood API / GloFAS v4)",
        "",
        f"- **Дата выгрузки:** `{ISSUE_DATE.isoformat()}`",
        f"- **Период ряда:** `{START_DATE}` … `{ISSUE_DATE.isoformat()}` + прогноз на `{FORECAST_DAYS}` суток (до `{END_DATE}`)",
        f"- **Охват:** `{len(df_sum)}` гидропостов (`{len(df_daily):,}` суточных записей)",
        f"- **Источник:** `https://flood-api.open-meteo.com/v1/flood` (`models=seamless_v4`)",
        "",
        "| point_id | Пост | Бассейн | Актуальная дата | Расход, м³/с | Прогноз 14 дн. (ср.) | Q90 (2015–2026) | Макс. (2015–2026) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df_sum.iterrows():
        lines.append(
            f"| `{r['point_id']}` | {r['name']} | {r['basin']} | {r['latest_date']} | "
            f"{r['latest_discharge_m3s']} | {r['fc_14d_mean_m3s']} | {r['hist_q90_m3s']} | {r['hist_max_m3s']} |"
        )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {OUT_DAILY} ({len(df_daily)} rows), {OUT_SUMMARY}, {OUT_REPORT}")


if __name__ == "__main__":
    main()
