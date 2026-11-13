"""Fetch real Landsat windows, train optical M2, validate on later seasons.

python scripts/train_m2_landsat.py --download
Requires requirements-data.txt only for reading the existing Qiu reference TIFFs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score

from okno import landsat as L, m2_landsat as M, m2
from okno.gaofen import _atomic_write
from okno.paths import REPORTS
from okno.report import md_table
from scripts.s1_chips import ice_dates

LOCAL = threading.local()


def client():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


def collect(per_month=1):
    catalog_file = L.DIRECTORY / "training_catalog.json"
    if catalog_file.exists():
        catalog = json.loads(catalog_file.read_text())
    else:
        catalog = {}
    tasks = [(key, year, point) for key, point in L.points().items() for year in range(2018, 2025)]

    def discover(task):
        key, year, point = task
        cache_key = f"{key}/{year}"
        if cache_key not in catalog:
            end = f"{year}-08-31" if year == 2024 else f"{year}-10-31"
            items = L.search(point, f"{year}-03-01", end, session=client())
        else:
            items = catalog[cache_key]
        return key, year, items

    selected = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for key, year, items in executor.map(discover, tasks):
            catalog[f"{key}/{year}"] = items
            for month in range(3, 11):
                group = sorted([r for r in items if int(r["date"][5:7]) == month],
                               key=lambda r: (r["scene_cloud_cover"], r["observed_at"]))
                dates = set()
                for item in group:
                    if item["date"] in dates:
                        continue
                    selected[(key, item["item_id"])] = item
                    dates.add(item["date"])
                    if len(dates) >= per_month:
                        break
            _atomic_write(catalog_file, (json.dumps(catalog, ensure_ascii=False)+"\n").encode())
    print(f"Selected {len(selected)} point/scene windows", flush=True)
    rows, errors = [], []

    def fetch(task):
        (key, _), item = task
        path = L.download(key, item, session=client())
        return L.index_row(key, item, path)

    existing = L.load_index()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch, task): task[0] for task in selected.items()}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({"point": futures[future][0], "item_id": futures[future][1], "error": type(exc).__name__})
            if i % 10 == 0 or i == len(futures):
                print(f"Downloaded {i}/{len(futures)}; valid files {len(rows)}, errors {len(errors)}", flush=True)
                L.save_index(pd.concat([existing, pd.DataFrame(rows)], ignore_index=True))
    _atomic_write(REPORTS / "landsat_download_errors.json", (json.dumps(errors, indent=2)+"\n").encode())
    return L.load_index()


def make_masks(index):
    masks, info = {}, {}
    seeds = m2.load_masks()
    for key in L.points():
        reference = index[(index.point == key) & index.date.str[:4].isin(["2018", "2019", "2020"])
                          & index.date.str[5:7].isin(["07", "08"])]
        stacks = [L.load_chip(L.chip_path(key, item)) for item in reference.item_id]
        mask = M.build_mask(stacks, seeds.get(key))
        info[key] = {"summer_reference_scenes": list(reference.item_id), "river_pixels": int(mask.sum()) if mask is not None else 0}
        if mask is not None:
            masks[key] = mask
    stream = io.BytesIO()
    np.savez_compressed(stream, **masks)
    _atomic_write(M.MASKS, stream.getvalue())
    return masks, info


def score(data):
    y, p = data.label.eq("ice"), data.p_ice
    return {"n": len(data), "ice": int(y.sum()), "water": int((~y).sum()),
            "accuracy": float(accuracy_score(y, p >= 0.5)), "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
            "f1_ice": float(f1_score(y, p >= 0.5, zero_division=0)),
            "f1_water": float(f1_score(~y, p < 0.5, zero_division=0)), "brier": float(brier_score_loss(y, p))}


def train(index):
    masks, mask_info = make_masks(index)
    dates = {key: ice_dates(p["lat"], p["lon"]) for key, p in L.points().items()}
    rows = []
    for r in index.to_dict("records"):
        feat, quality = M.features(L.load_chip(L.chip_path(r["point"], r["item_id"])), masks.get(r["point"]))
        label, hy = M.reference_label(r["date"], dates[r["point"]])
        rows.append({**r, **quality, "label": label, "hydro_year": hy, **(feat or {})})
    data = pd.DataFrame(rows)
    _atomic_write(L.DIRECTORY/"features.csv", data.to_csv(index=False, lineterminator="\n").encode())
    labeled = data[data.label.isin(["ice", "water"]) & data.quality.eq("usable")].copy()
    print("Labeled rows:", len(labeled), labeled.label.value_counts().to_dict(), "masks", {k: int(v.sum()) for k, v in masks.items()}, flush=True)
    if len(labeled) < 80 or labeled.label.value_counts().min() < 15 or labeled.label.nunique() != 2:
        raise ValueError("Insufficient independent phenology-labeled optical scenes")
    out = []
    for year in range(2021, 2025):
        tr, te = labeled[labeled.hydro_year < year], labeled[labeled.hydro_year == year]
        if tr.label.nunique() < 2 or te.label.nunique() < 2 or len(te) < 10:
            continue
        model = M.make_model().fit(tr[M.FEATURES], tr.label.eq("ice"))
        part = te[["point", "item_id", "date", "platform", "hydro_year", "label"]].copy()
        part["p_ice"] = model.predict_proba(te[M.FEATURES])[:, 1]
        part["trained_through_season"] = int(tr.hydro_year.max())
        out.append(part)
        print("Forward season", year, score(part), flush=True)
    if not out:
        raise ValueError("No eligible forward-season validation folds")
    evaluation = pd.concat(out, ignore_index=True)
    metrics = score(evaluation)
    metrics["seasons"] = int(evaluation.hydro_year.nunique())
    metrics["unique_scenes"] = int(evaluation.item_id.nunique())
    enabled = bool(metrics["n"] >= 40 and min(metrics["ice"], metrics["water"]) >= 10 and metrics["seasons"] >= 3
                   and metrics["balanced_accuracy"] >= 0.8 and min(metrics["f1_ice"], metrics["f1_water"]) >= 0.75)
    model = M.make_model().fit(labeled[M.FEATURES], labeled.label.eq("ice"))
    model.save_model(str(M.MODEL))
    now = datetime.now(timezone.utc).isoformat()
    meta = {"schema_version": 1, "enabled": enabled, "trained_at": now, "source": L.SOURCE,
            "model_sha256": hashlib.sha256(M.MODEL.read_bytes()).hexdigest(),
            "features": M.FEATURES, "training_scenes": len(labeled), "training_windows": len(labeled),
            "training_class_counts": labeled.label.value_counts().to_dict(),
            "trained_through_season": int(labeled.hydro_year.max()), "validation": metrics,
            "validation_protocol": "Forward chaining by hydrological season; masks from summers 2018-2020 only",
            "reference": "Qiu et al. 2026, doi:10.5281/zenodo.17054619; ±14-day transition exclusion",
            "mask_calibration_years": [2018, 2019, 2020], "masks": mask_info,
            "mask_sha256": hashlib.sha256(M.MASKS.read_bytes()).hexdigest(),
            "features_sha256": hashlib.sha256((L.DIRECTORY/"features.csv").read_bytes()).hexdigest()}
    _atomic_write(M.META, (json.dumps(meta, ensure_ascii=False, indent=2)+"\n").encode())
    _atomic_write(REPORTS/"m2_landsat_oof.csv", evaluation.to_csv(index=False, lineterminator="\n").encode())
    by_year = pd.DataFrame([{**score(g), "season": int(year)} for year, g in evaluation.groupby("hydro_year")])
    by_point = pd.DataFrame([{**score(g), "point": key} for key, g in evaluation.groupby("point")])
    by_platform = pd.DataFrame([{**score(g), "platform": key} for key, g in evaluation.groupby("platform")])
    report = ["# М2: оптическая ветка Landsat 8/9", "", f"Ветка включена: **{enabled}**. Обучающих вырезок: {len(labeled)}; "
              f"вода {meta['training_class_counts'].get('water', 0)}, лёд {meta['training_class_counts'].get('ice', 0)}.", "",
              "Исходные USGS Collection 2 Level-2, шесть каналов отражательной способности и QA через публичный API "
              "Microsoft Planetary Computer. Вырезки 6×6 км, около 30 м/пиксель; RGB используется только для просмотра.", "",
              "Разметка — спутниковые даты Qiu с собственной ошибкой около 11 суток, а не ручная проверка каждого "
              "кадра. Исключены ±14 суток от ледостава/вскрытия и неправдоподобные даты. Облачность, тени, циррус, "
              "заполнение и насыщение отбрасываются; снег сохраняется. На русле нужно ≥30 пригодных пикселей и ≥35% покрытия.", "",
              "Проверка последовательная по сезонам: каждый тестовый сезон строго позже всех обучающих. "
              "Дата, координаты и платформа не входят в признаки. Маски воды построены по лету 2018–2020; "
              "географический ориентир — существующая маска Sentinel-1. Метрики относятся к классификации "
              "вне обучения, не к точности дат закрытия переправ и не к экономии в планировщике.", "",
              f"Проверочные {metrics['n']} вырезок относятся к {metrics['unique_scenes']} уникальным сценам; "
              "все вырезки одной даты остаются в одном сезоне.", "",
              "## Общая проверка", "", md_table(pd.DataFrame([metrics]).round(4)), "",
              "## По сезонам", "", md_table(by_year.round(4)), "", "## По точкам", "", md_table(by_point.round(4)), "",
              "## По спутникам", "", md_table(by_platform.round(4)), "",
              "Ночные/полярные и закрытые облаками участки не получают уверенной оценки. "
              "Модель Sentinel-1 сохранена. Автоматическое применение проверяет время поступления данных, "
              "версию модели и свежесть; исторические снимки не считаются доступными проекту задним числом.", ""]
    _atomic_write(REPORTS/"m2_landsat_metrics.md", "\n".join(report).encode())
    M.predict_index()
    print("FINAL", json.dumps(metrics), "enabled", enabled, flush=True)
    return enabled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--per-month", type=int, default=1)
    args = parser.parse_args()
    index = collect(args.per_month) if args.download else L.load_index()
    return 0 if train(index) else 2


if __name__ == "__main__":
    raise SystemExit(main())
