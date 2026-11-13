"""Download official measurements and regional bulletins without changing models.

    python scripts/download_official_hydro.py --bootstrap --history-pages 8
    python scripts/download_official_hydro.py
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okno.hydro_ingest import update
from okno.official_hydro import coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true", help="Also collect verified historical starting products")
    parser.add_argument("--history-pages", type=int, default=1, choices=range(1, 31))
    args = parser.parse_args()
    result = update(bootstrap=args.bootstrap, history_pages=args.history_pages)
    print(json.dumps(result, ensure_ascii=False))
    for row in coverage():
        print(f"{row['name']}: {row['measured_points']} точек с датированными измерениями; "
              f"{row['bulletins']} сводок; последний замер {row['last_observation'] or 'нет'}")
    if result["status"] == "unavailable":
        raise SystemExit("Источники недоступны; предыдущие данные сохранены")


if __name__ == "__main__":
    main()
