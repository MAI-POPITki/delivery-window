"""Download public GF-3 browse PNGs. These are not inputs for M2 inference."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from okno.gaofen import update
from okno.s1_points import LENA_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--points", nargs="+", choices=list(LENA_NAMES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    snapshot = update(directory=args.output, days=args.days, point_ids=args.points)
    selected = args.points or LENA_NAMES
    failed = False
    for key in selected:
        row = snapshot.get("points", {}).get(key, {})
        scene = row.get("scene", {})
        print(f"{key}: {row.get('status', 'not_checked')}; "
              f"full AOI: {row.get('covered_count', 0)}; image: {scene.get('acquired_date', 'none')}")
        failed |= row.get("status") != "ready"
    print("GF-3 browse PNG only; M2 predictions and model weights were not changed.")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
