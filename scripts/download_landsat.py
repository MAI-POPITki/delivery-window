"""Fetch recent Landsat 8/9 windows and apply the validated optical M2."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from okno import landsat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    if not 1 <= args.days <= 366:
        parser.error("--days must be in 1..366")
    result = landsat.update(days=args.days)
    failed = False
    for key, row in result.get("points", {}).items():
        print(key, row.get("date", "no image"), row.get("platform", ""), row.get("status"), row.get("valid_fraction"))
        failed |= row.get("status") == "source_unavailable"
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
