"""Download daily MODIS/VIIRS pictures for the project's seven satellite points."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okno import daily_imagery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, choices=range(1, 32), metavar="1..31")
    args = parser.parse_args()
    snapshot = daily_imagery.update(days=args.days)
    print("NASA GIBS:", snapshot.get("checked_at"))
    print("Pictures:", dict(Counter(r["status"] for r in snapshot.get("records", []))))
    print("Source errors:", len(snapshot.get("errors", [])))
    return 2 if snapshot.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
