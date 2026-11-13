"""Refresh flood sources on demand or in a scheduled job; keeps last good data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from okno import flood_refresh as R
from okno.flood_sites import SITES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',nargs='+',choices=list(SITES),default=list(SITES))
    parser.add_argument('--sources',nargs='+',choices=list(R.INTERVALS),default=list(R.INTERVALS))
    parser.add_argument('--force',action='store_true',help='Проверить независимо от срока последнего опроса')
    args=parser.parse_args()
    failed=[]
    for source in args.sources:
        for pid in args.points:
            if source not in R.kinds(pid):continue
            state=R.refresh_one(source,pid,force=args.force)
            print(f'{source} {pid}: {state}',flush=True)
            if state=='error':failed.append((source,pid))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
