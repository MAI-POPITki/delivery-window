"""Retrieve prior weekly Meteo-DV observations, separately from M1 archives."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from okno import flood_observations as O, flood_refresh as R
from okno.flood_store import update_lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days',type=int,default=31)
    args=parser.parse_args()
    with update_lock(R.DIRECTORY/'locks/levels-lena_yakutsk.lock') as acquired:
        if not acquired:parser.error('Обновление уровней уже выполняется. Повторите после его завершения.')
        result=O.backfill_meteo(days=args.days)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if result['failed_windows'] else 0


if __name__=='__main__':raise SystemExit(main())
