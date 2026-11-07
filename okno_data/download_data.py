"""
Совместимая точка входа `okno_data/download_data.py` (по структуре REPO_ROADMAP.md).
Проксирует вызовы в основной загрузчик `scripts/download_data.py`, поддерживая флаги
`--hydroposts`, `--prices`, `--weather`, `--ice`, `--gauges`, `--recent`.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts" / "download_data.py"), run_name="__main__")
