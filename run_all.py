"""Полный пересчёт: сборка данных → проверки → М1 → «до/после» → страница проекта.  python run_all.py"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["build_dataset.py", "validate_data.py", "train_m1.py", "before_after.py", "stress_test.py", "transfer_test.py",
         "forecast_journal.py", "build_site.py"]

for s in STEPS:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / s)], check=True)
print("\nВсё пересчитано. Отчёты — в reports/, дашборд: streamlit run app/dashboard.py")
