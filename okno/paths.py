from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
WEATHER = DATA / "weather"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
RECENT = DATA / "weather_recent"  # оперативные данные Open-Meteo: последние дни + прогноз на 16 дней
