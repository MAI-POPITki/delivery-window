from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
WEATHER = DATA / "weather"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
RECENT = DATA / "weather_recent"  # оперативные данные Open-Meteo: последние дни + прогноз на 16 дней
WEATHER_FORECASTS = DATA / "weather_forecasts"  # архив прогнозов с датой решения и временем доступности
ROSSTAT_PRICES = DATA / "rosstat_prices"  # тарифы переправы, сезонная динамика цен и сводки МЧС/Минтранса РС(Я)
EXTERNAL = DATA / "external"  # внешние подтверждённые данные по VALIDATION_PLAN.md (раздел 3)
DOCS = ROOT / "docs"


