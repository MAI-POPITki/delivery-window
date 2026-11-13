"""Проверка реальных сохранённых ответов; отчёт не заявляет точность прогноза."""
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okno.flood import DATA, POINTS, ROOT, daily_frame, read_response, summarize
from okno.flood_routes import load_routes


def main():
    now = datetime.now(timezone.utc)
    lines = ["## Проверка сохранённых данных GloFAS", "",
             f"Проверено: {now.isoformat()}. Источник: [Open-Meteo / GloFAS](https://open-meteo.com/en/docs/flood-api).",
             "", "Все расходы ниже — реальные ответы внешней модели, не измерения гидропостов.", "",
             "| Точка | Полных лет | Средний расход архива, м³/с | Порог q80, м³/с | Полных сценариев | Выше порога за 7 суток |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    errors = []
    cases = []
    roads = []
    for pid, point in POINTS.items():
        try:
            hist = read_response(DATA / "history" / f"{pid}.json")
            live = read_response(DATA / "latest" / f"{pid}.json")
            frame = daily_frame(hist["payload"], point)
            if str(frame.index.min().date()) != "1991-01-01" or str(frame.index.max().date()) != "2020-12-31":
                raise ValueError("Не соответствует период архива")
            result = summarize(hist, live, point, now)
            routes, metadata = load_routes(pid)
            if not routes or not metadata.get('source_url','').startswith('https://www.openstreetmap.org/api/0.6/map?'):
                raise ValueError('Не подтверждены исходные геометрии OpenStreetMap')
            roads.append(f"| {point['name']} | {len(routes)} | {metadata['fetched_at']} |")
            lines.append(f"| {point['river']} — {point['name']} | {result['years']} | "
                         f"{frame.river_discharge.mean():.2f} | {result['threshold']:.2f} | "
                         f"{result['members']} | {result['fraction']:.1%} |")
            case_path = DATA / "cases" / "2024" / f"{pid}.json"
            if case_path.exists():
                case = daily_frame(read_response(case_path)["payload"], point).river_discharge
                if len(case) != 366 or case.isna().any():
                    raise ValueError("Неполный диагностический 2024 год")
                cases.append(f"| {point['river']} — {point['name']} | {case.max():.2f} | "
                             f"{case.idxmax().date()} | {'да' if case.max() > result['threshold'] else 'нет'} |")
            # Проверяем сохранённые выпуски, а не только последнюю копию.
            for path in (DATA / "runs").glob(f"*/{pid}.json"):
                envelope = read_response(path)
                daily_frame(envelope["payload"], point)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{pid}: {exc}")
    lines += ["", "Запрошен архив 1991–2020; поставщик вернул пропуски за 1991–1996. "
              "В расчёт вошли 24 полных года 1997–2020. Пропуски не заполнены выдуманными значениями.",
              "", "Проверены SHA-256, происхождение, единицы, координаты ячеек, порядок дат, "
              "неотрицательность, полнота годовых рядов и ансамбля, свежесть загрузки. "
              "Горизонт: семь полных UTC-дней начиная с завтрашнего дня.", "",
              "**Технические проверки данных: " + ("ПРОЙДЕНЫ." if not errors else "НЕ ПРОЙДЕНЫ.") + "**", "",
              "**Точность предупреждения о затоплении: НЕ ПОДТВЕРЖДЕНА.** "
              "Нет независимой многосезонной проверки сохранённых оперативных выпусков по замерам. "
              "Нулевая доля сценариев выше расходного порога не исключает заторного затопления.", "",
              "Привязка: выбраны речные ячейки рядом с городами, а не центры городов. "
              "Предварительно сравнивались соседние ячейки по расходу за 2020–2021. "
              "Это проверка правдоподобия; топологическая привязка к руслу и независимым гидропостам "
              "ещё требует верификации. Особенно важны регулирование Урала и заторы на Лене.", "",
              "Архив: `data/flood/history/`; последний ответ: `data/flood/latest/`; "
              "сохранённые выпуски по UTC-дню получения: `data/flood/runs/`. "
              "Каждый JSON содержит URL запроса, время получения, модель, исходный ответ и SHA-256.", "",
              "Воспроизведение: `python scripts/download_flood.py --history`, "
              "`python scripts/validate_flood.py`, `python -m unittest discover -s tests -p 'test_flood*.py'`.", "",
              "Метод и заранее заданные критерии: `docs/FLOOD_VALIDATION_PLAN.md`."]
    if cases:
        lines += ["", "### Диагностический сезон 2024", "",
                  "| Точка | Модельный максимум, м³/с | Дата UTC | Выше фиксированного порога |",
                  "| --- | ---: | --- | --- |", *cases, "",
                  "Это исторические значения seamless API, а не сохранённые выпуски до наводнения. "
                  "Таблица проверяет реакцию ряда на сезон; не доказывает заблаговременность или точность предупреждения. "
                  "Порог по 1997–2020 не подбирался под 2024 год. Официальные сообщения о последствиях: "
                  "[Оренбург, Курган, Тюмень](https://mchs.gov.ru/deyatelnost/press-centr/novosti/5280317), "
                  "[Якутия](https://mchs.gov.ru/uploads/document/2025-05-21/c2d8f5ede16d0aa263d22ef7dbc22833.pdf)."]
    if roads:
        lines += ['', '### Реальные дорожные геометрии OpenStreetMap', '',
                  '| Населённый пункт | Участков | Получено, UTC |', '| --- | ---: | --- |', *roads, '',
                  'Исходные XML сохранены с gzip-сжатием; проверены SHA-256 и разбор геометрий. '
                  'Источники и рамки запросов: `data/flood/routes/*.json`. '
                  'Это текущая геометрия, не исторический реестр ограничений движения.']
    lines += ["", *errors] if errors else []
    (ROOT / "reports" / "flood_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Flood validation: {len(POINTS) - len(errors)}/{len(POINTS)} points OK")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
