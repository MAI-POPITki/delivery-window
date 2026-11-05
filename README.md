# Окно завоза

Прогноз того, когда откроется и закроется сезонный путь Якутии (зимник, ледовая переправа),
и график отправок для опорных складов северного завоза по этому прогнозу.
Только открытые данные, без датчиков. Кейс-чемпионат, кейс А3.

| | |
| --- | --- |
| 🌐 **Страница проекта** | https://gneh3.github.io/delivery-window/ — ключевые цифры, графики, документ |
| 📊 **Живой дашборд** | https://okno-zavoza.streamlit.app/ — карта, прогнозы, планировщик (если «спит» — первое открытие ~30–60 с) |
| 📄 **Документ проекта** | [docs/project.pdf](docs/project.pdf) |

**Изменение для А3:** график завоза по вероятностному прогнозу окна вместо календаря.
**Пилот:** переправа Якутск — Нижний Бестях и зимник на Батагай (Верхоянский арктический район).

## Как повторить расчёт

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt     # requirements-data.txt — если пересчитывать спутниковые слои
python run_all.py                 # сборка данных → проверки → М1 → «до/после»
streamlit run app/dashboard.py    # дашборд
```

Страница проекта для GitHub Pages собирается в `docs/` (`scripts/build_site.py`, входит в `run_all.py`);
документ проекта — `docs/project.pdf`.

Данные уже лежат в `data/`. Скачать заново: `python scripts/download_data.py`
(погода ERA5 из Open-Meteo ограничена по частоте — скрипт ждёт и продолжает сам).

## Структура

| Путь | Что там |
| --- | --- |
| `scripts/download_data.py` | загрузка: ERA5 (Open-Meteo), спутниковые даты льда Лены (Zenodo), замеры гидропостов (Arctic Data Center) |
| `scripts/build_dataset.py` | сырые файлы → единый набор `data/processed/` с `source_type` и ссылкой на источник |
| `scripts/validate_data.py` | проверки валидности → `reports/data_checks.md` |
| `scripts/train_m1.py` | М1: CatBoost-квантили, бэктест leave-one-season-out → `reports/m1_metrics.md`, `models/` |
| `scripts/before_after.py` | М3 на прошлых сезонах, «до/после», чувствительность ±30% → `reports/before_after.md` |
| `okno/` | код моделей: физика льда (`ice.py`, `calib.py`), признаки (`features.py`), М1 (`m1.py`), М3 (`planner.py`), прогноз на дату (`forecast.py`) |
| `app/dashboard.py` | дашборд: карта, карточка пути, планировщик, рейтинг риска |
| `reports/` | все результаты, которые идут в паспорт и презентацию |

## Модели

- **М1 — прогноз окна.** Физика: закон Стефана `h = α·√FDD` (α подобран по реальным замерам толщины,
  см. `reports/stefan_calibration.csv`) и сумма оттепельных градусо-дней для весны.
  Поверх — CatBoost с квантильной функцией потерь (10/50/90%), предсказывает отклонение от календаря.
  Сравнение: календарь (средняя дата за прошлые 10 сезонов) и чистая физика.
  Проверка: сезон выкидывается целиком; MAE за 7/14/28 дней до события и покрытие диапазона 10–90%.
- **М3 — планировщик.** 1 000 сценариев даты закрытия из М1, минимизация ожидаемых расходов:
  перевозка + хранение при раннем завозе + дорогой канал для недовезённого.

## Словарь данных

Дни сезона: `0` = 1 сентября, гидрологический год `hydro_year = Y` — сезон с сентября Y−1 по август Y.

| Файл | Строка | Ключевые поля | Единицы | Тип источника |
| --- | --- | --- | --- | --- |
| `data/weather/*.csv`, `data/weather/gauges/*.csv` | точка × сутки | `temperature_2m_mean/min/max`, `snowfall_sum`, `precipitation_sum` | °C, см, мм | реанализ (ERA5) |
| `data/processed/gauges.csv` | гидропост | `gauge_code`, `lat`, `lon`, `region`, `yakutia` | ° | замер |
| `data/processed/hydro_ice_thickness.csv` | пост × дата замера | `ice_cm` (очищено), `ice_cm_raw`, `spike`, `frozen_to_bottom` | см | замер |
| `data/processed/hydro_ice_seasons.csv` | пост × сезон | даты `first_ice`, `freezeup`, `melt_start`, `drift_start`, `ice_end`; `open_{10,20,30}t_*` — лёд достиг нормативной толщины; `valid`, `flags` | дата / день сезона | замер |
| `data/processed/yakutsk_sat_seasons.csv` | сезон | `sat_freezeup_*`, `sat_breakup_*`, `sat_ice_duration_days` | дата / день сезона / сут | спутник |
| `data/processed/m1_dataset_*.csv` | точка × сезон × дата выпуска | признаки М1, `target_day`, `cal_target`, `lead_days` | — | производные |

Нормативная толщина для нагрузки P т — допущение `h = 11·√P` см (10 т → 35 см, 20 т → 49 см),
сверить с ОДН 218.010-98. Экономика М3 (объёмы, тарифы, хранение) — синтетика/допущения,
не участвует в оценке точности М1.

## Источники

- Shiklomanov A. et al. Data of river ice timing and thickness for selected river gauges in Russian
  pan-Arctic over 1970–2012. Arctic Data Center, doi:10.18739/A22Z12Q02.
- Qiu et al. 2026, ESSD — даты ледостава и вскрытия Лены по сетке 3 км, 2001–2024,
  doi:10.5281/zenodo.17054619 (CC BY 4.0).
- ERA5 (Hersbach et al. 2020) через Open-Meteo Historical Weather API.
