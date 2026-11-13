# Проверки расширений, 13.11.2026

1. Целевые проверки низкой воды, водоёмов, HTTP-range, массы, рисков и погоды М3:
   **33 теста, OK**. Команда:
   `OKNO_FLOOD_AUTO_REFRESH=0 OKNO_HYDRO_AUTO_REFRESH=0 python -m unittest tests.test_waterbodies tests.test_http_range tests.test_low_water tests.test_truck_load tests.test_handling_wind tests.test_risk_advice -v`.
2. Полный прогон: **339 тестов, OK**, 125,4 с. Использован отдельный checkout
   с зафиксированными исходными данными ZXC и текущими изменениями кода,
   без фоновых сетевых обновлений:
   `OKNO_FLOOD_AUTO_REFRESH=0 OKNO_HYDRO_AUTO_REFRESH=0 python -m unittest discover -s tests`.
3. `python scripts/check_extension_candidates.py`: обе спутниковые маски повторно
   рассчитаны из сохранённых DN, совпадение точное; выполнены три варианта MNDWI,
   три сценария массы и шесть сценариев погоды/смен с известным ожидаемым ответом.
   Численные результаты: `extension_candidates.json`.
4. Chromium/Playwright: существующий локальный Streamlit, `/?module=waterbodies`.
   Проверены экран снимков и вкладка качества. Ошибок JavaScript и исключений
   приложения нет. [Снимок экрана](waterbody_beta_preview.png).
5. `git diff --check`: OK. Рабочие модели М1/М2 и существующий паводковый загрузчик
   не менялись. Новые источники и расчёты сохранены отдельно в `data/waterbodies/`.

Эти проверки не означают независимую точность береговой линии, верность любых
пользовательских ограничений техники, безопасность проезда или фактическую экономию.
Итоговое сравнение возможностей: [extension_candidates.md](extension_candidates.md).
