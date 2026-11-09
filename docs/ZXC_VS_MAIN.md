# Отличия ZXC от main

Сравнение с `main` на коммите `71b144a924202d90721fad5ca0b7f61605cf6e63`
(09.11.2026). Этот main уже объединён с ZXC. Ниже описано итоговое поведение,
а не все промежуточные эксперименты истории ветки.

[Ссылка сравнения на GitHub](https://github.com/MAI-POPITki/delivery-window/compare/main...ZXC)
станет актуальной для этой версии после публикации ZXC. Изменения сохранены
локально; автоматическая проверка отклонила push до явного разрешения автора.

## Что увидит пользователь

| Область | В main | В ZXC |
| --- | --- | --- |
| Навигация | Семь вкладок: карта, карточка, коридор, планировщик, рейтинг, проверки, данные | Два направления: «Зимники и переправы» и «Паводки». В зимниках пять вкладок; анализ поездки — в «Рисках и рекомендациях» |
| Карта зимников | Прогноз, спутниковые данные, притоки Лены | Дополнительные слои выключены при первом открытии; подписи и легенда уточнены. Карта не перегружена формами создания маршрута |
| Проверка поездки | Отдельные планировщик и рейтинг риска | Выбор дороги из опубликованного реестра, дата проезда, масса машины и груза; ограничения и прогноз рассматриваются вместе |
| Рекомендации | Общие результаты планировщика | Проверка более раннего проезда, превышения полной массы и необходимости более лёгкой машины; все варианты используют один выпуск прогноза и опубликованные на дату решения сведения |
| Исторический коридор А3 | Самостоятельная вкладка | Раскрываемый раздел в «Рисках и рекомендациях»; существующие расчёты, источники, графики и режим «сервис» сохранены |
| Паводки | Отдельного режима нет | Дата планирования, карта реальных дорог, прогноз воды, риски и рекомендации, оптимизатор доставки и данные с источниками |
| Стартовый экран | Описание прогноза и планирования | Описание доступных сейчас режимов и проверки поездки; исправлены упоминания скрытых вкладок |

## Новый прогноз высокой воды

Регионы: Якутия (Лена у Якутска), Тюменская область (Ишим у Ишима),
Курганская область (Тобол у Кургана), Оренбургская область (Урал у Оренбурга и Орска).

Источник — GloFAS v4 через Open-Meteo, версия явно зафиксирована параметром
`models=seamless_v4`. Сохранены реальные JSON-ответы: исторический запрос
1991–2020 (полны 1997–2020), диагностический 2024 год, последний ансамбль и
первые полученные за каждый UTC-день выпуски. Есть URL, время получения и SHA-256.

Индикатор — доля полных сценариев, в которых расход превысит q80 годовых
максимумов хотя бы раз за следующие семь полных UTC-суток. Это исследовательский
порог, не официальная опасная отметка. Минимум — 20 полных исторических лет и
30 полных сценариев. Пропуски не заполняются; старые, повреждённые или неполные
расчёты дают серую точку. Внешняя сеть не нужна для открытия сохранённого прогноза.

Улучшения отображения: обычные даты и разряды чисел, пояснения показателей,
выбор реки, выделение выбранной точки, автоматическое приближение к региону,
контрастные подписи, понятные названия графика и столбцов CSV. Исторический
порог включается на графике отдельно, чтобы не сжимать текущие значения.
Полные технические отчёты доступны для скачивания, а не занимают основной экран.

**Точность предупреждений о затоплении не доказана.** Исторический ряд 2024 года
превысил фиксированный порог у Ишима, Кургана и Оренбурга, но не у Орска,
несмотря на известное наводнение. Нет расчёта опасных уровней, ледовых заторов
и границ затопления. Ограничение явно указано в интерфейсе.

### Рабочая область паводков после замечания пользователя

Вместо отдельного просмотра прогноза добавлены пять разделов: «Карта путей»,
«Прогноз», «Риски», «План», «Данные». Компактные названия помещаются в узком окне;
внутри разделов показаны полные названия и пояснения.

В боковой панели доступны дата планирования, регион, речная точка, конкретный
дорожный участок и длительность проезда. Для прошлой даты выбирается только
выпуск, фактически полученный не позже конца этих UTC-суток. Исторический
реанализ не подменяет отсутствующий оперативный прогноз.

Из официального API OSM сохранён 421 дорожный и мостовой участок: Якутск — 95,
Ишим — 14, Курган — 211, Оренбург — 90, Орск — 11. Сохранены XML.gz, запросы,
время получения и SHA-256. Путь выделяется синим; речные точки используют общую
с зимниками шкалу `RdYlGn`: красный → жёлтый → зелёный. Показатель точки — доля
сценариев без высокой воды, а не подтверждённая проходимость дороги.

Раздел рисков сравнивает даты с учётом всей длительности рейса. Планировщик
оптимизирует распределение объёма между подтверждёнными днями по стоимости
перевозки, хранения и оценке довоза остатка. Мощности, тарифы и подтверждения
вводит оператор; учебных чисел нет. Источник и дата подтверждения обязательны,
поздние сведения исключаются. Соперник — равномерный план с теми же ограничениями.
Гидрологический фильтр — настраиваемая политика предосторожности, а не модель
закрытия дороги. Ввод сохраняется между вкладками; результаты выгружаются в CSV.

В разделе данных видны источник и возраст выпуска, число сценариев, история,
метод порога, источники геометрий и ограничения. Текущие геометрии не выдаются
за архив состояния дороги. Проверки расширения: **27 тестов**, включая
независимый перебор 27 маленьких задач оптимизации и расчёт через интерфейс.

После замечания о неизменном прогнозе разделены «Данные, известные на дату»
и «Дата оценки / проезда». Карта, суточный расход, показатель за рейс и начало
отправок привязаны ко второй дате. На графике выделены выбранные сутки,
под ним добавлена таблица. Для выбранного пользователем 20.03.2026 явно показано,
что в текущем архиве есть только ответы, полученные 09.11.2026; есть переход
к доступному выпуску. При переходе сбрасывается устаревшее значение календаря.
Порог не менялся ради искусственного изменения процентов. Пройден **31 профильный тест**.
[Подробная методика, источники и разбор примера](FLOOD_METHOD.md).

## Проверка сезонного пути

- Учитываются только опубликованные на дату решения записи той же дороги и сезона.
- Наличие гидропоста не означает разрешённый зимник; привязка к дороге задаётся явно.
- Полная масса включает пустую машину с оборудованием и груз. Ограничение
  массы не подменяется расчётной толщиной льда.
- Неизвестный предел и отсутствие записей не считаются разрешением проезда.
- Модельный риск физического закрытия применяется в пределах порога 20 т;
  для более тяжёлых машин процент не переносится.
- Более ранняя дата предлагается только при меньшем риске и известном допустимом
  пределе массы; при закрытии дороги уменьшение груза не снимает запрет.
- Сравнение дат выгружается в CSV вместе с источниками.

## Расчётные модули для команды

В ветке также сохранены предыдущие разработки, которые сейчас не представлены
как отдельные пользовательские экраны:

1. `supply_scenarios`: заданные сезонные окна, перерывы, дни обслуживания и
   календари доступности для существующего симулятора; календарь должен быть
   известен на старте. Добавлены проверки баланса, лагов и склада.
2. `backhaul` / `backhaul_io`: дискретное планирование совместного завоза и
   обратной загрузки, парк, совместимость кузовов, сроки, возврат машин,
   расходы и импорт/экспорт. Примеры в отчётах искусственные и помечены как учебные.
3. `routing` / `journey`: поиск населённых пунктов, геометрия дорожных подъездов
   и сезонных участков, согласование с реестром. Формы построения маршрута
   убраны из текущего интерфейса; модули и проверки сохранены для команды.
4. `truck_load`: проверка полной массы по датированному реестру.
5. `decision_support` / `bottlenecks`: сравнение даты начала отправок,
   чувствительность расходов и диагностическое изменение мощностей коридора.
6. В `planner`, `economic_planner` и `corridor` расширена обработка заданной
   доступности/диагностических сценариев. Эти изменения не подтверждают реальную
   доступность запасных маршрутов, техники или обратного груза.

## Дополнительные данные и документы

Добавлены геометрии сезонных участков проекта, справочник населённых пунктов
OpenStreetMap и описание источника, погодный файл `gauge_70378.csv`,
исследовательский отчёт о покрытии Коми, отчёты дорожной проверки и обратной загрузки.
Коми остаётся исследовательским случаем, а не включённым регионом сервиса.

Документы описывают переносимые сценарии, обратную загрузку, проверку массы,
риски поездки, ограничения дорожной геометрии и протокол нового прогноза высокой воды.
История решений и отрицательных результатов дополнена в `docs/DEVLOG.md`.

## Исправления и производительность

- Архивные пороги кэшируются по пути, времени изменения, размеру файла и году
  отсечения; свежесть оперативного прогноза проверяется при каждом открытии.
  Изменение файла или удаление данных не сохраняет прежний положительный статус.
- Локальный замер пяти точек: медиана пяти запусков 153,0 мс со сбросом кэша
  и 48,5 мс с прогретым кэшем (примерно 3,15 раза). Это время расчёта данных,
  без загрузки страницы, картографических плиток и сети.
- Исправлены некорректные координаты `NaN` и даты с временем/пропусками.
- Исправлено обновление модулей паводков в уже работающем Streamlit.
- Добавлены прямая ссылка `?module=flood` и возврат к описанию проекта.
- Устранена двойная установка начального значения выбора подхода в коридоре.
- Два устаревших UI-теста переведены на актуальную навигацию; математические
  проверки исторического коридора продолжают выполняться.

## Что уже пришло из main

Наблюдения PANGAEA, снег ERA5-Land, модели обновлённой М1 и М1 со снегом,
расширение новыми постами Магаданской области и Чукотки, уточнение источников,
свежие данные и `docs/TUTORIAL.pdf` сохранены при слиянии. Они не являются
дополнительными возможностями ZXC относительно указанного main.

Проверки: [протокол паводков](FLOOD_VALIDATION_PLAN.md),
[исходные числа](../reports/flood_validation.md),
[проверки программы](../reports/flood_implementation_checks.md).
Полный прогон после добавления рабочей области паводков: **172/172 теста пройдены**;
валидация гидрологии и дорожных геометрий — **5/5 точек**.

## Полный перечень отличающихся файлов

Перечень ниже сформирован из Git; A — добавлен, M — изменён.

<!-- FILE_INVENTORY -->

Всего файлов: **100**.

| Статус | Файл |
| --- | --- |
| M | [.github/workflows/daily-update.yml](../.github/workflows/daily-update.yml) |
| M | [README.md](../README.md) |
| A | [app/backhaul_view.py](../app/backhaul_view.py) |
| M | [app/corridor_view.py](../app/corridor_view.py) |
| M | [app/dashboard.py](../app/dashboard.py) |
| A | [app/decision_view.py](../app/decision_view.py) |
| A | [app/flood_view.py](../app/flood_view.py) |
| A | [app/flood_workspace.py](../app/flood_workspace.py) |
| A | [app/journey_view.py](../app/journey_view.py) |
| M | [app/planner_view.py](../app/planner_view.py) |
| A | [app/truck_load_view.py](../app/truck_load_view.py) |
| M | [app/welcome.py](../app/welcome.py) |
| A | [data/flood/README.md](../data/flood/README.md) |
| A | [data/flood/cases/2024/ishim_ishim.json](../data/flood/cases/2024/ishim_ishim.json) |
| A | [data/flood/cases/2024/lena_yakutsk.json](../data/flood/cases/2024/lena_yakutsk.json) |
| A | [data/flood/cases/2024/tobol_kurgan.json](../data/flood/cases/2024/tobol_kurgan.json) |
| A | [data/flood/cases/2024/ural_orenburg.json](../data/flood/cases/2024/ural_orenburg.json) |
| A | [data/flood/cases/2024/ural_orsk.json](../data/flood/cases/2024/ural_orsk.json) |
| A | [data/flood/history/ishim_ishim.json](../data/flood/history/ishim_ishim.json) |
| A | [data/flood/history/lena_yakutsk.json](../data/flood/history/lena_yakutsk.json) |
| A | [data/flood/history/tobol_kurgan.json](../data/flood/history/tobol_kurgan.json) |
| A | [data/flood/history/ural_orenburg.json](../data/flood/history/ural_orenburg.json) |
| A | [data/flood/history/ural_orsk.json](../data/flood/history/ural_orsk.json) |
| A | [data/flood/latest/ishim_ishim.json](../data/flood/latest/ishim_ishim.json) |
| A | [data/flood/latest/lena_yakutsk.json](../data/flood/latest/lena_yakutsk.json) |
| A | [data/flood/latest/tobol_kurgan.json](../data/flood/latest/tobol_kurgan.json) |
| A | [data/flood/latest/ural_orenburg.json](../data/flood/latest/ural_orenburg.json) |
| A | [data/flood/latest/ural_orsk.json](../data/flood/latest/ural_orsk.json) |
| A | [data/flood/routes/ishim_ishim.json](../data/flood/routes/ishim_ishim.json) |
| A | [data/flood/routes/ishim_ishim.osm.gz](../data/flood/routes/ishim_ishim.osm.gz) |
| A | [data/flood/routes/lena_yakutsk.json](../data/flood/routes/lena_yakutsk.json) |
| A | [data/flood/routes/lena_yakutsk.osm.gz](../data/flood/routes/lena_yakutsk.osm.gz) |
| A | [data/flood/routes/tobol_kurgan.json](../data/flood/routes/tobol_kurgan.json) |
| A | [data/flood/routes/tobol_kurgan.osm.gz](../data/flood/routes/tobol_kurgan.osm.gz) |
| A | [data/flood/routes/ural_orenburg.json](../data/flood/routes/ural_orenburg.json) |
| A | [data/flood/routes/ural_orenburg.osm.gz](../data/flood/routes/ural_orenburg.osm.gz) |
| A | [data/flood/routes/ural_orsk.json](../data/flood/routes/ural_orsk.json) |
| A | [data/flood/routes/ural_orsk.osm.gz](../data/flood/routes/ural_orsk.osm.gz) |
| A | [data/flood/runs/2026-11-09/ishim_ishim.json](../data/flood/runs/2026-11-09/ishim_ishim.json) |
| A | [data/flood/runs/2026-11-09/lena_yakutsk.json](../data/flood/runs/2026-11-09/lena_yakutsk.json) |
| A | [data/flood/runs/2026-11-09/tobol_kurgan.json](../data/flood/runs/2026-11-09/tobol_kurgan.json) |
| A | [data/flood/runs/2026-11-09/ural_orenburg.json](../data/flood/runs/2026-11-09/ural_orenburg.json) |
| A | [data/flood/runs/2026-11-09/ural_orsk.json](../data/flood/runs/2026-11-09/ural_orsk.json) |
| A | [data/osm/project_segments.json](../data/osm/project_segments.json) |
| A | [data/osm/settlements_ru.csv](../data/osm/settlements_ru.csv) |
| A | [data/osm/settlements_ru_SOURCE.md](../data/osm/settlements_ru_SOURCE.md) |
| M | [data/weather/gauges/gauge_70378.csv](../data/weather/gauges/gauge_70378.csv) |
| A | [docs/BACKHAUL.md](../docs/BACKHAUL.md) |
| A | [docs/BACKHAUL_VALIDATION.md](../docs/BACKHAUL_VALIDATION.md) |
| A | [docs/DECISION_SUPPORT_ZXC.md](../docs/DECISION_SUPPORT_ZXC.md) |
| M | [docs/DEVLOG.md](../docs/DEVLOG.md) |
| A | [docs/EXPANSION_ZXC.md](../docs/EXPANSION_ZXC.md) |
| A | [docs/FLOOD_METHOD.md](../docs/FLOOD_METHOD.md) |
| A | [docs/FLOOD_VALIDATION_PLAN.md](../docs/FLOOD_VALIDATION_PLAN.md) |
| A | [docs/JOURNEY_WORKSPACE.md](../docs/JOURNEY_WORKSPACE.md) |
| A | [docs/ROUTING_ZXC.md](../docs/ROUTING_ZXC.md) |
| A | [docs/ZXC_VS_MAIN.md](../docs/ZXC_VS_MAIN.md) |
| A | [okno/backhaul.py](../okno/backhaul.py) |
| A | [okno/backhaul_io.py](../okno/backhaul_io.py) |
| A | [okno/bottlenecks.py](../okno/bottlenecks.py) |
| M | [okno/corridor.py](../okno/corridor.py) |
| A | [okno/decision_support.py](../okno/decision_support.py) |
| M | [okno/economic_planner.py](../okno/economic_planner.py) |
| A | [okno/flood.py](../okno/flood.py) |
| A | [okno/flood_planning.py](../okno/flood_planning.py) |
| A | [okno/flood_routes.py](../okno/flood_routes.py) |
| A | [okno/journey.py](../okno/journey.py) |
| A | [okno/map_style.py](../okno/map_style.py) |
| M | [okno/planner.py](../okno/planner.py) |
| A | [okno/risk_advice.py](../okno/risk_advice.py) |
| A | [okno/routing.py](../okno/routing.py) |
| A | [okno/supply_scenarios.py](../okno/supply_scenarios.py) |
| A | [okno/truck_load.py](../okno/truck_load.py) |
| A | [reports/backhaul_validation.csv](../reports/backhaul_validation.csv) |
| A | [reports/backhaul_validation.md](../reports/backhaul_validation.md) |
| A | [reports/flood_implementation_checks.md](../reports/flood_implementation_checks.md) |
| A | [reports/flood_preview.jpg](../reports/flood_preview.jpg) |
| A | [reports/flood_validation.md](../reports/flood_validation.md) |
| A | [reports/komi_coverage.csv](../reports/komi_coverage.csv) |
| A | [reports/komi_coverage.md](../reports/komi_coverage.md) |
| A | [reports/routing_smoke.json](../reports/routing_smoke.json) |
| A | [scripts/check_komi_coverage.py](../scripts/check_komi_coverage.py) |
| A | [scripts/download_flood.py](../scripts/download_flood.py) |
| A | [scripts/download_flood_routes.py](../scripts/download_flood_routes.py) |
| A | [scripts/validate_flood.py](../scripts/validate_flood.py) |
| A | [scripts/verify_backhaul.py](../scripts/verify_backhaul.py) |
| A | [tests/test_backhaul.py](../tests/test_backhaul.py) |
| A | [tests/test_backhaul_ui.py](../tests/test_backhaul_ui.py) |
| M | [tests/test_corridor_a3.py](../tests/test_corridor_a3.py) |
| A | [tests/test_decision_support.py](../tests/test_decision_support.py) |
| A | [tests/test_flood.py](../tests/test_flood.py) |
| A | [tests/test_flood_workspace.py](../tests/test_flood_workspace.py) |
| M | [tests/test_import_reload.py](../tests/test_import_reload.py) |
| A | [tests/test_journey.py](../tests/test_journey.py) |
| A | [tests/test_risk_advice.py](../tests/test_risk_advice.py) |
| A | [tests/test_routing.py](../tests/test_routing.py) |
| A | [tests/test_supply_scenarios.py](../tests/test_supply_scenarios.py) |
| A | [tests/test_truck_load.py](../tests/test_truck_load.py) |
| M | [tests/test_welcome.py](../tests/test_welcome.py) |
| M | [tests/test_yakutsk_replay_ui.py](../tests/test_yakutsk_replay_ui.py) |
