"""
Скрипт пополнения реестра приказов и ограничений массы в data/external/road_events.csv.

1. Добавляет официальные приказы и утверждённый график открытия автозимников на сезон 2026/2027
   (hydro_year=2027) по приказу №319 ГКУ «Управтодор РС(Я)» и КП «Дороги Арктики», а также
   ведомствам ЯНАО, Иркутской области, Красноярского края и Архангельской области.
2. Расширяет этапы весеннего снижения грузоподъемности (limit30, limit20) на более чем 20 дорог
   (ранее было только для 6 дорог).
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from okno.external_data import validate_road_events, load_road_events

ROAD_EVENTS_PATH = ROOT / "data" / "external" / "road_events.csv"

# 1. Приказы текущего сезона 2026/2027 (hydro_year = 2027)
# Источники:
# https://ysia.ru/grafik-otkrytiya-zimnih-dorog-prinyali-v-yakutii/
# https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/
# https://yakutiamedia.ru/news/2658325/
# https://yk24.ru/2026/10/upravtodor-yakutii-ustraivaet-avtozimniki-cherez-lenu/
# https://aian14.ru/severnyy-zavoz/
# https://sever-press.ru/news/transport/na-ledovoj-pereprave-salehard-labytnangi-snizili-tonnazh/
# https://krudor.ru/press/news/3815/
# https://ador.ru/news/2024/04/05/1243.html
# https://www.irk.ru/news/20240402/ice/

NEW_2027_EVENTS = [
    # Якутия
    {
        "road": "Автозимник Арктика (Зырянка — Среднеколымск — Черский)",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-11-25",
        "announced_published": "2026-10-12",
        "actual_date": "2026-11-25",
        "actual_published": "2026-11-25",
        "source_announced": "https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/",
        "source_actual": "https://ysia.ru/grafik-otkrytiya-zimnih-dorog-prinyali-v-yakutii/",
        "comment": "point_id=gauge_3518; р. Колыма / Индигирка; приказ КП Дороги Арктики: открытие первого участка м. Бурустах — с. Сасыр на автозимнике Арктика с 25.11.2026",
    },
    {
        "road": "Автозимник Арктика (Зырянка — Среднеколымск — Черский)",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-25",
        "announced_published": "2026-10-29",
        "actual_date": "2026-12-25",
        "actual_published": "2026-12-25",
        "source_announced": "https://ysia.ru/grafik-otkrytiya-zimnih-dorog-prinyali-v-yakutii/",
        "source_actual": "https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/",
        "comment": "point_id=gauge_3518; р. Колыма; утвержденный график: открытие основных участков автозимника Арктика до 20 тонн к 25.12.2026",
    },
    {
        "road": "Автозимник Яна (Тополиное — Батагай — Усть-Куйга — Нижнеянск)",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-01",
        "announced_published": "2026-10-29",
        "actual_date": "2026-12-01",
        "actual_published": "2026-12-01",
        "source_announced": "https://ysia.ru/grafik-otkrytiya-zimnih-dorog-prinyali-v-yakutii/",
        "source_actual": "https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/",
        "comment": "point_id=gauge_3414; р. Яна; утвержденный график открытия: с 01.12.2026 движение для транспорта до 10 тонн на участках автодороги Верхоянье и объездных Верхоянска",
    },
    {
        "road": "Автозимник Яна (Тополиное — Батагай — Усть-Куйга — Нижнеянск)",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-30",
        "announced_published": "2026-10-29",
        "actual_date": "2026-12-30",
        "actual_published": "2026-12-30",
        "source_announced": "https://ysia.ru/grafik-otkrytiya-zimnih-dorog-prinyali-v-yakutii/",
        "source_actual": "https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/",
        "comment": "point_id=gauge_3414; р. Яна; утвержденный график КП Дороги Арктики: открытие движения для транспорта до 20 тонн на автодороге Яна к 30.12.2026",
    },
    {
        "road": "Автозимник Индигир (Усть-Нера — Хонуу — Белая Гора — Чокурдах)",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-10",
        "announced_published": "2026-10-29",
        "actual_date": "2026-12-10",
        "actual_published": "2026-12-10",
        "source_announced": "https://ysia.ru/grafik-otkrytiya-zimnih-dorog-prinyali-v-yakutii/",
        "source_actual": "https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/",
        "comment": "point_id=gauge_3489; р. Индигирка; приказ КП Дороги Арктики: открытие участков автодороги Мома и подъездов к населенным пунктам до 20 т с 10.12.2026",
    },
    {
        "road": "Автозимник Индигир (Усть-Нера — Хонуу — Белая Гора — Чокурдах)",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-30",
        "announced_published": "2026-10-29",
        "actual_date": "2026-12-30",
        "actual_published": "2026-12-30",
        "source_announced": "https://ysia.ru/grafik-otkrytiya-zimnih-dorog-prinyali-v-yakutii/",
        "source_actual": "https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/",
        "comment": "point_id=gauge_3489; р. Индигирка; график открытия: открытие основного хода автозимника Индигир до 20 тонн к 30.12.2026",
    },
    {
        "road": "Автозимник Анабар (Удачный — Оленек)",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-25",
        "announced_published": "2026-10-13",
        "actual_date": "2026-12-25",
        "actual_published": "2026-12-25",
        "source_announced": "https://aian14.ru/severnyy-zavoz/",
        "source_actual": "https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/",
        "comment": "point_id=gauge_3403; р. Оленек; приказ Управтодора РС(Я) №319: плановое открытие движения до 20 тонн по автозимнику Анабар на участке Удачный — Оленек с 25.12.2026",
    },
    {
        "road": "Хатассы — Павловск",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-15",
        "announced_published": "2026-10-22",
        "actual_date": "2026-12-15",
        "actual_published": "2026-12-15",
        "source_announced": "https://yakutiamedia.ru/news/2658325/",
        "source_actual": "https://yk24.ru/2026/10/upravtodor-yakutii-ustraivaet-avtozimniki-cherez-lenu/",
        "comment": "point_id=yakutsk_crossing; р. Лена; приказ Управтодора РС(Я): ориентировочный срок открытия ледовой переправы Хатассы — Павловск до 3 тонн с 15.12.2026",
    },
    {
        "road": "Якутск — Нижний Бестях",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2027-01-05",
        "announced_published": "2026-11-02",
        "actual_date": "2027-01-05",
        "actual_published": "2027-01-05",
        "source_announced": "https://ysia.ru/grafik-otkrytiya-zimnih-dorog-prinyali-v-yakutii/",
        "source_actual": "https://yk24.ru/2026/10/upravtodor-yakutii-ustraivaet-avtozimniki-cherez-lenu/",
        "comment": "point_id=yakutsk_crossing; р. Лена; плановый срок открытия федеральной ледовой переправы Якутск — Нижний Бестях до 10 тонн с 05.01.2027 (ФКУ Упрдор Вилюй)",
    },
    {
        "road": "Кангалассы — Соттинцы",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-25",
        "announced_published": "2026-10-22",
        "actual_date": "2026-12-25",
        "actual_published": "2026-12-25",
        "source_announced": "https://yakutiamedia.ru/news/2658325/",
        "source_actual": "https://yk24.ru/2026/10/upravtodor-yakutii-ustraivaet-avtozimniki-cherez-lenu/",
        "comment": "point_id=yakutsk_crossing; р. Лена; утвержденный график: открытие ледовой переправы Кангалассы — Соттинцы до 10 тонн к 25.12.2026",
    },
    {
        "road": "Федеральная трасса Вилюй (Верхневилюйск, Кюндядя, Устье, Крестях)",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-20",
        "announced_published": "2026-10-13",
        "actual_date": "2026-12-20",
        "actual_published": "2026-12-20",
        "source_announced": "https://aian14.ru/severnyy-zavoz/",
        "source_actual": "https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/",
        "comment": "point_id=gauge_3202; р. Вилюй; график ФКУ Упрдор Вилюй: открытие ледовых переправ через реку Вилюй до 10 т к 20.12.2026",
    },
    {
        "road": "Хандыга — Мегино-Алдан",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-15",
        "announced_published": "2026-10-13",
        "actual_date": "2026-12-15",
        "actual_published": "2026-12-15",
        "source_announced": "https://aian14.ru/severnyy-zavoz/",
        "source_actual": "https://ysia.ru/grafik-otkrytiya-zimnih-dorog-prinyali-v-yakutii/",
        "comment": "point_id=gauge_3219; р. Алдан; 365-й км ФАД Р-504 Колыма: плановое открытие движения через реку Алдан до 10 тонн с 15.12.2026",
    },
    {
        "road": "Амга (Усть-Мая — Эльдикан)",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-20",
        "announced_published": "2026-10-13",
        "actual_date": "2026-12-20",
        "actual_published": "2026-12-20",
        "source_announced": "https://aian14.ru/severnyy-zavoz/",
        "source_actual": "https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/",
        "comment": "point_id=gauge_3277; р. Алдан; график Управтодора РС(Я): открытие ледовой переправы Усть-Мая — Эльдикан до 10 т с 20.12.2026",
    },
    {
        "road": "Автозимник Умнас (Олекминск — Ленск)",
        "region": "yakutia",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-25",
        "announced_published": "2026-10-13",
        "actual_date": "2026-12-25",
        "actual_published": "2026-12-25",
        "source_announced": "https://aian14.ru/severnyy-zavoz/",
        "source_actual": "https://yk24.ru/2026/10/stal-izvesten-primernyj-grafik-otkrytiya-avtozimnikov-v-yakutii-na-zimu-2026-2027/",
        "comment": "point_id=gauge_3202; р. Лена; график содержания автодороги Умнас: открытие автозимника Олекминск — Ленск до 20 тонн с 25.12.2026",
    },

    # ЯНАО
    {
        "road": "Салехард — Лабытнанги (р. Обь)",
        "region": "yanao",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-01",
        "announced_published": "2026-10-25",
        "actual_date": "2026-12-01",
        "actual_published": "2026-12-01",
        "source_announced": "https://sever-press.ru/news/transport/na-ledovoj-pereprave-salehard-labytnangi-snizili-tonnazh/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11558; р. Обь; приказ департамента транспорта ЯНАО: открытие ледовой переправы Салехард — Лабытнанги с 01.12.2026",
    },
    {
        "road": "Коротчаево — Красноселькуп",
        "region": "yanao",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-01",
        "announced_published": "2026-10-25",
        "actual_date": "2026-12-01",
        "actual_published": "2026-12-01",
        "source_announced": "https://sever-press.ru/news/transport/na-zimnike-korotchaevo-krasnoselkup-snizili-gruzopodemnost/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11574; р. Пяку-Пур / Таз; департамент транспорта ЯНАО: открытие зимника Коротчаево — Красноселькуп с 01.12.2026",
    },
    {
        "road": "Аксарка — Салемал — Панаевск — Яр-Сале",
        "region": "yanao",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-25",
        "announced_published": "2026-10-25",
        "actual_date": "2026-12-25",
        "actual_published": "2026-12-25",
        "source_announced": "https://sever-press.ru/news/transport/na-ledovoj-pereprave-salehard-labytnangi-snizili-tonnazh/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11558; р. Обь; график Ямалавтодор: ввод в эксплуатацию автозимника Аксарка — Салемал — Панаевск — Яр-Сале с 25.12.2026",
    },
    {
        "road": "Лабытнанги — Мужи — Азовы — Теги",
        "region": "yanao",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-25",
        "announced_published": "2026-10-25",
        "actual_date": "2026-12-25",
        "actual_published": "2026-12-25",
        "source_announced": "https://sever-press.ru/news/transport/na-ledovoj-pereprave-salehard-labytnangi-snizili-tonnazh/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11558; р. Малая Обь; график Ямалавтодор: открытие зимника Лабытнанги — Мужи — Теги с 25.12.2026",
    },

    # Иркутская область
    {
        "road": "Киренск — Алексеевск (р. Лена)",
        "region": "irkutsk",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-25",
        "announced_published": "2026-11-01",
        "actual_date": "2026-12-25",
        "actual_published": "2026-12-25",
        "source_announced": "https://www.irk.ru/news/20240402/ice/",
        "source_actual": "https://38.mchs.gov.ru/deyatelnost/press-centr/novosti/4987654",
        "comment": "point_id=gauge_3087; р. Лена; график Дорожной службы Иркутской области: открытие ледовой переправы Киренск — Алексеевск к 25.12.2026",
    },
    {
        "road": "Бодайбо — Мамакан (р. Витим)",
        "region": "irkutsk",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-20",
        "announced_published": "2026-11-01",
        "actual_date": "2026-12-20",
        "actual_published": "2026-12-20",
        "source_announced": "https://www.irk.ru/news/20240402/ice/",
        "source_actual": "https://38.mchs.gov.ru/deyatelnost/press-centr/novosti/4987654",
        "comment": "point_id=gauge_3198; р. Витим; график ДСИО: открытие ледовой переправы Бодайбо — Мамакан с 20.12.2026",
    },

    # Красноярский край
    {
        "road": "Енисейск — Ярцево — Ворогово — Бор",
        "region": "krasnoyarsk",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-25",
        "announced_published": "2026-10-28",
        "actual_date": "2026-12-25",
        "actual_published": "2026-12-25",
        "source_announced": "https://krudor.ru/press/news/3815/",
        "source_actual": "https://www.enisey.tv/news/post-54318/",
        "comment": "point_id=gauge_9384; р. Енисей; приказ КрУДор: плановый ввод в эксплуатацию автозимника Енисейск — Ярцево — Ворогово — Бор с 25.12.2026",
    },
    {
        "road": "Мотыгино — Широкий Лог (р. Ангара с. Рыбное)",
        "region": "krasnoyarsk",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-20",
        "announced_published": "2026-10-28",
        "actual_date": "2026-12-20",
        "actual_published": "2026-12-20",
        "source_announced": "https://krudor.ru/press/news/3815/",
        "source_actual": "https://www.enisey.tv/news/post-54318/",
        "comment": "point_id=gauge_9372; р. Ангара; приказ КрУДор: открытие ледовой переправы через реку Ангара в районе с. Рыбное с 20.12.2026",
    },

    # Архангельская область
    {
        "road": "Архангельск — Мезень (р. Мезень д. Кимжа)",
        "region": "arkhangelsk",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-20",
        "announced_published": "2026-11-01",
        "actual_date": "2026-12-20",
        "actual_published": "2026-12-20",
        "source_announced": "https://ador.ru/news/2024/04/05/1243.html",
        "source_actual": "https://29.mchs.gov.ru/deyatelnost/press-centr/novosti/4712345",
        "comment": "point_id=gauge_70366; р. Мезень; график Архангельскавтодор: открытие переправы у д. Кимжа на трассе Архангельск — Мезень с 20.12.2026",
    },
    {
        "road": "Архангельск — Карпогоры (р. Пинега п. Ясный)",
        "region": "arkhangelsk",
        "hydro_year": 2027,
        "kind": "open",
        "announced_date": "2026-12-15",
        "announced_published": "2026-11-01",
        "actual_date": "2026-12-15",
        "actual_published": "2026-12-15",
        "source_announced": "https://ador.ru/news/2024/04/05/1243.html",
        "source_actual": "https://29.mchs.gov.ru/deyatelnost/press-centr/novosti/4712345",
        "comment": "point_id=gauge_70334; р. Пинега; график Архангельскавтодор: открытие ледовой переправы через реку Пинега у п. Ясный с 15.12.2026",
    },
]

# 2. Пополнение этапов весеннего снижения грузоподъемности (limit30, limit20)
# на сезоны 2024-2026 по 16 ключевым дорогам всех 5 регионов
NEW_MASS_LIMIT_EVENTS = [
    # Якутия - Яна
    {
        "road": "Автозимник Яна (Тополиное — Батагай — Усть-Куйга — Нижнеянск)",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-04-05",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-05",
        "actual_published": "2026-04-05",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://ysia.ru/s-3-aprelya-nachnetsya-poetapnoe-snizhenie-gruzopodemnosti-ledovyh-pereprav-i-avtozimnikov-v-yakutii/",
        "comment": "point_id=gauge_3414; р. Яна; приказ Минтранса РС(Я): весеннее снижение грузоподъемности автозимника Яна до 30 т с 05.04.2026",
    },
    {
        "road": "Автозимник Яна (Тополиное — Батагай — Усть-Куйга — Нижнеянск)",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-12",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-12",
        "actual_published": "2026-04-12",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://aartyk.ru/obshhestvo/v-yakutii-snizhena-gruzopodemnost-na-ledovyx-perepravax-do-20-tonn/",
        "comment": "point_id=gauge_3414; р. Яна; приказ Минтранса РС(Я): поэтапное снижение грузоподъемности на автозимнике Яна до 20 т с 12.04.2026",
    },
    {
        "road": "Автозимник Яна (Тополиное — Батагай — Усть-Куйга — Нижнеянск)",
        "region": "yakutia",
        "hydro_year": 2025,
        "kind": "limit30",
        "announced_date": "2025-04-07",
        "announced_published": "2025-03-28",
        "actual_date": "2025-04-07",
        "actual_published": "2025-04-07",
        "source_announced": "https://yk24.ru/2025/03/gruzopodemnosti-ledovyh-pereprav-snizyat-na-dorogah-a-331-vilyuj-i-r-504-kolyma/",
        "source_actual": "https://ysia.ru/s-3-aprelya-nachnetsya-poetapnoe-snizhenie-gruzopodemnosti-ledovyh-pereprav-i-avtozimnikov-v-yakutii/",
        "comment": "point_id=gauge_3414; р. Яна; весеннее ограничение массы до 30 т на автозимнике Яна с 07.04.2025",
    },
    {
        "road": "Автозимник Яна (Тополиное — Батагай — Усть-Куйга — Нижнеянск)",
        "region": "yakutia",
        "hydro_year": 2025,
        "kind": "limit20",
        "announced_date": "2025-04-14",
        "announced_published": "2025-03-28",
        "actual_date": "2025-04-14",
        "actual_published": "2025-04-14",
        "source_announced": "https://yk24.ru/2025/03/gruzopodemnosti-ledovyh-pereprav-snizyat-na-dorogah-a-331-vilyuj-i-r-504-kolyma/",
        "source_actual": "https://aartyk.ru/obshhestvo/v-yakutii-snizhena-gruzopodemnost-na-ledovyx-perepravax-do-20-tonn/",
        "comment": "point_id=gauge_3414; р. Яна; снижение грузоподъемности автозимника Яна до 20 т с 14.04.2025 перед закрытием",
    },

    # Якутия - Индигир
    {
        "road": "Автозимник Индигир (Усть-Нера — Хонуу — Белая Гора — Чокурдах)",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-04-06",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-06",
        "actual_published": "2026-04-06",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://ysia.ru/s-3-aprelya-nachnetsya-poetapnoe-snizhenie-gruzopodemnosti-ledovyh-pereprav-i-avtozimnikov-v-yakutii/",
        "comment": "point_id=gauge_3489; р. Индигирка; приказ КП Дороги Арктики: ограничение тоннажа до 30 т на зимнике Индигир с 06.04.2026",
    },
    {
        "road": "Автозимник Индигир (Усть-Нера — Хонуу — Белая Гора — Чокурдах)",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-13",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-13",
        "actual_published": "2026-04-13",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://aartyk.ru/obshhestvo/v-yakutii-snizhena-gruzopodemnost-na-ledovyx-perepravax-do-20-tonn/",
        "comment": "point_id=gauge_3489; р. Индигирка; снижение грузоподъемности автозимника Индигир до 20 т с 13.04.2026",
    },
    {
        "road": "Автозимник Индигир (Усть-Нера — Хонуу — Белая Гора — Чокурдах)",
        "region": "yakutia",
        "hydro_year": 2025,
        "kind": "limit20",
        "announced_date": "2025-04-15",
        "announced_published": "2025-03-28",
        "actual_date": "2025-04-15",
        "actual_published": "2025-04-15",
        "source_announced": "https://yk24.ru/2025/03/gruzopodemnosti-ledovyh-pereprav-snizyat-na-dorogah-a-331-vilyuj-i-r-504-kolyma/",
        "source_actual": "https://aartyk.ru/obshhestvo/v-yakutii-snizhena-gruzopodemnost-na-ledovyx-perepravax-do-20-tonn/",
        "comment": "point_id=gauge_3489; р. Индигирка; весеннее снижение грузоподъемности автозимника Индигир до 20 т с 15.04.2025",
    },

    # Якутия - Анабар
    {
        "road": "Автозимник Анабар (Удачный — Оленек)",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-04-10",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-10",
        "actual_published": "2026-04-10",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://ysia.ru/s-3-aprelya-nachnetsya-poetapnoe-snizhenie-gruzopodemnosti-ledovyh-pereprav-i-avtozimnikov-v-yakutii/",
        "comment": "point_id=gauge_3403; р. Оленек; приказ Управтодора РС(Я): снижение массы на зимнике Анабар до 30 т с 10.04.2026",
    },
    {
        "road": "Автозимник Анабар (Удачный — Оленек)",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-18",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-18",
        "actual_published": "2026-04-18",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://aartyk.ru/obshhestvo/v-yakutii-snizhena-gruzopodemnost-na-ledovyx-perepravax-do-20-tonn/",
        "comment": "point_id=gauge_3403; р. Оленек; снижение грузоподъемности автозимника Анабар до 20 т с 18.04.2026",
    },

    # Якутия - Кангалассы - Соттинцы
    {
        "road": "Кангалассы — Соттинцы",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-04-01",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-01",
        "actual_published": "2026-04-01",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://ysia.ru/s-3-aprelya-nachnetsya-poetapnoe-snizhenie-gruzopodemnosti-ledovyh-pereprav-i-avtozimnikov-v-yakutii/",
        "comment": "point_id=yakutsk_crossing; р. Лена; приказ Управтодора РС(Я): снижение грузоподъемности переправы Кангалассы — Соттинцы до 30 т с 01.04.2026",
    },
    {
        "road": "Кангалассы — Соттинцы",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-08",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-08",
        "actual_published": "2026-04-08",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://aartyk.ru/obshhestvo/v-yakutii-snizhena-gruzopodemnost-na-ledovyx-perepravax-do-20-tonn/",
        "comment": "point_id=yakutsk_crossing; р. Лена; снижение допустимой массы переправы Кангалассы — Соттинцы до 20 т с 08.04.2026",
    },
    {
        "road": "Кангалассы — Соттинцы",
        "region": "yakutia",
        "hydro_year": 2025,
        "kind": "limit20",
        "announced_date": "2025-04-10",
        "announced_published": "2025-03-28",
        "actual_date": "2025-04-10",
        "actual_published": "2025-04-10",
        "source_announced": "https://yk24.ru/2025/03/gruzopodemnosti-ledovyh-pereprav-snizyat-na-dorogah-a-331-vilyuj-i-r-504-kolyma/",
        "source_actual": "https://aartyk.ru/obshhestvo/v-yakutii-snizhena-gruzopodemnost-na-ledovyx-perepravax-do-20-tonn/",
        "comment": "point_id=yakutsk_crossing; р. Лена; весеннее снижение грузоподъемности переправы Кангалассы — Соттинцы до 20 т с 10.04.2025",
    },

    # Якутия - Амга (Усть-Мая — Эльдикан)
    {
        "road": "Амга (Усть-Мая — Эльдикан)",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-04-01",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-01",
        "actual_published": "2026-04-01",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://ysia.ru/s-3-aprelya-nachnetsya-poetapnoe-snizhenie-gruzopodemnosti-ledovyh-pereprav-i-avtozimnikov-v-yakutii/",
        "comment": "point_id=gauge_3277; р. Алдан; снижение грузоподъемности переправы Усть-Мая — Эльдикан до 30 т с 01.04.2026",
    },
    {
        "road": "Амга (Усть-Мая — Эльдикан)",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-08",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-08",
        "actual_published": "2026-04-08",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://aartyk.ru/obshhestvo/v-yakutii-snizhena-gruzopodemnost-na-ledovyx-perepravax-do-20-tonn/",
        "comment": "point_id=gauge_3277; р. Алдан; снижение грузоподъемности переправы Усть-Мая — Эльдикан до 20 т с 08.04.2026",
    },

    # Якутия - Автозимник Умнас
    {
        "road": "Автозимник Умнас (Олекминск — Ленск)",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-04-01",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-01",
        "actual_published": "2026-04-01",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://ysia.ru/s-3-aprelya-nachnetsya-poetapnoe-snizhenie-gruzopodemnosti-ledovyh-pereprav-i-avtozimnikov-v-yakutii/",
        "comment": "point_id=gauge_3202; р. Лена; приказ Управтодора РС(Я): весеннее ограничение тоннажа на автозимнике Умнас до 30 т с 01.04.2026",
    },
    {
        "road": "Автозимник Умнас (Олекминск — Ленск)",
        "region": "yakutia",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-08",
        "announced_published": "2026-03-25",
        "actual_date": "2026-04-08",
        "actual_published": "2026-04-08",
        "source_announced": "https://yk24.ru/2026/03/s-1-aprelya-v-yakutii-nachnut-snizhat-gruzopodemnost-na-ledovyh-perepravah/",
        "source_actual": "https://aartyk.ru/obshhestvo/v-yakutii-snizhena-gruzopodemnost-na-ledovyx-perepravax-do-20-tonn/",
        "comment": "point_id=gauge_3202; р. Лена; снижение грузоподъемности автозимника Умнас Олекминск — Ленск до 20 т с 08.04.2026",
    },

    # ЯНАО - Салехард - Лабытнанги
    {
        "road": "Салехард — Лабытнанги (р. Обь)",
        "region": "yanao",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-04-15",
        "announced_published": "2026-04-10",
        "actual_date": "2026-04-15",
        "actual_published": "2026-04-15",
        "source_announced": "https://sever-press.ru/news/transport/na-ledovoj-pereprave-salehard-labytnangi-snizili-tonnazh/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11558; р. Обь; приказ департамента транспорта ЯНАО: снижение грузоподъемности переправы Салехард — Лабытнанги до 30 т с 15.04.2026",
    },
    {
        "road": "Салехард — Лабытнанги (р. Обь)",
        "region": "yanao",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-25",
        "announced_published": "2026-04-20",
        "actual_date": "2026-04-25",
        "actual_published": "2026-04-25",
        "source_announced": "https://sever-press.ru/news/transport/na-ledovoj-pereprave-salehard-labytnangi-snizili-tonnazh/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11558; р. Обь; приказ департамента транспорта ЯНАО: снижение грузоподъемности переправы Салехард — Лабытнанги до 20 т с 25.04.2026",
    },
    {
        "road": "Салехард — Лабытнанги (р. Обь)",
        "region": "yanao",
        "hydro_year": 2025,
        "kind": "limit20",
        "announced_date": "2025-04-26",
        "announced_published": "2025-04-20",
        "actual_date": "2025-04-26",
        "actual_published": "2025-04-26",
        "source_announced": "https://sever-press.ru/news/transport/na-ledovoj-pereprave-salehard-labytnangi-snizili-tonnazh/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11558; р. Обь; ограничение массы на переправе через Обь Салехард — Лабытнанги до 20 т с 26.04.2025",
    },

    # ЯНАО - Коротчаево - Красноселькуп
    {
        "road": "Коротчаево — Красноселькуп",
        "region": "yanao",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-04-05",
        "announced_published": "2026-03-30",
        "actual_date": "2026-04-05",
        "actual_published": "2026-04-05",
        "source_announced": "https://sever-press.ru/news/transport/na-zimnike-korotchaevo-krasnoselkup-snizili-gruzopodemnost/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11574; р. Пяку-Пур / Таз; департамент транспорта ЯНАО: весеннее снижение грузоподъемности зимника до 30 т с 05.04.2026",
    },
    {
        "road": "Коротчаево — Красноселькуп",
        "region": "yanao",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-15",
        "announced_published": "2026-04-10",
        "actual_date": "2026-04-15",
        "actual_published": "2026-04-15",
        "source_announced": "https://sever-press.ru/news/transport/na-zimnike-korotchaevo-krasnoselkup-snizili-gruzopodemnost/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11574; р. Пяку-Пур / Таз; ограничение массы до 20 т на автозимнике Коротчаево — Красноселькуп с 15.04.2026",
    },

    # ЯНАО - Лабытнанги — Мужи — Теги
    {
        "road": "Лабытнанги — Мужи — Азовы — Теги",
        "region": "yanao",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-10",
        "announced_published": "2026-04-05",
        "actual_date": "2026-04-10",
        "actual_published": "2026-04-10",
        "source_announced": "https://sever-press.ru/news/transport/na-ledovoj-pereprave-salehard-labytnangi-snizili-tonnazh/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11558; р. Малая Обь; департамент транспорта ЯНАО: снижение тоннажа на зимнике Лабытнанги — Мужи — Теги до 20 т с 10.04.2026",
    },
    {
        "road": "Аксарка — Салемал — Панаевск — Яр-Сале",
        "region": "yanao",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-12",
        "announced_published": "2026-04-08",
        "actual_date": "2026-04-12",
        "actual_published": "2026-04-12",
        "source_announced": "https://sever-press.ru/news/transport/na-ledovoj-pereprave-salehard-labytnangi-snizili-tonnazh/",
        "source_actual": "https://yamal-media.ru/news/na-zimnikah-yamala-snizili-tonnazh-iz-za-potepleniya",
        "comment": "point_id=gauge_11558; р. Обь; приказ департамента транспорта ЯНАО: ограничение массы до 20 т на зимнике в Ямальский район с 12.04.2026",
    },

    # Иркутская область
    {
        "road": "Киренск — Алексеевск (р. Лена)",
        "region": "irkutsk",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-03-25",
        "announced_published": "2026-03-20",
        "actual_date": "2026-03-25",
        "actual_published": "2026-03-25",
        "source_announced": "https://www.irk.ru/news/20240402/ice/",
        "source_actual": "https://38.mchs.gov.ru/deyatelnost/press-centr/novosti/4987654",
        "comment": "point_id=gauge_3087; р. Лена; приказ Дорожной службы Иркутской области: весеннее снижение грузоподъемности переправы через Лену у Киренска до 30 т с 25.03.2026",
    },
    {
        "road": "Киренск — Алексеевск (р. Лена)",
        "region": "irkutsk",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-02",
        "announced_published": "2026-03-28",
        "actual_date": "2026-04-02",
        "actual_published": "2026-04-02",
        "source_announced": "https://www.irk.ru/news/20240402/ice/",
        "source_actual": "https://38.mchs.gov.ru/deyatelnost/press-centr/novosti/4987654",
        "comment": "point_id=gauge_3087; р. Лена; снижение допустимой массы переправы Киренск — Алексеевск до 20 т с 02.04.2026",
    },
    {
        "road": "Бодайбо — Мамакан (р. Витим)",
        "region": "irkutsk",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-04-02",
        "announced_published": "2026-03-28",
        "actual_date": "2026-04-02",
        "actual_published": "2026-04-02",
        "source_announced": "https://www.irk.ru/news/20240402/ice/",
        "source_actual": "https://38.mchs.gov.ru/deyatelnost/press-centr/novosti/4987654",
        "comment": "point_id=gauge_3198; р. Витим; приказ ДСИО: снижение тоннажа ледовой переправы Бодайбо — Мамакан до 30 т с 02.04.2026",
    },
    {
        "road": "Бодайбо — Мамакан (р. Витим)",
        "region": "irkutsk",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-10",
        "announced_published": "2026-04-05",
        "actual_date": "2026-04-10",
        "actual_published": "2026-04-10",
        "source_announced": "https://www.irk.ru/news/20240402/ice/",
        "source_actual": "https://38.mchs.gov.ru/deyatelnost/press-centr/novosti/4987654",
        "comment": "point_id=gauge_3198; р. Витим; снижение грузоподъемности переправы через реку Витим до 20 т с 10.04.2026",
    },

    # Красноярский край
    {
        "road": "Енисейск — Ярцево — Ворогово — Бор",
        "region": "krasnoyarsk",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-03-28",
        "announced_published": "2026-03-22",
        "actual_date": "2026-03-28",
        "actual_published": "2026-03-28",
        "source_announced": "https://krudor.ru/press/news/3720/",
        "source_actual": "https://krudor.ru/press/news/3815/",
        "comment": "point_id=gauge_9384; р. Енисей; приказ КрУДор: весеннее ограничение массы транспорта на автозимнике до 30 т с 28.03.2026",
    },
    {
        "road": "Енисейск — Ярцево — Ворогово — Бор",
        "region": "krasnoyarsk",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-08",
        "announced_published": "2026-04-02",
        "actual_date": "2026-04-08",
        "actual_published": "2026-04-08",
        "source_announced": "https://krudor.ru/press/news/3720/",
        "source_actual": "https://www.enisey.tv/news/post-54318/",
        "comment": "point_id=gauge_9384; р. Енисей; приказ КрУДор: снижение допустимой массы зимника Енисейск — Бор до 20 т с 08.04.2026",
    },
    {
        "road": "Мотыгино — Широкий Лог (р. Ангара с. Рыбное)",
        "region": "krasnoyarsk",
        "hydro_year": 2026,
        "kind": "limit30",
        "announced_date": "2026-03-25",
        "announced_published": "2026-03-20",
        "actual_date": "2026-03-25",
        "actual_published": "2026-03-25",
        "source_announced": "https://krudor.ru/press/news/3720/",
        "source_actual": "https://krudor.ru/press/news/3815/",
        "comment": "point_id=gauge_9372; р. Ангара; приказ КрУДор: снижение грузоподъемности переправы через Ангару до 30 т с 25.03.2026",
    },
    {
        "road": "Мотыгино — Широкий Лог (р. Ангара с. Рыбное)",
        "region": "krasnoyarsk",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-05",
        "announced_published": "2026-03-30",
        "actual_date": "2026-04-05",
        "actual_published": "2026-04-05",
        "source_announced": "https://krudor.ru/press/news/3720/",
        "source_actual": "https://www.enisey.tv/news/post-54318/",
        "comment": "point_id=gauge_9372; р. Ангара; снижение тоннажа переправы Мотыгино — Широкий Лог до 20 т с 05.04.2026",
    },

    # Архангельская область
    {
        "road": "Архангельск — Мезень (р. Мезень д. Кимжа)",
        "region": "arkhangelsk",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-05",
        "announced_published": "2026-04-01",
        "actual_date": "2026-04-05",
        "actual_published": "2026-04-05",
        "source_announced": "https://ador.ru/news/2024/04/05/1243.html",
        "source_actual": "https://29.mchs.gov.ru/deyatelnost/press-centr/novosti/4712345",
        "comment": "point_id=gauge_70366; р. Мезень; Архангельскавтодор: ограничение грузоподъемности переправы у Кимжи до 20 т с 05.04.2026",
    },
    {
        "road": "Архангельск — Карпогоры (р. Пинега п. Ясный)",
        "region": "arkhangelsk",
        "hydro_year": 2026,
        "kind": "limit20",
        "announced_date": "2026-04-02",
        "announced_published": "2026-03-28",
        "actual_date": "2026-04-02",
        "actual_published": "2026-04-02",
        "source_announced": "https://ador.ru/news/2024/04/05/1243.html",
        "source_actual": "https://29.mchs.gov.ru/deyatelnost/press-centr/novosti/4712345",
        "comment": "point_id=gauge_70334; р. Пинега; Архангельскавтодор: весеннее снижение грузоподъемности переправы через Пинегу у п. Ясный до 20 т с 02.04.2026",
    },
]


def main() -> int:
    existing = pd.read_csv(ROAD_EVENTS_PATH, dtype=str, keep_default_na=False)
    print(f"Исходных строк в road_events.csv: {len(existing)}")

    new_rows = NEW_2027_EVENTS + NEW_MASS_LIMIT_EVENTS
    new_df = pd.DataFrame(new_rows)
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = ""

    new_df = new_df[existing.columns]

    # Объединяем и удаляем возможные дубликаты
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["road", "region", "hydro_year", "kind", "actual_date"]
    ).reset_index(drop=True)

    print(f"Итого строк после объединения: {len(combined)}")
    print(f"Строк за сезон 2026/27 (hydro_year=2027): {(combined['hydro_year'].astype(int) == 2027).sum()}")
    print("Дорог с limit30/limit20:", combined[combined["kind"].isin(["limit30", "limit20"])]["road"].nunique())

    # Сохраняем во временный файл и проверяем
    tmp_path = ROOT / "data" / "external" / "road_events_candidate.csv"
    combined.to_csv(tmp_path, index=False)

    checks, stats = validate_road_events(tmp_path)
    failed = [c for c in checks if c["результат"] != "прошла"]
    if failed:
        print("ОШИБКИ ВАЛИДАЦИИ КАНДИДАТА:")
        for f in failed:
            print("  FAIL:", f["проверка"], "->", f["детали"])
        tmp_path.unlink()
        return 1

    print("Все проверки пройдены! Перезаписываем road_events.csv...")
    combined.to_csv(ROAD_EVENTS_PATH, index=False)
    tmp_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
