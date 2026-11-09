"""Скриншоты дашборда для туториала (docs/TUTORIAL.pdf).

    streamlit run app/dashboard.py --server.port 8510 --server.headless true   # в отдельном окне
    python scripts/tutorial_screenshots.py docs/tutorial/img
    python scripts/make_tutorial.py

Нужен Playwright (pip install playwright) и установленный Microsoft Edge (channel="msedge").
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8510"


def settle(page, extra=1.5, timeout=120):
    t = time.time()
    time.sleep(1.0)
    while time.time() - t < timeout:
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def shot(page, name, locator=None, full=False):
    p = OUT / f"{name}.png"
    page.mouse.move(1490, 940)
    page.add_style_tag(content='[data-testid="stToolbar"],[data-testid="stTooltipContent"],[role="tooltip"]{display:none!important}')
    time.sleep(0.4)
    if locator is not None:
        locator.screenshot(path=str(p))
    else:
        page.screenshot(path=str(p), full_page=full)
    print("saved", p.name)


def tab(page, label):
    page.get_by_role("tab", name=label).click()
    settle(page, 2)


def section(page, text, name, height=900):
    """Снимок окна, прокрученного к заголовку с текстом."""
    el = page.get_by_text(text, exact=False).first
    el.scroll_into_view_if_needed()
    page.mouse.wheel(0, -60)
    time.sleep(1.2)
    shot(page, name)


with sync_playwright() as pw:
    b = pw.chromium.launch(channel="msedge", headless=True)
    page = b.new_page(viewport={"width": 1500, "height": 950}, device_scale_factor=1)
    page.goto(URL); settle(page, 2)
    shot(page, "00_welcome")
    page.get_by_role("button", name="Открыть дашборд").click(); settle(page, 4)
    shot(page, "01_map")
    shot(page, "02_sidebar", page.locator('[data-testid="stSidebar"]'))
    section(page, "Пути с прогнозом", "01b_map_legend")
    page.mouse.wheel(0, -5000); time.sleep(1)

    tab(page, "Карточка пути")
    shot(page, "03_card")
    section(page, "Как менялся прогноз", "03b_card_chart")
    section(page, "Лена по радарным снимкам", "03c_card_m2")
    page.mouse.wheel(0, -5000); time.sleep(1)

    tab(page, "Сквозной коридор (А3)")
    settle(page, 3)
    shot(page, "04_corridor_top")
    page.get_by_text("Параметры и тарифы 4 звеньев маршрута").click(); settle(page, 1.5)
    section(page, "Параметры и тарифы 4 звеньев маршрута", "04b_corridor_params")
    page.get_by_text("Параметры и тарифы 4 звеньев маршрута").click(); settle(page, 1.5)
    section(page, "Подход для подробного сравнения", "04c_corridor_compare")
    section(page, "Режим «сервис»", "05_service")
    el = page.locator('[class*="st-key-corr_service_chart"]').first
    el.scroll_into_view_if_needed(); time.sleep(1.5)
    shot(page, "05b_service_chart", el)
    section(page, "План отправок по неделям", "06_weekly")
    section(page, "Все звенья и издержки ожидания", "07_costs")
    section(page, "Запасы и фактическая распутица", "07b_stocks")
    page.mouse.wheel(0, -20000); time.sleep(1)

    tab(page, "Планировщик")
    shot(page, "08_planner")
    section(page, "Экономический ML: отправки", "08b_planner_charts")
    page.mouse.wheel(0, -20000); time.sleep(1)

    tab(page, "Рейтинг риска")
    settle(page, 3)
    shot(page, "09_risk")
    tab(page, "До / после")
    shot(page, "10_before_after")
    tab(page, "О данных")
    shot(page, "11_about")

    page.get_by_role("button", name="Эксперимент: весна 2026").click(); settle(page, 5)
    shot(page, "12_replay")
    section(page, "Разница расходов", "12b_replay_result")
    section(page, "Накопленная доставка", "12c_replay_chart")
    b.close()
