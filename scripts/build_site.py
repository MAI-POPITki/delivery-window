"""
Статичная страница проекта для GitHub Pages → docs/index.html.

    python scripts/build_site.py

Берёт цифры из reports/ (после run_all.py), поэтому страница всегда совпадает с расчётом.
GitHub Pages: Settings → Pages → Deploy from a branch → main, папка /docs.
"""

import html
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go

from okno.paths import REPORTS, ROOT

REPO_URL = "https://github.com/MAI-POPITki/delivery-window"
DASHBOARD_URL = "https://okno-zavoza-dmitrii.streamlit.app/"
DOCS = ROOT / "docs"

BLUE, ORANGE, GREY = "#1f4e79", "#e07b00", "#9aa5b1"


def fmt(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def fig_html(fig, div_id, first=False):
    fig.update_layout(template="plotly_white", font=dict(family="Inter, Arial, sans-serif", size=13),
                      margin=dict(l=10, r=10, t=40, b=10), height=360,
                      legend=dict(orientation="h", y=-0.18), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False, div_id=div_id,
                       config={"displaylogo": False, "responsive": True})


def m2_section() -> str:
    """Раздел М2: точность классификатора, состояние Лены по последним снимкам, примеры вырезок."""
    import shutil
    met, pred = REPORTS / "m2_metrics.csv", ROOT / "data" / "s1" / "predictions.csv"
    if not (met.exists() and pred.exists()):
        return ""
    from okno.s1_points import LENA_NAMES
    m = pd.read_csv(met).iloc[0]
    p = pd.read_csv(pred, parse_dates=["date"]).dropna(subset=["p_ice"])
    last = p.sort_values("date").groupby("point").tail(1).sort_values("date", ascending=False)
    rows = "".join(
        f"<tr><td>{html.escape(LENA_NAMES.get(r.point, r.point))}</td><td>{r.date:%d.%m.%Y}</td>"
        f"<td><b>{'лёд' if r.p_ice >= 0.8 else 'вода' if r.p_ice <= 0.2 else 'неясно'}</b></td>"
        f"<td>{r.p_ice:.2f}</td></tr>" for r in last.itertuples())
    img = ""
    if (REPORTS / "s1_preview.png").exists():
        DOCS.mkdir(exist_ok=True)
        shutil.copy(REPORTS / "s1_preview.png", DOCS / "s1_preview.png")
        img = '<p><img src="s1_preview.png" alt="Примеры вырезок Sentinel-1" style="max-width:100%;border-radius:10px"></p>'
    return f"""<section><h2>М2: спутник видит лёд</h2>
<p>Нейросетевой блок проекта: по радарным снимкам Sentinel-1 (видят сквозь облака и полярную ночь) модель определяет,
что сейчас на Лене — <b>лёд или вода</b>. 7 точек от Ленска до Кюсюра, снимки обновляются каждый день.</p>
<div class="kpis"><div class="kpi"><div class="v">{fmt(100 * m['точность'])}%</div><div class="l">точность на годах вне обучения</div></div>
<div class="kpi"><div class="v">{fmt(len(p))}</div><div class="l">снимков 2015–2026 размечено и проверено</div></div>
<div class="kpi"><div class="v">±9 сут</div><div class="l">дата вскрытия по ряду снимков</div></div></div>
<table style="width:100%;border-collapse:collapse"><tr><th align="left">Точка на Лене</th><th align="left">Снимок</th>
<th align="left">Состояние</th><th align="left">P(лёд)</th></tr>{rows}</table>
{img}
<p class="muted">Метки для обучения — спутниковые даты льда (Qiu et al. 2026); проверка — сезон целиком вне обучения.
Честно: как поправка к прогнозу вскрытия М2 точности не добавила — её ценность в факте события там, где нет гидропоста,
и в живом состоянии реки. <a href="{REPO_URL}/blob/main/reports/m2_metrics.md">Отчёт об эффективности М2</a>.</p></section>"""


def main():
    met = pd.read_csv(REPORTS / "m1_metrics.csv")
    ba = pd.read_csv(REPORTS / "before_after_arctic.csv")
    sens = pd.read_csv(REPORTS / "sensitivity.csv")
    checks = pd.read_csv(REPORTS / "data_checks.csv")

    c = ba[ba.rule == "calendar"]
    f = ba[ba.rule == "forecast"]
    open14 = met[(met.kind == "open") & (met["за_сколько_дней"] == "14")].iloc[0]
    n_ok = (checks["результат"] == "прошла").sum()
    robust = sens[sens.point_id == "gauge_3414"]["после лучше"].mean()

    kpis = [
        (f"{fmt(c.undelivered_t.mean())} → {fmt(f.undelivered_t.mean())} т", "недовоз за сезон на зимнике до Батагая"),
        (f"{fmt(c.cost_rub_per_t.mean() / 1000, 1)} → {fmt(f.cost_rub_per_t.mean() / 1000, 1)} тыс. ₽/т",
         "расходы с учётом дорогих каналов"),
        (f"−{fmt(open14['выигрыш_к_календарю_%'])}%", "ошибка прогноза открытия за 14 дней"),
        (f"{n_ok} / {len(checks)}", "проверок данных пройдено"),
    ]

    # график 1: точность М1
    rows = []
    for kind, grp, name in (("open", "гидропосты (замеры)", "Открытие"),
                            ("close", "гидропосты (замеры)", "Закрытие, посты"),
                            ("close", "переправа Якутск (спутник)", "Закрытие, Якутск")):
        m = met[(met.kind == kind) & (met["группа"] == grp) & (met["за_сколько_дней"] == "14")]
        if len(m):
            rows.append((name, m.iloc[0]["MAE_календарь"], m.iloc[0]["MAE_М1"]))
    g1 = go.Figure([
        go.Bar(name="Календарь", x=[r[0] for r in rows], y=[r[1] for r in rows], marker_color=GREY),
        go.Bar(name="Модель М1", x=[r[0] for r in rows], y=[r[2] for r in rows], marker_color=BLUE),
    ])
    g1.update_layout(title="Средняя ошибка за 14 дней до события, сут", barmode="group")

    # график 2: недовоз по сезонам
    p = ba.pivot_table(index="season", columns="rule", values="undelivered_t")
    g2 = go.Figure([
        go.Bar(name="До: календарь", x=p.index, y=p["calendar"], marker_color=GREY),
        go.Bar(name="После: прогноз", x=p.index, y=p["forecast"], marker_color=ORANGE),
    ])
    g2.update_layout(title="Недовезено до закрытия зимника, т (бэктест по прошлым сезонам)", barmode="group")

    # график 3: переносимость — модель обучена только на Якутии
    m2_html = m2_section()
    transfer_html = ""
    tf = REPORTS / "transfer_m1.csv"
    if tf.exists():
        t = pd.read_csv(tf)
        t = t[(t["событие"].str.startswith("открытие")) & (t["сценарий"].str.startswith("B"))]
        t = t[t["регион"] != "все регионы"].sort_values("выигрыш_%")
        g3 = go.Figure([
            go.Bar(name="Календарь", y=t["регион"], x=t["MAE_календарь"], orientation="h", marker_color=GREY),
            go.Bar(name="М1, обученная только на Якутии", y=t["регион"], x=t["MAE_М1"], orientation="h",
                   marker_color=BLUE),
        ])
        g3.update_layout(title="Открытие в других регионах: средняя ошибка за 14 дней, сут", barmode="group")
        allr = pd.read_csv(tf)
        allr = allr[(allr["регион"] == "все регионы") & allr["событие"].str.startswith("открытие")
                    & allr["сценарий"].str.startswith("B")].iloc[0]
        transfer_html = f"""<section><h2>Работает ли в других регионах</h2>
<p class="muted">Модель обучена только на Якутии и проверена на гидропостах ЯНАО, Красноярского края, Иркутской и
Архангельской областей и Коми — без переобучения и без местных замеров толщины льда.</p>
{fig_html(g3, "chart-transfer")}
<p>Прогноз открытия в чужих регионах точнее календаря на <b>{fmt(allr['выигрыш_%'])}%</b>. Прогноз закрытия без
местной донастройки не переносится, а на единственном посту Коми модель хуже календаря — это тоже показываем.</p>
<p><b>По итогам проверки</b> ЯНАО, Красноярский край, Иркутская и Архангельская области включены в сервис —
переключатель «Регион» в дашборде; Коми не включена, Архангельская область помечена как подтверждённая слабо.
<a href="{REPO_URL}/blob/main/reports/transferability.md">Подробный разбор с цифрами по регионам</a>.</p></section>"""

    kpi_html = "".join(f'<div class="kpi"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(l)}</div></div>'
                       for v, l in kpis)
    dash = (f'<a class="btn" href="{DASHBOARD_URL}">Открыть живой дашборд</a>' if DASHBOARD_URL else
            '<span class="btn muted">Живой дашборд — ссылка появится после деплоя</span>')

    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Окно завоза</title>
<link rel="icon" type="image/png" href="logo.png">
<meta name="description" content="Прогноз окон сезонных путей Якутии и график северного завоза">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root{{--bg:#f6f8fb;--card:#fff;--text:#16212c;--muted:#5b6b7b;--accent:{BLUE};--line:#e3e8ef;--warn:#fff4e5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f151c;--card:#17202a;--text:#e8eef4;--muted:#9fb0c0;--accent:#6aa7e0;--line:#26323f;--warn:#2b2214}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 Inter,Arial,sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:0 16px}}
header{{padding:56px 0 28px}} h1{{font-size:clamp(32px,6vw,52px);margin:0 0 8px;letter-spacing:-.02em}}
.hero{{display:flex;align-items:center;gap:20px;flex-wrap:wrap}}
.hero-logo{{width:96px;height:96px;object-fit:contain;flex:0 0 auto}}
.lead{{font-size:19px;color:var(--muted);max-width:760px;margin:0}}
.actions{{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}}
.btn{{display:inline-block;padding:10px 16px;border-radius:10px;background:var(--accent);color:#fff;text-decoration:none;font-weight:600}}
.btn.ghost{{background:transparent;color:var(--accent);border:1px solid var(--accent)}} .btn.muted{{background:var(--line);color:var(--muted)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:8px 0 28px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}}
.kpi .v{{font-size:24px;font-weight:800;color:var(--accent)}} .kpi .l{{color:var(--muted);font-size:14px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;margin-bottom:16px}}
h2{{margin:0 0 10px;font-size:22px}} .steps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}
.step b{{display:block;margin-bottom:4px}} .note{{background:var(--warn);border-radius:12px;padding:14px 16px;margin-top:12px}}
.muted{{color:var(--muted)}} footer{{padding:24px 0 48px;color:var(--muted);font-size:14px}}
a{{color:var(--accent)}} ul{{padding-left:20px;margin:6px 0}}
</style></head><body>
<div class="wrap">
<header>
<div class="hero">
<img class="hero-logo" src="logo.png" alt="Логотип Окно завоза">
<div>
<h1>Окно завоза</h1>
<p class="lead">Прогнозируем, когда откроется и закроется каждый ледовый путь Якутии, и строим по прогнозу график
северного завоза. Только открытые данные, без датчиков.</p>
</div>
</div>
<div class="actions">{dash}
<a class="btn ghost" href="project.pdf">Документ проекта (PDF)</a>
<a class="btn ghost" href="{REPO_URL}">Код и данные</a></div>
</header>
<div class="kpis">{kpi_html}</div>

<section><h2>Как это работает</h2><div class="steps">
<div class="step"><b>1. Данные</b>Замеры льда на 13 гидропостах Якутии, спутниковые даты льда Лены, погода ERA5 с 1979 года.
Каждая строка помечена типом источника, все проверки — в отчёте.</div>
<div class="step"><b>2. Прогноз окна (М1)</b>Физика нарастания льда (формула Стефана) и CatBoost дают диапазон дат
открытия и закрытия: 10, 50 и 90%.</div>
<div class="step"><b>3. График завоза (М3)</b>1 000 сценариев даты закрытия → сколько везти каждую неделю и нужна ли
дополнительная техника, чтобы успеть и не переплатить.</div>
</div></section>

<section><h2>Точность прогноза</h2>
<p class="muted">Проверка на сезонах, которых модель не видела. Календарь — средняя дата за прошлые 10 сезонов: так планируют сейчас.</p>
{fig_html(g1, "chart-accuracy", first=True)}
<p>Открытие (лёд держит 20 т) модель прогнозирует заметно точнее календаря: от года к году эта дата гуляет на ~20 дней,
и физика морозов это объясняет. Начало таяния гуляет всего на ~7 дней и зависит от погоды апреля — тут модель на уровне календаря.</p>
</section>

<section><h2>«До» и «после» на прошлых сезонах</h2>
<p class="muted">Зимник на Батагай (Верхоянский арктический район). Фактические даты закрытия — по замерам гидропоста «Яна — Верхоянск».</p>
{fig_html(g2, "chart-before-after")}
<p>Вывод «после лучше, чем до» держится в <b>{robust:.0%}</b> вариантов, когда каждое экономическое допущение меняется на ±30%.</p>
<div class="note"><b>Честно о результате.</b> Экономию на закрытии даёт переход от одной календарной даты к распределению
возможных дат: тот же планировщик на разбросе прошлых лет работает не хуже. Тарифы и объёмы — объявленная синтетика,
поэтому рубли — иллюстрация, а устойчивый результат — направление и порядок эффекта.</div>
</section>

{m2_html}

{transfer_html}

<section><h2>Чтобы стало рабочим</h2><ul>
<li>Фактические даты открытия и закрытия зимников из приказов Управтодора РС(Я) за 10–15 лет.</li>
<li>План и факт завоза хотя бы одного торгово-логистического центра.</li>
<li>Сезонные прогнозы погоды Copernicus C3S как признак для прогноза закрытия.</li>
<li>Пилот на живом сезоне 2026/27: прогнозы публикуются заранее и сравниваются с фактом.</li>
</ul><p class="muted">Данные бесплатные, датчики не нужны, сервер — в пределах десятков тысяч рублей в год.</p></section>

<footer>Источники: Shiklomanov et al., Arctic Data Center (doi:10.18739/A22Z12Q02); Qiu et al. 2026, ESSD
(doi:10.5281/zenodo.17054619); ERA5 через Open-Meteo. Кейс-чемпионат, кейс А3.</footer>
</div></body></html>"""
    DOCS.mkdir(exist_ok=True)
    logo_src = ROOT / "app" / "assets" / "logo.png"
    if logo_src.exists():
        shutil.copy2(logo_src, DOCS / "logo.png")
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    print("готово:", DOCS / "index.html")


if __name__ == "__main__":
    main()
