"""Приветственный экран и переходы в существующий дашборд."""

from pathlib import Path

import streamlit as st

PROJECT_PDF = Path(__file__).resolve().parent / "assets" / "project.pdf"

_STYLES = """
<style>
.st-key-welcome {
    max-width: 1080px;
    margin: 0 auto;
}
.st-key-welcome_intro {
    padding: 3.2rem;
    border: 1px solid #ff4b4b;
    border-radius: 24px;
    background: #ff4b4b;
    color: #f4f9fc;
}
.st-key-welcome_intro h1 {
    color: #f4f9fc;
    font-size: clamp(2.6rem, 6vw, 4.4rem);
    font-weight: 700;
    letter-spacing: -0.045em;
    line-height: 1.1;
    padding: 0.3rem 0 1rem;
}
.st-key-welcome_intro .welcome-eyebrow {
    color: #a6dbe2;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin: 0;
}
.st-key-welcome_intro .welcome-lead {
    color: #e1edf3;
    font-size: clamp(1.1rem, 2.4vw, 1.35rem);
    line-height: 1.65;
    max-width: 760px;
    margin: 0 0 0.5rem;
}
.st-key-welcome_intro [data-testid="stCaptionContainer"] p {
    color: #b8cdd8;
    font-size: 0.9rem;
}
.st-key-welcome_actions button {
    min-height: 3.2rem;
    border-radius: 12px;
    font-weight: 600;
}
.st-key-welcome_actions [data-testid="stBaseButton-primary"] {
    background-color: #ff4b4b;
    border-color: #ff4b4b;
    color: white;
}
.st-key-welcome_actions [data-testid="stBaseButton-primary"]:hover {
    background-color: #e03e3e;
    border-color: #e03e3e;
}
.st-key-welcome_actions button:focus-visible {
    outline: 3px solid #ff4b4b;
    outline-offset: 3px;
}
@media (max-width: 640px) {
    .st-key-welcome_intro {
        padding: 1.6rem;
        border-radius: 18px;
    }
}
</style>
"""


def open_dashboard():
    st.session_state["dashboard_open"] = True


def show_welcome():
    st.session_state["dashboard_open"] = False


def open_backhaul():
    """Самостоятельный модуль доступен без загрузки моделей прогноза."""
    st.query_params["module"] = "backhaul"


def open_routes():
    st.query_params["module"] = "routes"


def render_welcome():
    st.markdown(_STYLES, unsafe_allow_html=True)

    with st.container(key="welcome"):
        with st.container(key="welcome_intro"):
            st.markdown(
                '<p class="welcome-eyebrow">Северный завоз · прогноз и планирование</p>',
                unsafe_allow_html=True,
            )
            st.title("Окно завоза")
            st.markdown(
                '<p class="welcome-lead">Помогаем планировать поставки на Север, '
                'пока сезонные пути открыты. Прогнозируем сроки открытия и закрытия '
                'зимников и ледовых переправ и строим по ним график завоза.</p>',
                unsafe_allow_html=True,
            )
            st.caption("Пилот в Якутии: зимник на Батагай и переправа Якутск — Нижний Бестях.")

        st.write("")
        with st.container(key="welcome_actions"):
            dashboard, document = st.columns(2, gap="medium")
            with dashboard:
                st.button(
                    "Открыть дашборд",
                    key="open_dashboard",
                    type="primary",
                    icon=":material/arrow_forward:",
                    on_click=open_dashboard,
                    width="stretch",
                )
            with document:
                try:
                    pdf = PROJECT_PDF.read_bytes()
                except OSError:
                    st.warning("Описание проекта временно недоступно для скачивания.")
                else:
                    st.download_button(
                        "Скачать описание проекта (PDF)",
                        data=pdf,
                        file_name="okno-zavoza-project.pdf",
                        mime="application/pdf",
                        key="download_project",
                        icon=":material/download:",
                        on_click="ignore",
                        width="stretch",
                    )

        st.button("Завоз и обратный груз — сравнить рейсы", key="open_backhaul",
                  on_click=open_backhaul, width="stretch")
        st.button("Построить маршрут и проверить условия доставки",key="open_routes",
                  on_click=open_routes,width="stretch")
        st.caption("В дашборде — карта путей, прогнозы, планировщик отправок и рейтинг риска. "
                   "В PDF — подробное описание проекта, методика и результаты.")
        st.divider()

        data, forecast, plan = st.columns(3, gap="large")
        with data:
            st.markdown("**01 · Открытые данные**")
            st.write("Погода, замеры толщины льда и спутниковые наблюдения. Без установки датчиков.")
        with forecast:
            st.markdown("**02 · Прогноз окна**")
            st.write("Диапазон возможных дат открытия и закрытия пути с учётом неопределённости.")
        with plan:
            st.markdown("**03 · План поставок**")
            st.write("График отправок по неделям, чтобы оценить риск недовоза и потребность в технике.")
