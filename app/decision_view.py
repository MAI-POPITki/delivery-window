"""Видимое сравнение решений без текстовой генерации и вымышленных фактов."""
import math
import streamlit as st
from okno.decision_support import start_options, start_sensitivity
from okno.ice import from_season_day


def render_decisions(forecast, issue_day, p, hydro_year,availability=None,handling_capacity=None):
    st.markdown('### Что изменить, чтобы успеть?')
    st.caption('Расчёт М3 для выбранного участка. Объём и мощности берутся из параметров ниже; это ввод пользователя, а не подтверждённые данные перевозчика. Разрешение на проезд проверяется отдельно.')
    floor = math.ceil(max(issue_day, p.start_day))
    a, b = st.columns(2)
    delay = a.number_input('Через сколько дней планируете начать отправки?', 0, 90, 7, key='decision_delay')
    advance = b.number_input('На сколько дней реально можете приблизить начало?', 0, int(delay), min(7, int(delay)), key='decision_advance')
    planned, early = floor+delay, floor+delay-advance
    st.caption(f'Отсчёт от {from_season_day(floor, hydro_year):%d.%m.%Y}: не раньше даты решения и начала сезона. Более ранний проезд допустим только после подтверждения открытия пути.')
    table = start_options(forecast, issue_day, p, planned, early,availability=availability,handling_capacity=handling_capacity)
    view = table[['start_day', 'planned_road_t', 'expected_backup_t', 'expected_cost_rub']].copy()
    view['start_day'] = view.start_day.map(lambda d: from_season_day(d, hydro_year).date())
    view.columns = ['Начало отправок', 'План по зимнику, т', 'Ожидаемый остаток для резерва, т', 'Ожидаемые расходы, ₽']
    st.dataframe(view, hide_index=True, width='stretch')
    st.caption('Одинаковый прогноз и тарифы во всех вариантах. Остаток для резерва не означает доставку к сроку: расписание резервного канала здесь не моделируется. Масса машины не меняется автоматически: для этого нужны число машин и время их оборота.')
    st.markdown('**Насколько устойчив результат?**')
    spread = st.slider('Изменение неизвестных тарифов, ±%', 0, 100, 30, key='decision_spread')
    sensitivity = start_sensitivity(forecast, issue_day, p, planned, early, spread/100,availability=availability,handling_capacity=handling_capacity)
    low, high = -sensitivity.saving_rub.max(), -sensitivity.saving_rub.min()
    st.write(f'Изменение ожидаемых расходов при самом раннем старте: от {low:+,.0f} до {high:+,.0f} ₽ относительно исходного начала.')
    st.caption('Минус — расходы снижаются; плюс — растут. Диапазон относится только к проверенным вариантам тарифов.')
    if early == planned:
        st.info('Начало не изменено: варианты совпадают. Чтобы сравнить решения, укажите доступное приближение старта.')
    elif low < -0.01 and high > 0.01:
        st.warning('Вывод о выгоде меняется при изменении тарифов: ранний старт не всегда дешевле.')
    st.caption('Хранение, резервный канал и надбавка за технику меняются по одному; каждый план пересчитывается. Это проверка допущений, не вероятность успеха и не доказанная экономия оператора.')
    sensitivity['parameter'] = sensitivity.parameter.map({'c_hold_week':'Хранение','c_exp':'Резервный канал','c_extra':'Дополнительная техника'})
    sensitivity = sensitivity.rename(columns={'parameter':'Изменяемый тариф', 'factor':'Множитель тарифа', 'saving_rub':'Экономия раннего старта, ₽', 'backup_reduction_t':'Сокращение остатка для резерва, т'})
    st.download_button('Скачать проверку устойчивости', sensitivity.to_csv(index=False).encode('utf-8-sig'), 'decision-sensitivity.csv', key='decision_export')
    st.info('Где ограничение всей доставки? В этой вкладке доступна историческая проверка сквозного коридора А3: ж/д → переправа → склад. Вывод по зимнику нельзя переносить на всю цепочку.')
