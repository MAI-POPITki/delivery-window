import streamlit as st
from okno.flood_day import forecast_day,efficiency


def render_day(pid,issue,target):
    try:result=forecast_day(pid,issue,target)
    except (ValueError,OSError,KeyError) as exc:
        st.warning(f'Вероятность на выбранную дату не рассчитана: {exc}');return
    st.subheader(f"Риск на {target:%d.%m.%Y}")
    a,b=st.columns([1,2])
    value=(f"{result['probability']:.0%}–{result['probability_upper']:.0%}" if result['threshold']!=result['lower_threshold']
           else f"{result['probability']:.1%}")
    a.metric('Вероятность разлива' if result['physical'] else 'Вероятность высокой воды',value)
    b.markdown('**'+result['label']+'**')
    b.write(f"На выбранные сутки, через {result['lead']} сут. после даты исходных измерений {issue:%d.%m.%Y}.")
    b.caption(result['note'])
    if not result['physical']:st.warning('Подтверждённого порога разлива для этого поста нет: высокая вода показана как отдельный индикатор.')
    st.caption('Модель оценивает превышение уровня на гидропосту в выбранные сутки. '
               'Это не вероятность появления воды в конкретном доме или на дороге. '
               'Диапазон Орска отражает два порога, а не доверительный интервал.')
    with st.expander('Эффективность: прогноз против реальных измерений',expanded=False):
        score=efficiency(pid,result['lead'])
        if score:
            st.write(f"**{score['n']} архивных прогнозов**, {score['events']} фактических превышений. "
                     f"Период: {score['first']} — {score['last']}.")
            st.table({'Метод':['Наша вероятность','Сохранить текущий уровень с ошибками','Историческая частота'],
                      'Brier score ↓':[round(score[k],4) for k in ['brier','persistence_brier','climatology_brier']]})
            st.write(f"Выигрыш относительно исторической частоты: **{score['skill_climatology']:+.1%}**; "
                     f"относительно сохранения уровня: **{score['skill_persistence']:+.1%}**.")
            st.caption('Brier score — средний квадрат ошибки вероятности: меньше лучше. '
                       'Положительный выигрыш означает меньшую ошибку; это не процент точности.')
            st.write(f"При предупреждении от 50%: найдено событий **{score['tp']}**, пропущено **{score['fn']}**, "
                     f"ложных предупреждений **{score['fp']}**, верных отсутствий **{score['tn']}**.")
            if score['brier']>=min(score['persistence_brier'],score['climatology_brier']):
                st.warning('Преимущество над обоими простыми правилами для этого события не получено.')
            st.caption('Повторная архивная проверка, соседние дни зависимы. Порог из справочника 2026 применяется '
                       'ретроспективно; это не независимая проверка фактических затоплений. '
                       'Для Орска таблица относится к верхней отметке 450 см.')
        else:st.info('Для этого поста и горизонта нет сохранённой проверки.')
    with st.expander('На чём строится процент и источник порога'):
        st.write('М1 v2 прогнозирует уровень по прошлым ежедневным измерениям, изменению уровня за 1–30 суток '
                 'и сезону. Вероятность рассчитывается по распределению ошибок предыдущего года '
                 'относительно выбранного порога. Будущие измерения в прогноз не входят. '
                 'Спутник показывает наблюдаемую воду и не подменяет прогноз вероятности.')
        st.markdown(f"[Источник определения события]({result['source']})")
        st.caption(f"Проверяемый уровень: {result['threshold']:.0f} см над нулём этого гидропоста.")
    return result
