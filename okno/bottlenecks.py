"""Диагностика ограничений цепочки через контролируемые изменения мощности."""
from dataclasses import replace
import pandas as pd
from okno.corridor import simulate_corridor


def diagnose_corridor(cfg, case, fraction=.1):
    if not 0 < fraction <= 1:
        raise ValueError('Изменение мощности должно быть в пределах (0, 1]')
    def run(config):
        return simulate_corridor(config, case['ac'], case['ao'], case['cc'], case['co'])
    base = run(cfg)
    rows = []
    for field, label in [('rail_cap_tons','Отправка по ж/д'), ('crossing_cap_normal_tons','Штатная переправа'), ('crossing_cap_alt_tons','Резервная переправа')]:
        value = getattr(cfg, field)
        result = run(replace(cfg, **{field:value*(1+fraction)}))
        rows.append(dict(Участок=label, Мощность=value, Проверяемая_мощность=value*(1+fraction),
                         Сокращение_ожидания_т_сут=base['shortage_ton_days']-result['shortage_ton_days'],
                         Сокращение_остатка_т=base['terminal_backlog_tons']-result['terminal_backlog_tons']))
    return pd.DataFrame(rows)
