"""Воспроизводимые модельные примеры; не оценка экономии реального перевозчика.

Запуск из корня: python -m scripts.verify_backhaul
"""
from dataclasses import replace
from pathlib import Path
import pandas as pd
from okno.backhaul import Order,Vehicle,Route,compare,audit_plan
from okno.backhaul_io import example_frames,parse_frames


def run():
    orders,vehicles=parse_frames(*example_frames('2026-03-01'),'2026-03-01')
    route=Route(300,8,0,96,20)
    scenarios=[('Учебный пример интерфейса',orders,vehicles,route)]
    small=[Order('Завоз','out',5,'тент',0,8,0,0),Order('Вывоз','back',4,'тент',1,8,0,0)]
    fleet=[Vehicle('М1','тент',10,8,1,1,0)]
    road=Route(100,1,0,8,20,1,0)
    scenarios.extend([
        ('Контроль 4L → 2L',small,fleet,road),
        ('Нет обратного заказа',small[:1],fleet,road),
        ('Дорогое ожидание', [replace(small[0],due_h=2),replace(small[1],ready_h=20,due_h=24)],
         [replace(fleet[0],occupied_rub_h=100)],replace(road,close_h=24)),
        ('Разные кузова',[small[0],replace(small[1],body='рефрижератор')],
         fleet+[replace(fleet[0],id='М2',body='рефрижератор')],road),
        ('В окно помещается только совмещение',small,fleet,replace(road,close_h=2)),
        ('Не успевает вернуться',small[:1],fleet,replace(road,close_h=1.5)),
    ])
    rows=[]
    for name,jobs,cars,path in scenarios:
        plans=compare(jobs,cars,path)
        if all(p.status=='optimal' for p in plans.values()):
            assert plans['combined'].summary(jobs)['cost_rub']<=plans['separate'].summary(jobs)['cost_rub']+1e-5
        for key,p in plans.items():
            if p.feasible:
                audit_plan(p.trips,jobs,cars,path)
            rows.append(dict(пример=name,схема=key,статус=p.status,**(p.summary(jobs) or {})))
    table=pd.DataFrame(rows)
    root=Path(__file__).resolve().parents[1]/'reports'
    root.mkdir(exist_ok=True)
    table.to_csv(root/'backhaul_validation.csv',index=False,encoding='utf-8-sig')
    lines=['# Проверка завоза с обратной загрузкой','',
           'Все примеры ниже искусственные. Они проверяют математическую работоспособность, '
           'но не подтверждают наличие обратного груза или экономию в Якутии.','',
           'Протокол: docs/BACKHAUL_VALIDATION.md. Повторить: `python -m scripts.verify_backhaul`.','',
           '| Пример | Схема | Статус | Общий пробег, км | Пустой, км | Расходы, ₽ |',
           '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        number=lambda k:f"{row[k]:.1f}" if k in row else 'не рассчитывается'
        scheme='Отдельные' if row['схема']=='separate' else 'С совмещением'
        status={'optimal':'Оптимум на сетке','infeasible':'Невыполнимо','feasible':'Допустимый план','unknown':'Не определено'}[row['статус']]
        lines.append(f"| {row['пример']} | {scheme} | {status} | {number('total_km')} | {number('empty_km')} | {number('cost_rub')} |")
    lines+=['','Выполнимые планы прошли проверку массы, кузова, сроков, полного возврата, '
            'занятости машин, расчёта расходов и выполнения каждой заявки ровно один раз.',
            '', 'Объединение может уменьшать пробег, но не обязано давать экономию. '
            'При дорогом ожидании или несовместимости кузовов остаются отдельные рейсы. '
            'Относительно невыполнимой базовой схемы процент экономии не рассчитывается.',
            '', 'Для практической проверки нужны реальные пары заявок, фактические расходы и '
            'подтверждённое окно проезда. Базовую практику оператора следует установить по журналу рейсов.']
    (root/'backhaul_validation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Проверено сценариев:',len(scenarios))
    print(table[['пример','схема','статус','total_km','empty_km','cost_rub']].to_string(index=False))


if __name__=='__main__': run()
