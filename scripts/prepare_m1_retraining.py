"""Audit whether the new official data can supervise opening/closing M1.

This command deliberately does not overwrite model weights. A preparation
report is always saved; exit 2 means there are no new valid event targets.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from okno import official_hydro as H
from okno.hydro_training import extract_dated_ice, prepare_targets
from okno.hydro_ingest import merge_records, _write_snapshot, update_lock
from okno.paths import REPORTS
from okno.report import md_table


def main():
    now = datetime.now(timezone.utc)
    with update_lock(H.DIRECTORY) as locked:
        if not locked:
            raise SystemExit("Дождитесь завершения обновления официальных данных")
        snapshot = H.load_snapshot()
        if not snapshot:
            raise SystemExit("Нет снимка официальных данных")
        snapshot = dict(snapshot)
        derived = [r for report in snapshot.get("bulletins", []) for r in extract_dated_ice(report, now)]
        snapshot["observations"] = merge_records(snapshot.get("observations", []), derived)
        # Preparation from already downloaded, hash-verified originals;
        # checked_at/last_success_at retain the actual last network check.
        _write_snapshot(H.DIRECTORY, snapshot)
    candidates, targets = prepare_targets(snapshot)
    REPORTS.mkdir(exist_ok=True)
    columns=["point_id","hydro_year","measured_dates","min_ice_cm","max_ice_cm","eligible","reason"]
    candidate_frame=pd.DataFrame(candidates,columns=columns)
    candidate_frame.to_csv(REPORTS/"m1_official_training_candidates.csv",index=False)
    counts=Counter((r["metric"],r["quality"]) for r in snapshot["observations"])
    dated=[r for r in snapshot["observations"] if r["metric"]=="ice_cm" and r["quality"]=="measured"]
    data=pd.DataFrame([dict(Точка=f"{r['river']} — {r['station_name']}",Дата=H.stamp(r["observed_at"]).date().isoformat(),
                            **{"Лёд, см":r["value"],"Тип точки":"гидропост" if r['scope']=='post' else 'опорный пост',
                               "Подтверждение":r["evidence"]}) for r in dated])
    status=dict(checked_at=now.isoformat(),status="eligible_targets_found" if targets else "blocked_missing_event_targets",
                measured_water_levels=counts[("water_level_cm","measured")],
                undated_ice=counts[("ice_cm","undated")],dated_ice=len(dated),
                new_opening_targets=len(targets),new_closing_targets=0,weights_overwritten=False,
                candidates=candidates,targets=targets)
    (REPORTS/"m1_retraining_readiness.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    lines=["# Подготовка переобучения М1 на новых официальных данных","",
           f"Техническая метка подготовки (UTC): {now.isoformat()}. Статус: **{status['status']}**.","",
           f"Измерений уровня воды: {status['measured_water_levels']}. Они не являются целями модели дат ледовых событий. "
           f"Значений льда без даты: {status['undated_ice']}; в обучение не допускаются. "
           f"В текстах сводок найдены {len(dated)} точных датированных значения толщины льда.","",
           md_table(data),"","Источник трёх значений — официальный обзор Якутского УГМС за 29 декабря 2022 года "
           "(https://ykuthydromet.ru/reki/); исходный HTML и SHA-256 сохранены в снимке данных. "
           "Значение Якутска относится к городскому гидропосту, а не к обследованию переправы.","",
           md_table(candidate_frame),"",
           f"Новых пригодных целей открытия: **{len(targets)}**. Новых подтверждённых целей начала таяния: **0**.","",
           "Для открытия используется прежнее определение цели: достижение порога толщины для 20 т "
           "по серии минимум из трёх замеров с переходом через порог. Одиночное значение не определяет дату перехода. "
           "Для закрытия нужны датированные ледовые фазы конкретного поста. Дата публикации и первая встреча слова "
           "«ледоход» не заменяют начало таяния. Региональные диапазоны не превращены в точечные измерения.","",
           "Полный уже существующий архив содержит замеры до 2018 года; они уже участвовали в обучении "
           "обновлённых моделей для новых постов. Повторное обучение на тех же таблицах нельзя представлять "
           "как обучение на наблюдениях Росгидромета 2025–2026 годов.","",
           "**Веса основной М1 не перезаписаны.** При отсутствии новых целевых дат запуск прежнего тренера "
           "не выполнит запрос на переобучение по новым сведениям. Требуются новые последовательные замеры "
           "толщины льда либо подтверждённые даты соответствующей ледовой фазы."]
    (REPORTS/"m1_retraining_readiness.md").write_text('\n'.join(lines)+'\n')
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if not targets:
        raise SystemExit(2)


if __name__=="__main__":
    main()
