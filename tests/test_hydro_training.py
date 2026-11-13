"""Training eligibility must not turn levels, ranges or singletons into dates."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from okno import official_hydro as H
from okno.hydro_training import extract_dated_ice, prepare_targets
from scripts import prepare_m1_retraining as command

NOW=datetime(2026,11,13,tzinfo=timezone.utc)
REPORT=dict(source_url='https://ykuthydromet.ru/reki/',raw_sha256='a'*64,
            published_at='2022-12-29T23:59:59+00:00',
            text='По данным измерений от 29 декабря на р. Лена толщина льда составляет 41 – 95 см. '
                 'У г. Якутск толщина льда составляет 47 см. '
                 'На р. Яна у г. Верхоянск толщина льда составляет 84 см. '
                 'На р. Колыма у п. Зырянка толщина льда составляет 81 см.')


def empty_history():
    return pd.DataFrame(columns=['gauge_code','hydro_year','date','ice_cm'])


class ExactIceExtractionTests(unittest.TestCase):
    def test_absolute_point_measurements_keep_actual_survey_date_and_scope(self):
        rows=extract_dated_ice(REPORT,NOW)
        self.assertEqual({r['point_id']:r['value'] for r in rows},
                         {'yakutsk_crossing':47,'gauge_3414':84,'gauge_1018':81})
        self.assertTrue(all(r['observed_at']=='2022-12-29T00:00:00+00:00' for r in rows))
        self.assertTrue(all(r['quality']=='measured' and r['evidence'] for r in rows))
        reference=next(r for r in rows if r['point_id']=='yakutsk_crossing')
        self.assertEqual(reference['scope'],'reference_post')

    def test_river_range_and_anomaly_are_not_absolute_point_measurements(self):
        for text in [
            'По данным измерений от 29 декабря на р. Яна у г. Верхоянск толщина льда составляет 70–84 см.',
            'По данным измерений от 29 декабря на р. Яна у г. Верхоянск толщина льда выше нормы на 84 см.',
            'По данным измерений от 29 декабря на реках Яна и Колыма толщина льда составляет 84 см.',
        ]:
            self.assertEqual(extract_dated_ice({**REPORT,'text':text},NOW),[])

    def test_publication_day_does_not_replace_an_absent_or_future_survey_date(self):
        missing={**REPORT,'text':REPORT['text'].replace('По данным измерений от 29 декабря','По данным УГМС')}
        self.assertEqual(extract_dated_ice(missing,NOW),[])
        future={**REPORT,'text':REPORT['text'].replace('29 декабря','30 декабря')}
        self.assertEqual(extract_dated_ice(future,NOW),[])

    def test_old_december_survey_published_in_january_keeps_correct_year(self):
        rows=extract_dated_ice({**REPORT,'published_at':'2023-01-02T12:00:00+00:00'},NOW)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r['observed_at'].startswith('2022-12-29') for r in rows))


class EventTargetEligibilityTests(unittest.TestCase):
    def test_single_measurements_do_not_become_opening_dates(self):
        rows=extract_dated_ice(REPORT,NOW)
        candidates,targets=prepare_targets({'observations':rows},historical=empty_history())
        self.assertEqual(len(candidates),2)  # exclude the nearby city reference post
        self.assertEqual(targets,[])
        self.assertTrue(all(r['measured_dates']==1 and not r['eligible'] for r in candidates))

    def test_levels_and_undated_ice_do_not_create_training_targets(self):
        rows=extract_dated_ice(REPORT,NOW)
        levels=[{**r,'metric':'water_level_cm'} for r in rows]
        undated=[{**r,'quality':'undated','observed_at':None} for r in rows]
        self.assertEqual(prepare_targets({'observations':levels+undated},historical=empty_history()),([],[]))

    def test_real_bracketed_series_produces_only_a_physical_proxy_target(self):
        row=next(r for r in extract_dated_ice(REPORT,NOW) if r['point_id']=='gauge_3414')
        series=[{**row,'observed_at':d,'value':v} for d,v in
                [('2022-11-01T00:00:00Z',20),('2022-11-10T00:00:00Z',40),('2022-11-20T00:00:00Z',65)]]
        candidates,targets=prepare_targets({'observations':series},historical=empty_history())
        self.assertTrue(candidates[0]['eligible'])
        self.assertEqual(len(targets),1)
        self.assertEqual(targets[0]['target_definition'],'interpolated_20t_thickness_crossing')
        self.assertGreater(targets[0]['target_date'],'2022-11-10')
        self.assertLess(targets[0]['target_date'],'2022-11-20')

    def test_conflicting_values_on_the_same_date_are_not_averaged(self):
        row=next(r for r in extract_dated_ice(REPORT,NOW) if r['point_id']=='gauge_3414')
        series=[{**row,'observed_at':d,'value':v} for d,v in
                [('2022-11-01T00:00:00Z',20),('2022-11-10T00:00:00Z',40),
                 ('2022-11-10T00:00:00Z',60),('2022-11-20T00:00:00Z',65)]]
        candidates,targets=prepare_targets({'observations':series},historical=empty_history())
        self.assertEqual(targets,[])
        self.assertIn('Противоречивые',candidates[0]['reason'])

    def test_preparation_records_a_block_instead_of_pretending_to_train(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'official';directory.mkdir()
            reports=Path(root)/'reports'
            snapshot=directory/'snapshot.json'
            snapshot.write_text(json.dumps(dict(schema_version=1,observations=[],bulletins=[REPORT])))
            with patch.object(H,'DIRECTORY',directory),patch.object(H,'SNAPSHOT',snapshot), \
                 patch.object(command,'REPORTS',reports):
                with self.assertRaises(SystemExit) as result:
                    command.main()
            self.assertEqual(result.exception.code,2)
            saved=json.loads((reports/'m1_retraining_readiness.json').read_text())
            self.assertFalse(saved['weights_overwritten'])
            self.assertEqual(saved['new_opening_targets'],0)
            self.assertEqual(saved['new_closing_targets'],0)
            self.assertEqual(saved['status'],'blocked_missing_event_targets')
            self.assertFalse((directory/'.update.lock').exists())


if __name__=='__main__':
    unittest.main()
