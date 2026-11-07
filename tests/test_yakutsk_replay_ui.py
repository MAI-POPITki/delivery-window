from pathlib import Path
import datetime as dt
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class ReplayDashboardTests(unittest.TestCase):
    def open_replay(self):
        app = AppTest.from_file(str(ROOT/'app/dashboard.py'), default_timeout=120).run()
        app.button(key='open_dashboard').click().run()
        app.sidebar.button(key='activate_replay').click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        return app

    def summary(self, app):
        return next(d.value for d in app.dataframe if 'Расходы, ₽' in d.value.columns)

    def test_preset_selects_the_right_context_and_recalculates(self):
        app=self.open_replay()
        self.assertEqual(app.sidebar.selectbox(key='filter_region').value, 'yakutia')
        self.assertEqual(app.sidebar.selectbox(key='filter_season').value, 2026)
        self.assertEqual(app.sidebar.selectbox(key='filter_point').value, 'yakutsk_crossing')
        self.assertEqual(app.sidebar.date_input(key='filter_date').value, dt.date(2026,3,16))
        self.assertEqual(app.number_input(key='replay_season_total').value, 400000.)
        self.assertEqual(app.number_input(key='replay_Q').value, 5000.)
        self.assertEqual(app.number_input(key='replay_base').value, 10)
        self.assertEqual(app.number_input(key='replay_extra').value, 5)
        self.assertEqual(app.number_input(key='replay_tare').value, 12.)
        frame=self.summary(app)
        self.assertEqual(len(frame),8)
        self.assertAlmostEqual(frame.loc['Календарь с пересчётом','По зимней дороге, т'],5000.)
        self.assertGreater(frame.loc['М1 + спутник + планировщик','Расходы, ₽'],
                           frame.loc['Календарь с пересчётом','Расходы, ₽'])
        app.number_input(key='replay_Q').set_value(6500.).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        frame=self.summary(app)
        self.assertGreater(frame.loc['М1 + спутник + планировщик','По зимней дороге, т'],
                           frame.loc['Календарь с пересчётом','По зимней дороге, т'])
        self.assertTrue(app.download_button(key='replay_download_schedule'))
        self.assertTrue(app.download_button(key='replay_download_report'))

    def test_season_total_is_separate_and_large_remainders_and_fleets_are_supported(self):
        app=self.open_replay()
        self.assertEqual(app.number_input(key='replay_season_total').value, 400000.)
        self.assertEqual(app.number_input(key='replay_Q').value, 5000.)
        fleet=next(d.value for d in app.dataframe if 'Одна машина за период, т' in d.value.columns)
        self.assertEqual(fleet['Одна машина за период, т'].tolist(), [444.,516.])
        self.assertEqual(fleet['Нужно машин без найма дополнительного парка'].tolist(), [901,776])
        app.number_input(key='replay_Q').set_value(400000.)
        app.number_input(key='replay_extra').set_value(0)
        app.run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        frame=self.summary(app)
        for _, row in frame.iterrows():
            self.assertAlmostEqual(row['По зимней дороге, т'], 4440.)
            self.assertAlmostEqual(row['По зимней дороге, т']+row['Резерв, т'], 400000.)
        self.assertTrue(app.warning)
        app.number_input(key='replay_base').set_value(1000).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertAlmostEqual(self.summary(app).loc['М1 + спутник + планировщик','По зимней дороге, т'],400000.)
        self.assertEqual(app.number_input(key='replay_season_total').value,400000.)

    def test_can_leave_replay_and_change_region_and_season(self):
        app=self.open_replay()
        app.sidebar.button(key='leave_replay').click().run()
        app.sidebar.selectbox(key='filter_region').select('irkutsk').run()
        app.sidebar.selectbox(key='filter_season').select(2027).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.session_state['replay_active'])
        self.assertEqual(app.sidebar.selectbox(key='filter_region').value, 'irkutsk')
        self.assertTrue(any(w.label=='Осталось вывезти на дату решения, т' for w in app.number_input))


if __name__=='__main__':
    unittest.main()
