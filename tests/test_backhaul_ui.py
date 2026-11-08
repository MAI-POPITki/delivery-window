"""Пользовательский переход, расчёт, устаревший результат и возврат к проекту."""
from pathlib import Path
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


class BackhaulUITests(unittest.TestCase):
    def test_entry_calculation_edit_and_return_without_forecast(self):
        with patch('okno.forecast.data_version',side_effect=AssertionError('М1 не нужен для обратного груза')):
            app=AppTest.from_file(str(ROOT/'app/dashboard.py'),default_timeout=60).run()
            app.button(key='open_backhaul').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value,'Завоз и обратный груз')
            app.button(key='bh_calculate').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric),3)
            self.assertEqual(len(app.get('plotly_chart')),2)
            self.assertTrue(any('Выполнены все заявки (2)' in x.value for x in app.success))
            self.assertEqual(len(app.download_button),5)
            app.number_input(key='bh_travel').set_value(9.).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric),0)
            self.assertTrue(any('Параметры изменены' in x.value for x in app.info))
            app.button(key='bh_calculate').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric),3)
            app.button(key='backhaul_return').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value,'Окно завоза')

    def test_direct_link_and_invalid_window(self):
        app=AppTest.from_file(str(ROOT/'app/dashboard.py'),default_timeout=60)
        app.query_params['module']='backhaul'
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value,'Завоз и обратный груз')
        app.date_input(key='bh_close').set_value(app.date_input(key='bh_date').value).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Закрытие должно быть позже' in x.value for x in app.error))
        self.assertEqual(len(app.metric),0)


if __name__=='__main__': unittest.main()
