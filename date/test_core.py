import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
# nationwide_weather imports streamlit, so these tests run in the deployed dependency environment.
from v15_features import climate_report, forecast_risk, closest_region, evaluate_alerts
from nationwide_weather import REGIONS, parse_rain
class TestCore(unittest.TestCase):
    def test_regions(self):
        self.assertGreaterEqual(len(REGIONS),70)
        self.assertEqual(closest_region(35.8714,128.6014,REGIONS)[0],'대구')
    def test_rain(self):
        self.assertEqual(parse_rain('강수없음'),0.0)
        self.assertTrue(np.isnan(parse_rain('1mm 미만')))
        self.assertTrue(np.isnan(parse_rain('1~4mm')))
    def test_stale_risk(self):
        f=pd.DataFrame({'POP':[99],'PCP':['25mm']},index=pd.DatetimeIndex(['2020-01-01']))
        self.assertEqual(forecast_risk(f)['level'],0)
        self.assertEqual(forecast_risk(f)['hours'],0)
    def test_climate_index_collision(self):
        dates=pd.date_range('2018-01-01','2022-12-31',freq='D')
        f=pd.DataFrame({'date':dates,'rain':np.zeros(len(dates)),'tmax':np.ones(len(dates))*20,'tmin':np.ones(len(dates))*10}).set_index('date',drop=False)
        summary,annual=climate_report(f)
        self.assertEqual(len(annual),5)
        self.assertFalse(summary.empty)
    def test_alert_future(self):
        future=pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).ceil('h')
        f=pd.DataFrame({'POP':[80],'TMP':[30]},index=pd.DatetimeIndex([future]))
        self.assertTrue(any('강수확률' in x for x in evaluate_alerts(f,70,35,-5)))
if __name__=='__main__':unittest.main()
