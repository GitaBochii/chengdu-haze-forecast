import unittest
import numpy as np
import pandas as pd
import torch
from experiments.prithvi_wxc_probe_20260924.pilot_training_data import prepare_pollution_histories


class TrainingDataTests(unittest.TestCase):
    def make_data(self):
        times = pd.date_range('2020-01-01', periods=80, freq='h', tz='Asia/Shanghai')
        rows = []
        for i, timestamp in enumerate(times):
            for node in range(3):
                rows.append(dict(timestamp=timestamp, node_id=str(node), longitude=104+node*.1,
                                 latitude=30+node*.1, pm2_5=20+i, pm10=40+i, aqi=60+i,
                                 so2=3., no2=5., co=.6, o3_8h=7.))
        stations = pd.DataFrame(rows)
        weather = pd.DataFrame({'timestamp':times,'temperature_c':20.,'relative_humidity_pct':50.,
                                'wind_u_ms':3.,'wind_v_ms':4.})
        candidates = [{'timestamp':times[i].isoformat(),'partition':p} for i,p in
                      [(23,'train'),(24,'train'),(70,'dev')]]
        return stations, weather, candidates

    def test_dev_values_do_not_affect_fit_and_train_hours_unique(self):
        stations, weather, candidates = self.make_data()
        first = prepare_pollution_histories(stations,weather,candidates)
        self.assertEqual(len(first['training_timestamps']),25)
        changed = stations.copy()
        changed.loc[changed.timestamp.ge(pd.Timestamp(candidates[-1]['timestamp'])-pd.Timedelta(hours=23)), 'pm2_5'] = 1e9
        second = prepare_pollution_histories(changed,weather,candidates)
        np.testing.assert_array_equal(first['scaler_mean'],second['scaler_mean'])
        np.testing.assert_array_equal(first['scaler_scale'],second['scaler_scale'])
        torch.testing.assert_close(first['x'][:2],second['x'][:2],rtol=0,atol=0)
        self.assertAlmostEqual(first['scaler_mean'][0],32.)

    def test_training_only_nodes_order_and_missing_mask(self):
        stations, weather, candidates = self.make_data()
        stations.loc[stations.node_id.eq('2') & stations.timestamp.le(pd.Timestamp(candidates[1]['timestamp'])), 'pm2_5'] = np.nan
        stations.loc[stations.node_id.eq('0') & stations.timestamp.eq(pd.Timestamp(candidates[0]['timestamp'])), 'pm10'] = np.nan
        result = prepare_pollution_histories(stations,weather,candidates)
        self.assertEqual(result['nodes'],['0','1'])
        self.assertEqual(result['x'].shape,(3,24,2,17))
        self.assertEqual(result['active'].shape,(3,2,2))
        self.assertEqual(result['x'].dtype,torch.float32)
        self.assertTrue(result['raw_missing_mask'][0,-1,0,1])
        self.assertTrue(torch.isfinite(result['x']).all())
        self.assertEqual(result['feature_names'][:3],['pm2_5','pm10','aqi'])
        self.assertEqual(result['feature_names'][-2:],['wind_speed_kmh','wind_from_deg'])
        self.assertAlmostEqual(result['scaler_mean'][-2],18.)
        reversed_rows = prepare_pollution_histories(stations,weather,list(reversed(candidates)))
        torch.testing.assert_close(reversed_rows['x'][0],result['x'][-1])


if __name__ == '__main__':
    unittest.main()
