import unittest
import run_experiment as mod
import numpy as np
import pandas as pd

class ProtocolTests(unittest.TestCase):
    def test_recovery_censor_rebound_and_short_clean(self):
        a=np.full((4,24),150.)
        a[1,2:8]=80; a[2,:5]=80; a[3,18:24]=80
        np.testing.assert_array_equal(mod.recovery(a),[20,3,20,19])

    def test_eligibility_boundary_missing_and_lag(self):
        ts=pd.date_range('2019-12-28',periods=240,freq='h',tz='Asia/Shanghai')
        f=pd.DataFrame({'timestamp':ts,'pm2_5':10.,'pm10':20.,'aqi':50.})
        f.loc[50,'aqi']=np.nan
        w,e=mod.prepare(f)
        self.assertNotIn(50,w['indices'])
        for i,h in zip(w['indices'],w['history']):
            np.testing.assert_array_equal(h,f.loc[i-24:i-1,['pm2_5','pm10','aqi']])
            self.assertEqual(ts[i-24].year<=2019,ts[i+24].year<=2019)
        self.assertTrue((e.reason=='partition_boundary').any())

    def test_groups_rebound_missing_partition(self):
        ts=pd.date_range('2019-12-31 20:00',periods=18,freq='h',tz='Asia/Shanghai')
        a=np.array([150,80,80,150,80,80,80,80,80,80,np.nan,80,80,80,80,80,80,80.])
        g=mod.groups(ts,a,np.isfinite(a))
        self.assertEqual(g[0],g[3]); self.assertNotEqual(g[3],g[4]); self.assertNotEqual(g[9],g[11])

    def test_greedy_ties_and_insufficient(self):
        z=np.zeros((40,76)); times=np.arange(40,dtype=np.int64)*49*3600*10**9
        g=np.array([str(i) for i in range(40)])
        n,s=mod.neighbors(np.zeros(76),z,times,g)
        np.testing.assert_array_equal(n,np.arange(32))
        g[:]='same'; n,s=mod.neighbors(np.zeros(76),z,times,g)
        self.assertEqual(len(n),0); self.assertTrue(any(r[1]=='same_group' for r in s))

    def test_49_hour_spacing_and_future_not_features(self):
        z=np.zeros((100,76)); times=np.arange(100,dtype=np.int64)*24*3600*10**9
        n,s=mod.neighbors(np.zeros(76),z,times,np.array([str(i) for i in range(100)]))
        self.assertEqual(len(n),32); self.assertTrue((np.diff(times[n])>=49*3600*10**9).all())
        ts=pd.date_range('2018-01-01',periods=100,freq='h',tz='Asia/Shanghai')
        f=pd.DataFrame({'timestamp':ts,'pm2_5':10.,'pm10':20.,'aqi':50.})
        w,_=mod.prepare(f); f.loc[50:74,['pm2_5','pm10','aqi']]=150
        v,_=mod.prepare(f)
        np.testing.assert_array_equal(w['features'][w['indices']==50],v['features'][v['indices']==50])

    def test_censor_mass_and_all_reject(self):
        p,c=mod.distribution(np.array([1]*16+[20]*16))
        self.assertEqual(p[-1],.5); self.assertEqual(p.sum(),1)
        self.assertTrue((np.diff(c)>=0).all()); self.assertEqual(mod.decision(c[None],1.000001)[0],20)

    def test_fpr_denominator_and_reject_failure(self):
        m=mod.metrics(np.array([1,20,20,20]),np.array([1,1,20,20]),np.array(['a','b','c','d']))
        self.assertAlmostEqual(m['fpr'],1/3)
        r=dict(theta=1.000001,**mod.metrics(np.array([1,20]),np.array([20,20]),np.array(['a','b'])))
        self.assertIsNone(mod.select([r],1))

    def test_equal_group_rates_disclose_denominators(self):
        m=mod.metrics(np.array([1,1,20]),np.array([1,20,1]),np.array(['a','a','b']))
        self.assertEqual(m['group_equal']['recall']['valid_groups'],1)
        self.assertEqual(m['group_equal']['recall']['value'],.5)
        self.assertEqual(m['group_equal']['fpr']['valid_groups'],1)
        self.assertEqual(m['group_equal']['fpr']['value'],1)

if __name__=='__main__': unittest.main()
