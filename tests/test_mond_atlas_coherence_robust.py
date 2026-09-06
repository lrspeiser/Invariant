import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_mond_atlas_coherence_robust import predict,fit

def source():
    r=np.geomspace(.1,50,40)
    return dict(r=r,gas=r*0+10,disk=r*0+50,bulge=r*0+5,sb=np.geomspace(.1,1e4,40))

class CoherenceTests(unittest.TestCase):
    def test_limits_and_independent_transfer(self):
        s=source();base=predict(s,'newton',[1.])
        np.testing.assert_allclose(predict(s,'coherence_n1',[1.,0,np.log(10)]),base,atol=1e-10)
        np.testing.assert_allclose(predict(s,'relay_active',[1.,3,1]),base,atol=1e-10)
        sigma=.5*s['sb'];tau=.3*sigma/100
        expected=base+.5*np.log10(np.exp(-tau)+4*(1-np.exp(-tau)))
        np.testing.assert_allclose(predict(s,'relay_active',[1.,.3,4]),expected,atol=1e-10)
        self.assertTrue((predict(s,'relay_passive',[1.,10,0])<=base).all())
        self.assertTrue(np.isfinite(predict(s,'relay_passive',[1.,10,0])).all())

    def test_planted_recovery_and_training_isolation(self):
        s=source();y=predict(s,'coherence_n1',[1.2,3,np.log(20)])
        gi=np.repeat(np.arange(4),10);train=gi!=3
        best,attempts=fit(s,y,gi,train,'coherence_n1')
        self.assertLess(best['training_mse'],1e-12)
        altered=y.copy();altered[~train]=np.nan
        again,_=fit(s,altered,gi,train,'coherence_n1')
        np.testing.assert_allclose(best['parameters'],again['parameters'],atol=1e-10)

if __name__=='__main__':unittest.main()
