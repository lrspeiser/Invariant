import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_noise_dc_regularization import candidates,gaussian_statistics
import unittest
import numpy as np

class DCTests(unittest.TestCase):
    def test_correction_trace_scale(self):
        S=np.array([1.,2.,4.]);low=np.array([3.,2.,1.]);n=29;models=candidates(S,low,n)
        self.assertAlmostEqual(n/(n-1)*(1+1/n),(n+1)/(n-1),places=14)
        scaled=candidates(49*S,49*low,n)
        for name,m in models.items():
            self.assertTrue((m['variance']>0).all());np.testing.assert_allclose(scaled[name]['variance'],49*m['variance'],atol=1e-10)
            np.testing.assert_allclose(m['variance'].sum(),m['correction']*S.sum(),atol=1e-10)
        with self.assertRaises(ValueError):candidates(S,low,1)

    def test_independent_gaussian(self):
        C=np.diag([1.,2.,3.]);a=np.array([[1.,2.,3.],[-1.,3.,2.]])
        _,q,lp,_=gaussian_statistics(a,C);expected=np.einsum('bi,ij,bj->b',a,np.linalg.inv(C),a)
        np.testing.assert_allclose(q,expected,atol=1e-10);np.testing.assert_allclose(lp,-.5*(expected+np.linalg.slogdet(C)[1]+3*np.log(2*np.pi)),atol=1e-10)

if __name__=='__main__':unittest.main(verbosity=2)
