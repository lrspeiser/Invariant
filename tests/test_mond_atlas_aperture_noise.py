import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_aperture_noise import tiles,fit,scores


class ApertureTests(unittest.TestCase):
    def test_loop_aggregation(self):
        a=np.arange(2*12*12*3,dtype=float).reshape(2,12,12,3)
        for s in [1,2,3,4,6,12]:
            reference=np.array([[[a[b,y:y+s,x:x+s].mean(axis=(0,1)) for x in range(0,12,s)] for y in range(0,12,s)] for b in range(2)])
            np.testing.assert_array_equal(tiles(a,s),reference)

    def test_analytic_common_mode_covariance(self):
        # Sum of exact independent-pixel and shared-mode covariance, not sampled noise.
        independent=np.array([[2.,.3],[.3,1.]])
        common=np.array([[.4,.1],[.1,.2]])
        for s in [1,2,4]:
            n=s*s; full=np.kron(np.eye(n),independent)+np.kron(np.ones((n,n)),common)
            average=np.kron(np.ones((1,n))/n,np.eye(2))
            np.testing.assert_allclose(average@full@average.T,independent/n+common,atol=1e-14)
            # Exact basis-vector aggregation also matches this averaging operator.
            basis=np.eye(n*2).reshape(n*2,s,s,2)
            operator=tiles(basis,s)[:,0,0,:].T
            np.testing.assert_allclose(operator@full@operator.T,independent/n+common,atol=1e-14)

    def test_independent_score(self):
        a=np.arange(2*4*4*2,dtype=float).reshape(2,4,4,2)/30
        c=np.array([[2.,.3],[.3,1.]]); mean=np.array([.2,.4]); v=tiles(a-mean,2)
        q=np.einsum('...i,ij,...j->...',v,np.linalg.inv(c),v)
        expected=(-.5*(q+np.linalg.slogdet(c)[1]+2*np.log(2*np.pi))).mean(axis=(1,2))/2
        actual=scores(a,mean,c,2)
        np.testing.assert_allclose(actual['q_over_n'],q.mean(axis=(1,2))/2,atol=1e-12)
        np.testing.assert_allclose(actual['logpdf_per_channel'],expected,atol=1e-12)

    def test_training_and_rank_deficiency(self):
        rng=np.random.default_rng(912); training=rng.normal(size=(3,4,4,8))
        mean,c=fit(training,[1,2,4]); untouched={k:v.copy() for k,v in c.items()}
        scores(np.full((2,4,4,8),1e6),mean,c[4],4)
        for k,v in c.items():
            np.testing.assert_array_equal(v,untouched[k]); np.linalg.cholesky(v)
        self.assertTrue(np.all(np.linalg.eigvalsh(c[4])>0))

    def test_invalid(self):
        for side in [0,3,1.5]:
            with self.assertRaises(ValueError): tiles(np.ones((2,4,4,2)),side)
        with self.assertRaises(ValueError): tiles(np.full((2,4,4,2),np.nan),2)

if __name__=='__main__': unittest.main()
