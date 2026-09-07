import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_noise_mode_power import pool_power,fit,geometry,transform,band_score
import numpy as np
import unittest

class PowerTests(unittest.TestCase):
    def test_pooling_locality_scale(self):
        coords=np.array([[0,0],[0,1],[4,4]]);raw=np.array([1.,3.,7.]);power,info=pool_power(raw,coords,1)
        np.testing.assert_allclose(power,np.array([2.,2.,7.])/(11/3),atol=1e-10)
        scaled,_=pool_power(raw*49,coords,1);np.testing.assert_allclose(power,scaled,atol=1e-10)
        self.assertEqual(len(power),len(raw));self.assertTrue((power>0).all())

    def test_dense_covariance_parseval(self):
        bands,U,_=geometry(2,2);rng=np.random.default_rng(89);a=rng.normal(size=(2,2,2,3));coeff=transform(a,np.zeros(3));np.testing.assert_allclose(np.sum(a*a),np.sum(coeff*coeff),atol=1e-10)
        power=np.array([.2,.5,1.,2.3]);C=np.eye(3)+.2;block=np.kron(np.diag(power),C);T=np.kron(U,np.eye(3));cov=T@block@T.T;flat=a.reshape(2,12)
        q,lp=band_score(coeff,power,C);expected=np.einsum('bi,ij,bj->b',flat,np.linalg.inv(cov),flat)
        np.testing.assert_allclose(q.sum(axis=1),expected,atol=1e-10);np.testing.assert_allclose(lp.sum(axis=1),-.5*(expected+np.linalg.slogdet(cov)[1]+12*np.log(2*np.pi)),atol=1e-10)

    def test_rank_and_unit(self):
        a=np.random.default_rng(91).normal(size=(3,4,4,8));mean,models,_=fit(a);_,scaled,_=fit(7*a)
        for b,candidates in models.items():
            for n,m in candidates.items():
                self.assertGreater(np.linalg.eigvalsh(m['C']).min(),0);self.assertTrue((m['power']>0).all());np.testing.assert_allclose(m['power'],scaled[b][n]['power'],atol=1e-10);np.testing.assert_allclose(49*m['C'],scaled[b][n]['C'],atol=1e-10)

if __name__=='__main__':unittest.main(verbosity=2)
