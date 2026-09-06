import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_noise_stationary import covariance,lag_products,coefficient_fit,statistics,WIDTHS
import unittest
import numpy as np

class StationaryTests(unittest.TestCase):
    def test_psd_stationarity_and_scaling(self):
        coeff=np.array([.01,.1,.2,.3,.1,.2]);K=covariance(coeff,4,5)
        self.assertGreater(np.linalg.eigvalsh(K).min(),0)
        self.assertAlmostEqual(K[0,1],K[7,8],places=12)
        np.testing.assert_allclose(covariance(7*coeff,4,5),7*K,atol=1e-10)

    def test_explicit_kronecker(self):
        K=covariance([.1,.1,.2,.3,.1,.2],2,2);C=np.eye(3)+.2;a=np.random.default_rng(97).normal(size=(3,2,2,3));q,lp=statistics(a,np.zeros(3),C,K)
        full=np.kron(K,C);e=a.reshape(3,-1);reference=np.einsum('bi,ij,bj->b',e,np.linalg.inv(full),e)/12
        np.testing.assert_allclose(q,reference,atol=1e-10)
        np.testing.assert_allclose(lp,-.5*(reference*12+np.linalg.slogdet(full)[1]+12*np.log(2*np.pi))/12,atol=1e-10)

    def test_lag_indexing_and_planted_moments(self):
        z=np.random.default_rng(99).normal(size=(2,4,4,3));rows=lag_products(z,2)
        for row in rows:
            products=[]
            for y in range(4):
                for x in range(4):
                    yy=y+row['dy'];xx=x+row['dx']
                    if yy<4 and 0<=xx<4:products.extend((z[:,y,x]*z[:,yy,xx]).ravel())
            self.assertAlmostEqual(np.mean(products),row['product'],places=12)
        coeff=np.array([.02,.1,.2,.3,.1,.2]);planted=[]
        for dy in range(13):
            for dx in range(13):
                d2=dx*dx+dy*dy;value=coeff[0]*(d2==0)+sum(coeff[1:]*np.exp(-d2/(2*WIDTHS**2)))
                planted.append(dict(dy=dy,dx=dx,product=value,pairs=(24-dy)*(24-dx)))
        recovered=coefficient_fit(planted,dict(cap=12,floor=1e-6))
        np.testing.assert_allclose(recovered['coefficients'],coeff,atol=1e-10)

if __name__=='__main__':unittest.main(verbosity=2)
