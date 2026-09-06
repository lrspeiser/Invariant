import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_noise_joint_program import fit,statistics,aperture_operator
import unittest
import numpy as np

class JointTests(unittest.TestCase):
    def test_explicit_kronecker(self):
        rng=np.random.default_rng(93);a=rng.normal(size=(3,2,2,3));K=np.eye(4)+.2;C=np.eye(3)+.1
        q,lp=statistics(a,np.zeros(3),C,K);cov=np.kron(K,C);e=a.reshape(3,-1)
        expected=np.einsum('bi,ij,bj->b',e,np.linalg.inv(cov),e)/12
        np.testing.assert_allclose(q,expected,atol=1e-10)
        np.testing.assert_allclose(lp,-.5*(12*expected+np.linalg.slogdet(cov)[1]+12*np.log(2*np.pi))/12,atol=1e-10)

    def test_aperture_projection(self):
        K=np.eye(16)+.3;C=np.eye(2)+.2;A=aperture_operator(4,4,2);operator=np.kron(A,np.eye(2))
        np.testing.assert_allclose(operator@np.kron(K,C)@operator.T,np.kron(A@K@A.T,C),atol=1e-10)
        np.testing.assert_allclose(np.diag(A@K@A.T),.25+.3,atol=1e-10)

    def test_rank_and_scale(self):
        a=np.random.default_rng(95).normal(size=(2,3,3,2));m,C,ks=fit(a);m2,C2,ks2=fit(7*a)
        np.testing.assert_allclose(C2,49*C,rtol=1e-10)
        for alpha,K in ks.items():
            self.assertGreater(np.linalg.eigvalsh(K).min(),0)
            np.testing.assert_allclose(K,ks2[alpha],rtol=1e-10,atol=1e-10)

if __name__=='__main__':unittest.main(verbosity=2)
