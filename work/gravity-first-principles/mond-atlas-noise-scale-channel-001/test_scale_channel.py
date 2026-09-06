import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_noise_scale_channel import geometry,transform,score_coeff,fit
import numpy as np
import unittest

class ScaleChannelTests(unittest.TestCase):
    def test_parseval_basis(self):
        bands,U,_=geometry(3,4);a=np.random.default_rng(71).normal(size=(2,3,4,3));coeff=transform(a,np.zeros(3))
        np.testing.assert_allclose(U.T@U,np.eye(12),atol=1e-10)
        np.testing.assert_allclose(np.sum(a*a),np.sum(coeff*coeff),atol=1e-10)
        np.testing.assert_allclose(coeff,np.einsum('ms,bsc->bmc',U.T,a.reshape(2,12,3)),atol=1e-10)

    def test_explicit_covariance_and_aperture(self):
        bands,U,_=geometry(2,2);covs={name:np.eye(3)*(i+1)+.1 for i,name in enumerate(bands)};blocks=[]
        for mode in range(4):blocks.append(next(covs[b] for b,mask in bands.items() if mask[mode]))
        diagonal=np.zeros((12,12))
        for i,C in enumerate(blocks):diagonal[i*3:i*3+3,i*3:i*3+3]=C
        transform_matrix=np.kron(U,np.eye(3));full=transform_matrix@diagonal@transform_matrix.T
        a=np.random.default_rng(73).normal(size=(2,2,2,3));q,lp,_=score_coeff(transform(a,np.zeros(3)),bands,covs);e=a.reshape(2,12)
        expected=np.einsum('bi,ij,bj->b',e,np.linalg.inv(full),e)/12
        np.testing.assert_allclose(q,expected,atol=1e-10)
        np.testing.assert_allclose(lp,-.5*(12*expected+np.linalg.slogdet(full)[1]+12*np.log(2*np.pi))/12,atol=1e-10)
        A=np.array([[.5,.5,0,0]]);projection=np.kron(A,np.eye(3));weights=(A@U).ravel()**2
        derived=sum(weights[i]*C for i,C in enumerate(blocks));np.testing.assert_allclose(projection@full@projection.T,derived,atol=1e-10)

    def test_rank_regularization(self):
        data=np.random.default_rng(75).normal(size=(3,4,4,8));mean,models=fit(data)
        for candidates in models.values():
            for C in candidates.values():self.assertGreater(np.linalg.eigvalsh(C).min(),0)

if __name__=='__main__':unittest.main(verbosity=2)
