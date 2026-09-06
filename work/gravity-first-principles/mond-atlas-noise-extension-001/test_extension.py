import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_noise_extension import models,tiles,scores
import unittest
import numpy as np

class ExtensionTests(unittest.TestCase):
    def test_aggregate_independent(self):
        a=np.arange(3*4*4*5).reshape(3,4,4,5)
        expected=np.array([[[a[b,y:y+2,x:x+2].mean(axis=(0,1)) for x in (0,2)] for y in (0,2)] for b in range(3)])
        np.testing.assert_allclose(tiles(a,2),expected,atol=1e-10)

    def test_rank_psd_and_amplitude_scaling(self):
        rng=np.random.default_rng(319);a=rng.normal(size=(3,4,4,8))
        mean,covs=models(a,4);mean2,covs2=models(7*a,4)
        for name,c in covs.items():
            self.assertTrue(np.linalg.eigvalsh(c).min()>0)
            np.testing.assert_allclose(covs2[name],49*c,rtol=1e-10,atol=1e-10)
        with self.assertRaises(ValueError):models(a*np.nan,2)

    def test_inverse_score_and_planted_diagonal(self):
        rng=np.random.default_rng(321);a=rng.normal(size=(400,2,2,3));eye=np.eye(3)
        result=scores(a,np.zeros(3),eye,1)
        q=np.einsum('...i,ij,...j->...',a,np.linalg.inv(eye),a).mean(axis=(1,2))/3
        np.testing.assert_allclose(result['q_over_n'],q,atol=1e-10)
        wrong=np.full((3,3),.8);np.fill_diagonal(wrong,1)
        self.assertGreater(result['logpdf_per_channel'].mean(),scores(a,np.zeros(3),wrong,1)['logpdf_per_channel'].mean())
        expected=-.5*(3*q+np.linalg.slogdet(eye)[1]+3*np.log(2*np.pi))/3
        np.testing.assert_allclose(result['logpdf_per_channel'],expected,atol=1e-10)

if __name__=='__main__':unittest.main(verbosity=2)
