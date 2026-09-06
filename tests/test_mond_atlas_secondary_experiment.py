import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_secondary_experiment import disk_sources,secondary,potential
from mond_atlas_halo_return import field,numerical_gradient,mass_shape


class SecondaryTests(unittest.TestCase):
    def test_moments(self):
        for nr in (16,32):
            s,m=disk_sources(7.,3.,nr,2*nr)
            self.assertAlmostEqual(m.sum(),7.,places=12)
            self.assertAlmostEqual(np.sum(m*np.sum(s*s,axis=1))/7,54.,places=10)
            np.testing.assert_allclose(np.sum(m[:,None]*s,axis=0),0,atol=1e-12)

    def test_point_and_reciprocity(self):
        p=np.array([2.,3.,4.]);eta,L,M=2.,3.,5.
        np.testing.assert_allclose(secondary(p,[[0,0,0]],[M],eta,L),field(p,eta*M/(4*np.pi*L**3),L,'NFW'),rtol=1e-12)
        np.testing.assert_allclose(5*secondary(p,[[0,0,0]],[3],eta,L),-3*secondary([0,0,0],[p],[5],eta,L),atol=1e-12)

    def test_covariance_gradient_and_superposition(self):
        s=np.array([[1.,0,0],[-1,0,0]]);m=np.array([2.,3.]);p=np.array([2.,3.,4.]);shift=np.array([9,-8,7])
        Q=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        g=secondary(p,s,m,2,3)
        np.testing.assert_allclose(secondary(p+shift,s+shift,m,2,3),g,atol=1e-12)
        np.testing.assert_allclose(secondary(Q@p,s@Q.T,m,2,3),Q@g,atol=1e-12)
        np.testing.assert_allclose(g,-numerical_gradient(lambda x:potential(x,s,m,2,3),p),rtol=1e-6)
        np.testing.assert_allclose(g,sum(secondary(p,s[i:i+1],m[i:i+1],2,3) for i in range(2)),atol=1e-12)

    def test_far_field_and_truncation(self):
        s,m=disk_sources(5.,1.,32,64);p=np.array([1000.,0,500.])
        g=secondary(p,s,m,2,3);ref=secondary(p,[[0,0,0]],[5.],2,3)
        self.assertLess(np.linalg.norm(g-ref)/np.linalg.norm(ref),1e-4)
        for x in (1,5,10,20,100):
            trunc=secondary([x,0,0],[[0,0,0]],[1],1,1,cutoff=10)
            self.assertAlmostEqual(-trunc[0]*x*x,float(mass_shape(min(x,10),'NFW')),places=12)


if __name__=='__main__':unittest.main()
