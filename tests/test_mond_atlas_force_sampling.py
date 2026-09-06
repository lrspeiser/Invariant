"""Analytic controls independent of the galaxy density and FFT solver."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_force_sampling import sample_force,convergence


class ForceSamplingControls(unittest.TestCase):
    def test_full_quadratic_with_cross_terms_and_affine_potential(self):
        spacing=np.array([.17,.23,.11]);origin=np.array([-2.,-3.,-1.])
        axes=[o+np.arange(n)*h for o,n,h in zip(origin,(27,31,29),spacing)]
        x=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
        matrix=np.array([[3.,-.8,.35],[-.8,2.,.6],[.35,.6,4.]])
        linear=np.array([.4,-.7,1.1])
        potential=.5*np.einsum('...i,ij,...j->...',x,matrix,x)+x@linear+9.
        rng=np.random.default_rng(822)
        points=origin+(2+rng.random((100,3))*(np.array(potential.shape)-5))*spacing
        np.testing.assert_allclose(sample_force(potential,origin,spacing,points),-points@matrix-linear,rtol=0,atol=1e-10)
        np.testing.assert_allclose(sample_force(x@linear+5,origin,spacing,points),np.broadcast_to(-linear,points.shape),rtol=0,atol=1e-11)

    def test_boundary_rejection_and_group_discrepancy(self):
        with self.assertRaises(ValueError):sample_force(np.zeros((8,8,8)),[0]*3,[1]*3,np.array([[0.,4.,4.]]))
        reference=np.array([[10.,0.,0.],[10.,0.,0.],[10.,0.,1.],[10.,0.,1.]])
        trial=reference.copy();trial[2:,2]=3
        result=convergence(reference,trial,np.array([0,0,1,1]))
        self.assertAlmostEqual(result['vector_relative_rms'],np.sqrt(8/418))
        self.assertAlmostEqual(result['maximum_group_relative_rms'],2/np.sqrt(109))
        self.assertAlmostEqual(result['vertical_component_relative_rms'],2/3)


if __name__=='__main__':unittest.main()
