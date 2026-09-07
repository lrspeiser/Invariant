"""Independent cell integration and invariance controls before real fields."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.integrate import quad
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_source_cells import component_cells, uniform_axis
from run_mond_atlas_ngc2903_fields import interpolate2


class SourceCells(unittest.TestCase):
    def fixture(self):
        nodes=np.linspace(-1,1,17)
        a=np.maximum(0,1-np.abs((nodes-.125)/.625))
        b=np.maximum(0,1-np.abs((nodes+.125)/.5))
        return nodes,a,b,np.outer(a,b)

    def test_independent_piecewise_linear_and_vertical_quadrature(self):
        nodes,a,b,s=self.fixture()
        axes=[np.arange(-1.125,1.2,.25),np.arange(-1.05,1.1,.3),np.arange(-.4,.41,.2)]
        plane,vertical,receipt=component_cells(s,nodes,axes,[[.3,.1],[.7,.4]],.6)
        means=[]
        for values,axis in ((a,axes[0]),(b,axes[1])):
            d=axis[1]-axis[0]
            out=[]
            for center in axis:
                lo,hi=center-d/2,center+d/2
                knots=[lo,*[t for t in nodes if lo<t<hi],hi]
                value=sum(quad(lambda t:np.interp(t,nodes,values,left=0,right=0),l,u,epsabs=1e-12)[0]
                          for l,u in zip(knots[:-1],knots[1:]))/d
                out.append(value)
            means.append(out)
        np.testing.assert_allclose(plane,.6e6*np.outer(*means),rtol=1e-11,atol=1e-8)
        expected=[]
        for center in axes[2]:
            expected.append(quad(lambda z:.3*np.exp(-abs(z)/.1)/.2+.7*np.exp(-abs(z)/.4)/.8,
                                 center-.1,center+.1,points=[0] if abs(center)<.1 else None,epsabs=1e-12)[0]/.2)
        np.testing.assert_allclose(vertical,expected,rtol=1e-11,atol=1e-12)
        self.assertLess(receipt['vertical_enclosed_fraction'],1)
        self.assertFalse(receipt['resampling_normalization_applied'])

    def test_full_mass_and_unit_conversion(self):
        nodes,a,b,s=self.fixture()
        axes=[np.linspace(-2,2,33)]*2+[np.linspace(-10,10,161)]
        plane,v,rec=component_cells(s,nodes,axes,[[1,.2]],3.)
        expected=3e6*np.trapezoid(a,nodes)*np.trapezoid(b,nodes)
        self.assertAlmostEqual(rec['full_source_mass_msun']/expected,1,delta=1e-12)
        self.assertAlmostEqual(rec['finite_grid_mass_msun']/expected,1,delta=1e-12)
        self.assertAlmostEqual(plane.sum()*v.sum()*.125**3/expected,1,delta=1e-12)

    def test_field_subdivision_preserves_cell_integrals(self):
        nodes,a,b,s=self.fixture()
        coarse=[np.arange(-1,1.01,.25)]*2+[np.arange(-.5,.51,.25)]
        fine=[np.sort(np.r_[axis-.0625,axis+.0625]) for axis in coarse]
        pc,vc,_=component_cells(s,nodes,coarse,[[1,.2]])
        pf,vf,_=component_cells(s,nodes,fine,[[1,.2]])
        np.testing.assert_allclose(pc,pf.reshape(len(coarse[0]),2,len(coarse[1]),2).mean(axis=(1,3)),atol=1e-8)
        np.testing.assert_allclose(vc,vf.reshape(-1,2).mean(axis=1),atol=1e-12)

    def test_subcell_spike_is_not_lost_to_point_sampling(self):
        nodes=np.linspace(-.5,.5,33);s=np.zeros((33,33));s[18,18]=1
        axes=[np.linspace(-1,1,9)]*3
        x,y=np.meshgrid(axes[0],axes[1],indexing='ij')
        self.assertEqual(interpolate2(s,nodes,x,y).sum(),0)
        plane,v,rec=component_cells(s,nodes,axes,[[1,.1]])
        self.assertGreater(plane.sum(),0)
        self.assertAlmostEqual(rec['planar_enclosed_mass_msun']/rec['full_source_mass_msun'],1,delta=1e-12)

    def test_cropped_domain_retains_loss_without_renormalization(self):
        nodes,a,b,s=self.fixture()
        axes=[np.linspace(-.2,.2,5)]*3
        plane,v,rec=component_cells(s,nodes,axes,[[1,.4]])
        self.assertLess(rec['finite_fraction'],.5)
        self.assertLess(rec['planar_enclosed_mass_msun'],rec['full_source_mass_msun'])
        self.assertFalse(rec['resampling_normalization_applied'])

    def test_common_length_scaling_has_inverse_cubic_density(self):
        nodes,a,b,s=self.fixture();axes=[np.linspace(-1.5,1.5,25)]*3
        p,v,rec=component_cells(s,nodes,axes,[[1,.2]])
        k=3.
        ps,vs,recs=component_cells(s/k**2,nodes*k,[x*k for x in axes],[[1,.2*k]])
        np.testing.assert_allclose(ps, p/k**2,rtol=1e-11,atol=1e-8)
        np.testing.assert_allclose(vs,v/k,rtol=1e-11,atol=1e-12)
        self.assertAlmostEqual(recs['finite_grid_mass_msun']/rec['finite_grid_mass_msun'],1,delta=1e-12)

    def test_sheet_negative_edges_and_bad_axis_fail_closed(self):
        nodes,a,b,s=self.fixture();axes=[np.linspace(-2,2,17)]*3
        for layers in ([[1,0]],[[1,-.1]],[[.8,.1]],[[-1,.1],[2,.2]]):
            with self.assertRaises(ValueError):component_cells(s,nodes,axes,layers)
        s[0,0]=1
        with self.assertRaises(ValueError):component_cells(s,nodes,axes,[[1,.1]])
        s[0,0]=-1
        with self.assertRaises(ValueError):component_cells(s,nodes,axes,[[1,.1]])
        with self.assertRaises(ValueError):uniform_axis([0,1,1.9])


if __name__=='__main__':unittest.main()
