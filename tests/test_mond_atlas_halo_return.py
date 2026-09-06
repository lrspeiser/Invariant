import sys,unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_halo_return import mass_shape,density_shape,field,nfw_potential,mn_potential,mn_field,numerical_gradient,curl,return_shape,fit_return


class HaloReturnTests(unittest.TestCase):
    def test_independent_mass_integral(self):
        for profile in ['NFW','Burkert']:
            for x in np.geomspace(1e-5,1e3,30):
                # Rescale integration to [0,1] to resolve tiny masses accurately.
                ref=quad(lambda t:float(density_shape(x*t,profile))*t*t,0,1,epsabs=1e-12,epsrel=1e-12)[0]*x**3
                self.assertLess(abs(float(mass_shape(x,profile))/ref-1),1e-9)

    def test_galpy_independent_force(self):
        from galpy.potential import NFWPotential,BurkertPotential
        for profile,pot in [('NFW',NFWPotential(amp=1.,a=1.)),('Burkert',BurkertPotential(amp=1/(4*np.pi),a=1.))]:
            for R,z in [(0.1,.2),(1.,0.),(2.,3.),(10.,-7.)]:
                actual=field([R,0,z],1/(4*np.pi),1,profile)
                expected=np.array([pot.Rforce(R,z,use_physical=False),0,pot.zforce(R,z,use_physical=False)])
                self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-8)

    def test_potential_gradient_and_poisson(self):
        for p in [np.array([.2,.4,.6]),np.array([2.,3.,4.])]:
            np.testing.assert_allclose(-numerical_gradient(lambda v:nfw_potential(v,.7,1.3),p),field(p,.7,1.3,'NFW'),rtol=1e-6)
            np.testing.assert_allclose(-numerical_gradient(mn_potential,p),mn_field(p),rtol=1e-6)
        for profile in ['NFW','Burkert']:
            for x in [.02,.2,1.,10.]:
                h=x*1e-4
                derivative=(mass_shape(x+h,profile)-mass_shape(x-h,profile))/(2*h)
                self.assertLess(abs(derivative/(x*x*density_shape(x,profile))-1),1e-5)

    def test_symmetry_units_and_limits(self):
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1]]);p=np.array([1.,2.,3.])
        for profile in ['NFW','Burkert']:
            f=field(p,.7,2.,profile)
            np.testing.assert_allclose(field(p@rot,.7,2.,profile),f@rot,atol=1e-10)
            np.testing.assert_allclose(field(p*7,.7/7**3,14.,profile),f/49,rtol=1e-12)
            self.assertLess(np.linalg.norm(curl(lambda q:field(q,.7,2.,profile),p)),1e-7)
        self.assertAlmostEqual(float(mass_shape(1e-6,'NFW'))/1e-12,.5,places=5)
        self.assertAlmostEqual(float(mass_shape(1e-6,'Burkert'))/1e-18,1/3,places=5)
        with self.assertRaises(ValueError):field([0,0,0],1,1,'NFW')

    def test_finite_return_budget_and_geometry(self):
        for kind in ['bounded_p2','bounded_p3']:
            r=np.geomspace(1e-4,1e6,100);charge=r*r*return_shape(r,kind,.7,2.)
            self.assertTrue(np.all(np.diff(charge)>0));self.assertLess(abs(charge[-1]/2.8-1),1e-5)
        p=np.array([4.,0,2.]);disk=mn_field(p);radial=-p/np.linalg.norm(p)
        best=(radial@disk)/(disk@disk)*disk
        self.assertGreater(np.linalg.norm(best-radial),.1)
        # A central spherical source is a positive scalar-reinforcement control.
        spherical=-p/np.linalg.norm(p)**3
        np.testing.assert_allclose((radial@spherical)/(spherical@spherical)*spherical,radial,atol=1e-12)

if __name__=='__main__':unittest.main()
