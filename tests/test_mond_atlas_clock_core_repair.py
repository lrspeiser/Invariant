import sys,unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mond_atlas_clock_core_repair import grid,extra_acceleration,predict,select,G,A0

class ClockCoreTests(unittest.TestCase):
    def test_grid_zero_and_finite(self):
        self.assertEqual(len(grid()),144)
        r=np.geomspace(.01,100,30)
        s=dict(r=r,gas=np.ones(30)*10,disk=np.ones(30)*30,bulge=np.ones(30)*3,rd=np.ones(30)*3,luminosity=np.ones(30)*10,hi=np.ones(30)*2)
        for c in grid():
            p=predict(s,c);self.assertTrue(np.isfinite(p).all())
            if c['beta']==0:
                np.testing.assert_allclose(10**(2*p),100+c['mf']*(.5*900+.7*9),rtol=1e-10)

    def test_potential_integral_gradient(self):
        gm=5e4;d=3.;beta=3.
        for psi in [A0*d,.3*np.sqrt(gm*A0)]:
            B=gm/psi
            for r in [.1,1.,10.,100.]:
                # Compactified integral to infinity, separately written integrand.
                def potential(start):
                    def integrand(u):
                        t=start+u/(1-u)
                        return beta*gm*t/((t+d)**2*(t+d+B))/(1-u)**2
                    return -quad(integrand,0,1,epsabs=1e-8,epsrel=1e-12)[0]
                h=r*1e-3
                deriv=(potential(r-2*h)-8*potential(r-h)+8*potential(r+h)-potential(r+2*h))/(12*h)
                self.assertLess(abs(deriv/extra_acceleration(r,gm,d,beta,psi)-1),1e-6)

    def test_charge_derivative_and_limits(self):
        gm=1e4;d=2.;beta=3.;psi=1e3;B=gm/psi
        r=np.geomspace(1e-8,1e9,300)
        charge=r**3/((r+d)**2*(r+d+B))
        deriv=charge*(3/r-2/(r+d)-1/(r+d+B))
        self.assertTrue((deriv>0).all());self.assertTrue((np.diff(charge)>0).all())
        self.assertLess(abs(charge[-1]-1),1e-5)
        g=extra_acceleration(r,gm,d,beta,psi)
        self.assertLess(abs(g[0]/r[0]/(beta*gm/(d*d*(d+B)))-1),1e-5)
        self.assertLess(abs(g[-1]*r[-1]**2/(beta*gm)-1),1e-5)

    def test_dimensions_and_training_only(self):
        r=np.array([.1,1,10.]);gm=1e4;d=3;psi=1e3
        # Scale lengths and GM by same factor, preserving potential units.
        np.testing.assert_allclose(extra_acceleration(r*7,gm*7,d*7,3,psi),extra_acceleration(r,gm,d,3,psi)/7,rtol=1e-10)
        loss=np.array([[1.,2.,100.],[2.,3.,0.]])
        self.assertEqual(select(loss,[True,True,False]),0)
        loss[:,-1]=np.nan;self.assertEqual(select(loss,[True,True,False]),0)

if __name__=='__main__':unittest.main()
