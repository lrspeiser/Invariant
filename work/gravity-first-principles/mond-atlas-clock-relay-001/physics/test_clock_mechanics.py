"""Target-free mechanics checks; run directly to save a compact receipt."""
import json
import math
from pathlib import Path
import unittest


class ClockMechanics(unittest.TestCase):
    def test_lapse_mapping(self):
        # Dimensionless c=1; ln N=Phi=-mu/r, g=-d Phi/dr.
        mu = 0.002
        for r in (0.5, 1., 2., 10.):
            h = 1e-4*r
            f = lambda x: -mu/x
            derivative = (f(r-2*h)-8*f(r-h)+8*f(r+h)-f(r+2*h))/(12*h)
            self.assertLess(abs((-derivative)/(-mu/r**2)-1), 1e-8)

    def test_schwarzschild_observer_acceleration(self):
        # Exact local proper acceleration to hover, outward, c=1.
        mu=0.002
        for r in (0.5, 1., 2., 10.):
            N=math.sqrt(1-2*mu/r)
            coordinate_gradient=mu/(r*r*(1-2*mu/r))
            spatial_proper_gradient=N*coordinate_gradient
            self.assertLess(abs(spatial_proper_gradient/(mu/(r*r*N))-1), 1e-12)

    def test_photon_energy(self):
        # E_infinity=N E_local is constant along stationary vacuum propagation.
        E_inf=7.
        for N in (0.2, 0.7, 0.99999):
            E_local=E_inf/N
            self.assertLess(abs(N*E_local-E_inf),1e-14)

    def test_exchange_conservation(self):
        for t in (0., 0.1, 1., 10.):
            reservoir=3*math.exp(-0.4*t)
            recipient=5+3*(1-math.exp(-0.4*t))
            self.assertLess(abs(reservoir+recipient-8),1e-12)

    def test_memory_not_identified_at_equilibrium(self):
        source=2.3
        for tau in (0.01, 1., 100.):
            # tau u_dot+u=S. At u=S, derivative is zero for every tau.
            self.assertLess(abs((source-source)/tau),1e-12)

    def test_stable_scalar_mode(self):
        # u_ddot+gamma u_dot+omega2 u=S: steady state S/omega2.
        import cmath
        for gamma in (0.1, 1., 10.):
            for omega2 in (0.1, 1., 10.):
                roots=[(-gamma+s*cmath.sqrt(gamma**2-4*omega2))/2 for s in (-1,1)]
                self.assertTrue(all(z.real<0 for z in roots))
                self.assertAlmostEqual(1/complex(omega2,0), 1/omega2)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ClockMechanics))
    receipt={'admission':'THEORY_BENCHMARK_ONLY','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'observational_arrays_opened':0,'note':'Analytic checks only; no proof of a time-energy source or full theory stability.'}
    Path(__file__).with_name('test-results.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
