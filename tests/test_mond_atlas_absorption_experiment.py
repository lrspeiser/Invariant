import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_absorption_experiment import (transmission,packet,clumpy_transmission,
    attenuated_field,transverse_screen_field,curl,analytic_screen_curl)


class AbsorptionTests(unittest.TestCase):
    def test_independent_transport_ode(self):
        for tau in [0,.001,.1,1,3,10]:
            solution=solve_ivp(lambda s,y:-tau*y,[0,1],[1.],rtol=1e-12,atol=1e-14)
            self.assertTrue(solution.success)
            np.testing.assert_allclose(solution.y[0,-1],transmission(tau),rtol=1e-9,atol=1e-12)

    def test_conservation_and_active_accounting(self):
        for tau in [0,.1,1,10]:
            for eta in [0,.25,1,1.2]:
                for f in [0,.5,1]:
                    p=packet(tau,eta,f)
                    self.assertAlmostEqual(sum(p[k] for k in ['direct','forward','backward','retained']),1+p['external_input'],places=13)
                    if eta<=1:self.assertLessEqual(p['direct']+p['forward'],1+1e-15)

    def test_limits(self):
        self.assertEqual(float(transmission(0)),1.)
        self.assertLess(float(transmission(100)),1e-40)
        for tau in [.001,1,10]:
            p=packet(tau,1,1);self.assertAlmostEqual(p['direct']+p['forward'],1.,places=14)
        with self.assertRaises(ValueError):transmission(-1)

    def test_column_additivity_and_quadrature(self):
        # Moving a fixed column within the same ray leaves direct transmission unchanged.
        self.assertAlmostEqual(float(transmission(.2)*transmission(.8)),float(transmission(1)),places=14)
        errors=[]
        for n in [8,16,32,64]:
            s=(np.arange(n)+.5)/n;errors.append(abs(np.mean(1+s*s)-4/3))
        np.testing.assert_allclose(np.array(errors[:-1])/errors[1:],4,rtol=1e-9)

    def test_clumping_bound(self):
        for tau in [.1,1,3]:
            self.assertAlmostEqual(clumpy_transmission(tau,1),transmission(tau),places=14)
            for c in [.1,.25,.5]:
                self.assertGreaterEqual(clumpy_transmission(tau,c),transmission(tau))
                self.assertLessEqual(clumpy_transmission(tau,c),1.)

    def test_rotation_and_vacuum(self):
        p=np.array([1.,2.,3.]);q=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(attenuated_field(q@p),q@attenuated_field(p),atol=1e-12)
        np.testing.assert_allclose(attenuated_field(p,0),-p/np.linalg.norm(p)**3,atol=1e-12)

    def test_curl_against_analytic_and_convergence(self):
        p=[2.,1.,0.];target=analytic_screen_curl(p)
        errors=[np.linalg.norm(curl(transverse_screen_field,p,h)-target) for h in [.1,.05,.025,.0125]]
        self.assertTrue(all(errors[i]>errors[i+1] for i in range(3)))
        np.testing.assert_allclose(np.array(errors[:-1])/errors[1:],4,rtol=.03)
        np.testing.assert_allclose(curl(transverse_screen_field,p,.0001),target,rtol=1e-6,atol=1e-9)
        self.assertGreater(np.linalg.norm(target),.01)


if __name__=='__main__':unittest.main()
