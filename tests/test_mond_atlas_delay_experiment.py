import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_delay_experiment import kernel,step,transfer,integrate_mode


class DelayTests(unittest.TestCase):
    def test_symmetry_normalization_translation(self):
        k=kernel();np.testing.assert_allclose(k,k.T,atol=1e-15)
        np.testing.assert_allclose(k.sum(axis=0),1,atol=1e-15)
        x=np.arange(16.)
        np.testing.assert_allclose(k@np.roll(x,3),np.roll(k@x,3),atol=1e-14)
        self.assertAlmostEqual(np.linalg.eigvalsh(k)[-1],1.,places=14)

    def test_modes_step_frequency(self):
        for alpha in [0,.9,1.,1.1]:
            t=np.linspace(0,10,101)
            np.testing.assert_allclose(integrate_mode(t,alpha=alpha),step(t,alpha=alpha),rtol=1e-8,atol=1e-9)
        for alpha in [0,.9]:
            t=np.linspace(0,20,101);h=transfer(1,alpha=alpha)
            np.testing.assert_allclose(integrate_mode(t,alpha=alpha,omega=1),h*np.exp(1j*t),rtol=1e-8,atol=1e-9)

    def test_matrix_reference(self):
        k=kernel();rho=np.zeros(16);rho[0]=1
        a=.8*k-np.eye(16);b=k@rho
        times=np.linspace(0,10,51)
        numeric=solve_ivp(lambda t,y:a@y+b,(0,10),np.zeros(16),t_eval=times,method='DOP853',rtol=1e-10,atol=1e-12).y.T
        exact=np.array([np.linalg.solve(a,(expm(a*t)-np.eye(16))@b) for t in times])
        np.testing.assert_allclose(numeric,exact,atol=1e-9,rtol=1e-8)

    def test_time_scaling_static_degeneracy(self):
        np.testing.assert_allclose(step(np.array([1,2,3])*7,alpha=.5,tau=7),step([1,2,3],alpha=.5),rtol=1e-14)
        for tau in [.1,1,10]:
            self.assertAlmostEqual(transfer(0,alpha=.9,tau=tau).real,10)
        self.assertGreater(step(10,alpha=1.1),step(10,alpha=1))
        self.assertAlmostEqual(float(step(10,alpha=1)),10)


if __name__=='__main__': unittest.main()
