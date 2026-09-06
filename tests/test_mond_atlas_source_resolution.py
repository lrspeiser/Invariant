"""Independent finite-cell integrals, physical limits and inverse controls."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_source_resolution import cell_projection_matrix, project, adjoint, roughness_gradient, fit_fixed_image
from mond_atlas_nodal_projection import nodal_projection_matrix


class SourceResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_independent_tent_integral_against_laplace_cdf(self):
        def cdf(t, b):
            return .5*np.exp(t/b) if t < 0 else 1-.5*np.exp(-t/b)
        for h in (.125, .0625, .03125):
            for b in (.013, .1, .6):
                for x in (0, .02, .09, .18, .7, 1.7):
                    actual = cell_projection_matrix([x], .125, [0], h, b)[0, 0]
                    f = lambda u: (1-abs(u)/h)*(cdf(x+.0625-u, b)-cdf(x-.0625-u, b))/.125
                    points = sorted({-h, 0., h, *[v for v in (x-.0625, x+.0625) if -h < v < h]})
                    expected = sum(quad(f, l, r, epsabs=1e-12, epsrel=1e-12)[0] for l, r in zip(points[:-1], points[1:]))
                    self.assertLess(abs(actual-expected), 2e-9)

    def test_square_operator_agrees_with_existing_analytic_solution(self):
        axis = (np.arange(31)-15)*.125
        for height in (0., .1, .2, .4):
            a = cell_projection_matrix(axis, .125, axis, .125, height*np.tan(np.deg2rad(53.86233)))
            np.testing.assert_allclose(a, nodal_projection_matrix(31, .125, height, 53.86233), rtol=0, atol=2e-10)

    def test_nested_representation_preserves_the_same_continuum_source(self):
        coarse = np.linspace(-2, 2, 17)
        values = np.array([0, 0, 0, 1, 2, .2, 4, 1, 3, 0, .5, 1, 0, 0, 0, 0, 0.])
        cells = np.arange(-2.5, 2.6, .125)
        for blur in (0., .2, .6):
            reference = cell_projection_matrix(cells, .125, coarse, .25, blur)@values
            for factor in (2, 4):
                nodes = np.linspace(-2, 2, 16*factor+1)
                fine = np.interp(nodes, coarse, values)
                np.testing.assert_allclose(cell_projection_matrix(cells, .125, nodes, .25/factor, blur)@fine,
                                           reference, rtol=0, atol=2e-10)

    def test_thin_limit_units_symmetry_flux_and_finite_boundary(self):
        cells = np.arange(-20, 20.01, .125)
        for h in (.125, .03125):
            a = cell_projection_matrix(cells, .125, [0], h, .3)[:, 0]
            self.assertAlmostEqual(a.sum()*.125/h, 1, delta=1e-9)
            self.assertAlmostEqual(a@cells, 0, delta=1e-11)
            np.testing.assert_allclose(a, a[::-1], atol=1e-12)
            scaled = cell_projection_matrix(cells*1000, 125, [0], h*1000, 300)[:, 0]
            np.testing.assert_allclose(a, scaled, atol=2e-10)
            thin = cell_projection_matrix(cells, .125, [0], h, 0)
            np.testing.assert_allclose(cell_projection_matrix(cells, .125, [0], h, 1e-8), thin, atol=2e-10)
            crop = cell_projection_matrix(np.arange(-.5, .501, .125), .125, [0], h, .3)
            self.assertLess(crop.sum()*.125/h, .9)

    def test_rectangular_adjoint_gradient_and_safe_operator_bound(self):
        rng = np.random.default_rng(654)
        a = cell_projection_matrix(np.arange(5)*.2, .2, np.arange(9)*.1, .1, 0)
        b = cell_projection_matrix(np.arange(7)*.2, .2, np.arange(13)*.1, .1, .3)
        x, y = rng.normal(size=(9, 13)), rng.normal(size=(5, 7))
        self.assertAlmostEqual(np.sum(project(x,a,b)*y), np.sum(x*adjoint(y,a,b)), delta=1e-11)
        w, direction, penalty = rng.uniform(size=y.shape), rng.normal(size=x.shape), .01
        def objective(s):
            return .5*np.sum(w*(project(s,a,b)-y)**2)+.5*penalty*sum(np.sum(np.diff(s,axis=i)**2) for i in (0,1))
        eps = 1e-5
        fd = (objective(x+eps*direction)-objective(x-eps*direction))/(2*eps)
        g = adjoint(w*(project(x,a,b)-y),a,b)+penalty*roughness_gradient(x)
        self.assertAlmostEqual(fd, np.sum(g*direction), delta=1e-8)
        bound = a.sum(0).max()*a.sum(1).max()*b.sum(0).max()*b.sum(1).max()
        self.assertLessEqual((np.linalg.norm(a,2)*np.linalg.norm(b,2))**2, bound+1e-12)

    def fixture(self):
        rng = np.random.default_rng(651)
        nodes = np.arange(9)*.1
        cells = np.arange(5)*.2
        a = cell_projection_matrix(cells,.2,nodes,.1,0)
        b = cell_projection_matrix(cells,.2,nodes,.1,.13)
        support = np.ones((9,9),bool)
        support[[0,-1],:] = False
        support[:,[0,-1]] = False
        source = rng.uniform(size=(9,9))*support
        target = project(source,a,b)+rng.normal(0,.03,(5,5))
        weight = np.ones_like(target)
        weight[1,1] = 0
        return target,weight,a,b,support

    def test_optimizer_against_independent_augmented_design_least_squares(self):
        target,w,a,b,support = self.fixture()
        indices = np.flatnonzero(support)
        design = np.kron(a,b)[:,indices]
        full = np.eye(81).reshape(9,9,81)
        dx = np.diff(full,axis=0).reshape(-1,81)[:,indices]
        dy = np.diff(full,axis=1).reshape(-1,81)[:,indices]
        reg = .01
        matrix = np.vstack((np.sqrt(w.ravel())[:,None]*design, np.sqrt(reg)*dx, np.sqrt(reg)*dy))
        rhs = np.r_[np.sqrt(w.ravel())*target.ravel(), np.zeros(dx.shape[0]+dy.shape[0])]
        solution = lsq_linear(matrix,rhs,bounds=(0,np.inf),tol=1e-12,lsmr_tol=1e-12,max_iter=1000)
        self.assertTrue(solution.success)
        recovered,info = fit_fixed_image(target,w,a,b,support,refinement=2,regularization=reg,tolerance=1e-9)
        self.assertTrue(info['converged'])
        expected = design@solution.x
        self.assertLess(np.linalg.norm(project(recovered,a,b).ravel()-expected)/np.linalg.norm(expected),2e-4)
        self.assertTrue(np.all(recovered[~support]==0))
        self.assertGreaterEqual(recovered.min(),0)

    def test_gpu_agreement_and_omitted_measurement_independence(self):
        target,w,a,b,support = self.fixture()
        kwargs = dict(refinement=2,regularization=.01,max_iterations=200,tolerance=0)
        cpu,info = fit_fixed_image(target,w,a,b,support,**kwargs)
        gpu,ginfo = fit_fixed_image(target,w,a,b,support,backend='cupy',**kwargs)
        np.testing.assert_allclose(project(cpu,a,b),project(gpu,a,b),rtol=0,atol=2e-9)
        target[w==0] = 123456789
        other,_ = fit_fixed_image(target,w,a,b,support,**kwargs)
        np.testing.assert_array_equal(cpu,other)

    def test_invalid_geometry_and_fit_are_rejected(self):
        for h,b in ((0,0),(.1,-1),(.00001,1)):
            with self.assertRaises(ValueError):
                cell_projection_matrix([0],.1,[0],h,b)
        target,w,a,b,support = self.fixture()
        with self.assertRaises(ValueError):
            fit_fixed_image(target,w*2,a,b,support)
        with self.assertRaises(ValueError):
            fit_fixed_image(target,w,-a,b,support)


if __name__ == '__main__':
    unittest.main()
