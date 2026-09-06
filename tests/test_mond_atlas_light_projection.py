"""Independent synthetic numerical/contract tests; no observational inputs."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from mond_atlas_light_projection import (
    AngularDistances, C_SI, G_SI, KPC_M, MSUN_KG, ScalarMetric,
    SphericalComponent, deflection, lens_jacobian, lens_map, line_quadrature,
    manufactured_metric, signed_magnification, transverse_gradient,
)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.mass = 1e11*MSUN_KG
        self.point = manufactured_metric([SphericalComponent(self.mass)], eta=1)
        self.plummer = manufactured_metric([SphericalComponent(self.mass, .8*KPC_M)], eta=1)
        self.geom = AngularDistances(2e25, 4e25, 3e25)

    def test_point_integral_has_GR_factor_two_and_outward_sign(self):
        alpha = deflection(self.point, [KPC_M, 0])
        np.testing.assert_allclose(alpha, [4*G_SI*self.mass/(C_SI**2*KPC_M), 0], rtol=2e-7, atol=1e-14)

    def test_psi_not_silently_identified_with_phi(self):
        phi_only = manufactured_metric([SphericalComponent(self.mass)], eta=0)
        np.testing.assert_allclose(deflection(phi_only, [KPC_M, .3*KPC_M]),
                                   .5*deflection(self.point, [KPC_M, .3*KPC_M]), rtol=2e-7)

    def test_opposite_potentials_cancel(self):
        field = manufactured_metric([SphericalComponent(self.mass)], eta=-1)
        np.testing.assert_allclose(deflection(field, [KPC_M, 0]), [0,0], atol=1e-14)

    def test_no_closure_label_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalarMetric(lambda x,y,z: 0, lambda x,y,z: 0, "")

    def test_constant_callable_broadcasts(self):
        metric = ScalarMetric(lambda x,y,z: 1e8, lambda x,y,z: -2e8, "constant test")
        np.testing.assert_array_equal(deflection(metric, [KPC_M, 0]), [0,0])

    def test_polynomial_stencil_exactness(self):
        metric = ScalarMetric(lambda x,y,z: 1e-5*(x**4+2*y**3), lambda x,y,z: 0, "manufactured polynomial")
        result = transverse_gradient(metric, [2,3], [0,1,2], step_m=.02)
        np.testing.assert_allclose(result, [32e-5,54e-5], rtol=1e-10)

    def test_uniform_gradient_finite_integral(self):
        acceleration = np.array([2e-10,-3e-10])
        metric = ScalarMetric(lambda x,y,z: acceleration[0]*x+acceleration[1]*y,
                              lambda x,y,z: 0, "manufactured gradient")
        alpha = deflection(metric, [KPC_M, KPC_M], half_depth_m=3*KPC_M)
        np.testing.assert_allclose(alpha, acceleration*6*KPC_M/C_SI**2, rtol=1e-9)

    def test_individual_strong_potentials_rejected_even_if_sum_zero(self):
        metric = ScalarMetric(lambda x,y,z: C_SI**2, lambda x,y,z: -C_SI**2, "invalid weak field")
        with self.assertRaisesRegex(ValueError, "weak-field"):
            deflection(metric, [KPC_M, 0])

    def test_nonfinite_potential_rejected(self):
        metric = ScalarMetric(lambda x,y,z: float("nan"), lambda x,y,z: 0, "invalid field")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            deflection(metric, [KPC_M, 0])

    def test_point_singular_ray_guard(self):
        for x in [0, 1e-4*KPC_M, 4e-4*KPC_M]:
            with self.subTest(x=x), self.assertRaisesRegex(ValueError, "singular"):
                deflection(self.point, [x,0])

    def test_softened_central_ray_is_regular(self):
        np.testing.assert_allclose(deflection(self.plummer, [0,0]), [0,0], atol=1e-14)

    def test_finite_point_tail_not_implicitly_repaired(self):
        infinite = 4*G_SI*self.mass/(C_SI**2*KPC_M)
        finite = deflection(self.point, [KPC_M,0], half_depth_m=2*KPC_M)[0]
        self.assertAlmostEqual(finite/infinite, 2/np.sqrt(5), places=8)
        self.assertLess(finite, .9*infinite)

    def test_plummer_projected_mass_law(self):
        radius, a = 1.3*KPC_M, .8*KPC_M
        mass_projected = self.mass*radius**2/(radius**2+a**2)
        expected = 4*G_SI*mass_projected/(C_SI**2*radius)
        self.assertAlmostEqual(deflection(self.plummer, [radius,0])[0]/expected, 1, places=7)

    def test_independent_distances_use_supplied_Dls(self):
        theta = np.array([KPC_M/self.geom.D_l_m, 0])
        mapped = lens_map(self.point, theta, self.geom)
        expected = theta-np.array([4*G_SI*self.mass/(C_SI**2*KPC_M),0])*3/4
        np.testing.assert_allclose(mapped, expected, rtol=2e-7, atol=1e-14)

    def test_angular_distances_need_not_be_monotone(self):
        geom = AngularDistances(4e25, 2e25, 1e25)
        self.assertEqual(geom.efficiency, .5)

    def test_distance_validation(self):
        for values in [(0,1,1),(1,0,1),(1,1,-1),(1,np.nan,1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                AngularDistances(*values)

    def test_zero_efficiency_identity(self):
        geom = AngularDistances(2e25, 4e25, 0)
        theta = np.array([1e-5,2e-5])
        np.testing.assert_array_equal(lens_map(self.point, theta, geom), theta)

    def test_point_lens_full_2d_jacobian(self):
        te = np.sqrt(4*G_SI*self.mass/C_SI**2*self.geom.efficiency/self.geom.D_l_m)
        theta = te*np.array([1.4,.6])
        norm2 = theta@theta
        expected = np.eye(2)-te**2*(np.eye(2)/norm2-2*np.outer(theta,theta)/norm2**2)
        jac = lens_jacobian(self.point, theta, self.geom, angular_step_rad=.001*te)
        np.testing.assert_allclose(jac, expected, rtol=2e-5, atol=2e-6)

    def test_magnification_preserves_parity(self):
        self.assertEqual(signed_magnification(np.diag([2.,-3.])), -1/6)
        self.assertEqual(signed_magnification(np.diag([2.,3.])), 1/6)

    def test_critical_jacobian_rejected(self):
        with self.assertRaisesRegex(ValueError, "critical"):
            signed_magnification(np.diag([0.,1.]))

    def test_quadrature_finite_measure(self):
        ell, weights = line_quadrature(order=256, scale_m=2, half_depth_m=3, ell_origin_m=9)
        self.assertTrue(np.all((ell>6)&(ell<12)))
        self.assertAlmostEqual(float(weights.sum()), 6, places=12)

    def test_numerical_parameter_guards(self):
        for kwargs in [dict(order=1),dict(order=2.5),dict(order=True),dict(step_m=0),
                       dict(scale_m=-1),dict(half_depth_m=0),dict(c=0),dict(ell_origin_m=np.nan)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                deflection(self.point, [KPC_M,0], **kwargs)

    def test_mass_and_scale_validation(self):
        for mass,scale in [(-1,0),(1,-1),(np.inf,0)]:
            with self.subTest(mass=mass,scale=scale), self.assertRaises(ValueError):
                SphericalComponent(mass,scale)


if __name__ == "__main__":
    unittest.main()
