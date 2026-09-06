"""Offline mechanics tests; no observational source or velocity files are read."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mond_atlas_pressure_support as pressure

CONFIG = json.loads((ROOT / "configs/mond_atlas_pressure_support_v1.json").read_text(encoding="utf-8"))


class PressureSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=1)
        cls.limits.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.limits.__exit__(None, None, None)

    def setUp(self):
        self.r = np.linspace(0, 4, 41)
        self.column = pressure.gaussian_column(self.r, 1.2, 18)
        self.force = 625*self.r

    def test_frozen_config_hash_and_theory_disposition(self):
        freeze = json.loads((ROOT / "work/gravity-first-principles/mond-atlas-pressure-support-001/freeze.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256((ROOT / freeze["config_path"]).read_bytes()).hexdigest(), freeze["config_sha256"])
        self.assertEqual(CONFIG["admission"], "THEORY_BENCHMARK_ONLY")

    def test_all_independent_controls_pass(self):
        rows = pressure.numerical_controls(CONFIG)
        self.assertGreaterEqual(len(rows), 35)
        self.assertEqual(len(rows), len({row["name"] for row in rows}))
        self.assertEqual([], [row for row in rows if not row["passed"]])

    def test_force_balance_not_prescribed_rotation(self):
        balanced = pressure.surface_balance(self.column, self.force)
        np.testing.assert_allclose(balanced.speed(), 20*self.r, atol=1e-13)
        np.testing.assert_allclose(balanced.recovered_force(), self.force, atol=1e-12)
        self.assertFalse(np.allclose(balanced.speed(), 25*self.r))

    def test_pressureless_and_increasing_pressure(self):
        null = pressure.surface_balance(pressure.gaussian_column(self.r, 1.2, 0), self.force)
        np.testing.assert_allclose(null.speed(), 25*self.r, atol=1e-13)
        rising = pressure.SurfaceColumn(self.r, np.ones_like(self.r), 1+self.r**2, 2*self.r)
        result = pressure.surface_balance(rising, self.force)
        np.testing.assert_allclose(result.rotation_squared, 627*self.r**2)

    def test_units_mass_normalization_and_length(self):
        reference = pressure.surface_balance(self.column, self.force).rotation_squared
        for massfactor in [1e-6, 137, 1e9]:
            converted = pressure.SurfaceColumn(self.r*1000,
                self.column.sigma*massfactor/1e6,
                self.column.integrated_pressure*massfactor/1e6,
                self.column.pressure_gradient*massfactor/1e9)
            np.testing.assert_allclose(pressure.surface_balance(converted, self.force/1000).rotation_squared, reference, rtol=1e-13)

    def test_dispersion_gradient_cannot_be_omitted(self):
        radius = np.array([1., 2., 3.])
        col = pressure.gaussian_column(radius, 2, 16, 3)
        actual = pressure.surface_balance(col, 625*radius).support
        c2 = 256*np.exp(-radius**2/18)
        np.testing.assert_allclose(actual, radius**2*c2*(1/4+1/9))
        self.assertTrue(np.all(actual > radius**2*c2/4))

    def test_regular_center_and_cusp_rejection(self):
        regular = pressure.surface_balance(self.column, self.force)
        self.assertEqual(regular.rotation_squared[0], 0)
        self.assertEqual(regular.recovered_force()[0], 0)
        cusp = pressure.SurfaceColumn(np.array([0.]), np.array([1.]), np.array([18.**2]), np.array([-18.**2/1.2]))
        with self.assertRaisesRegex(ValueError, "Nonregular center"):
            pressure.surface_balance(cusp, np.array([0.]))

    def test_sampling_boundary_has_external_pressure(self):
        extended = pressure.gaussian_column(np.array([3.9, 4., 4.1]), 2, 16, 3)
        self.assertTrue(np.all(extended.integrated_pressure > 0))
        self.assertTrue(np.all(np.diff(extended.integrated_pressure) < 0))

    def test_impossible_force_balance_never_clips(self):
        result = pressure.case_balance(self.r, CONFIG["study"]["impossible_case"])
        self.assertTrue(np.all(result.rotation_squared[1:] < 0))
        signed_copy = result.rotation_squared.copy()
        with self.assertRaisesRegex(ValueError, "Negative vphi"):
            result.speed()
        np.testing.assert_array_equal(result.rotation_squared, signed_copy)
        self.assertEqual(result.status, "NO_STEADY_CIRCULAR_SOLUTION")

    def test_surface_volume_variables_are_not_interchangeable(self):
        volume = pressure.VolumeLayer(self.r, self.column.sigma, self.column.integrated_pressure, self.column.pressure_gradient)
        with self.assertRaises(TypeError):
            pressure.surface_balance(volume, self.force)
        with self.assertRaises(TypeError):
            pressure.volume_balance(self.column, self.force)
        np.testing.assert_allclose(pressure.volume_balance(volume, self.force).rotation_squared, 400*self.r**2)

    def test_invalid_inputs_rejected(self):
        for value in [0, -1, np.inf, np.nan]:
            with self.subTest(scale=value), self.assertRaises(ValueError):
                pressure.gaussian_column(self.r, value, 18)
        for density in [0, -1, np.nan]:
            col = pressure.SurfaceColumn(np.array([1.]), np.array([density]), np.array([1.]), np.array([0.]))
            with self.subTest(density=density), self.assertRaises(ValueError):
                pressure.surface_balance(col, np.array([1.]))

    def test_flow_rejected_even_when_small(self):
        for kwargs in [{"radial_flow": 1e-15}, {"vertical_flow": 2.0}, {"radial_flow": np.nan}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pressure.surface_balance(self.column, self.force, **kwargs)

    def test_thermal_pressure_and_tracer_broadening_are_distinct(self):
        gas_pressure = pressure.effective_pressure_variance(100, [9, 9, 9])
        tracer_line = pressure.spectral_variance(100, 0.05, 9, 4, 16)
        self.assertEqual(gas_pressure, 109)
        self.assertEqual(tracer_line, 34)

    def test_anisotropic_and_single_width_closures_rejected(self):
        for tensor in [18.0, [324, 100, 324], [324, -1, 324], [np.nan]*3]:
            with self.subTest(tensor=tensor), self.assertRaises(ValueError):
                pressure.effective_pressure_variance(0, tensor)

    def test_equal_line_width_does_not_determine_force_correction(self):
        a = pressure.spectral_variance(0, 1, 324, 36)
        b = pressure.spectral_variance(0, 1, 100, 36, 224)
        self.assertEqual(a, b)
        self.assertNotEqual(pressure.effective_pressure_variance(0, [324]*3),
                            pressure.effective_pressure_variance(0, [100]*3))

    def test_flaring_surface_requires_weighted_force(self):
        r = np.array([0.5, 1., 3.])
        c2, ls, lf = 144, 1.8, 2.5
        col = pressure.gaussian_column(r, ls, 12)
        averaged_force = 625*r-c2*r/(2*lf**2)
        expected = r*r*(625-c2/ls**2-c2/(2*lf**2))
        np.testing.assert_allclose(pressure.surface_balance(col, averaged_force).rotation_squared, expected)
        wrong = pressure.surface_balance(col, 625*r).rotation_squared
        np.testing.assert_allclose(wrong-expected, c2*r*r/(2*lf**2))

    def test_fit_matches_independent_linear_speed_solution(self):
        study = CONFIG["study"]
        case = study["cases"][1]
        r = np.linspace(.2, 4, 32)
        train = np.array(study["train_indices"])
        projection = np.sin(np.deg2rad(study["inclination_deg"]))
        # A fixed deterministic perturbation; reference is the closed OLS slope.
        y = 20*r*projection+0.2*np.sin(3*r)
        slope = np.dot(r[train], y[train])/(projection*np.dot(r[train], r[train]))
        for model, correction in [("pressure_blind", 0), ("known_pressure", 225)]:
            fit, pred = pressure.fit_amplitude(r, y, train, case, model, study)
            self.assertTrue(fit["success"])
            self.assertAlmostEqual(fit["amplitude"]/(slope**2+correction), 1, places=7)
            np.testing.assert_allclose(pred, slope*r*projection, rtol=1e-7)

    def test_infeasible_fit_interval_rejected(self):
        study = CONFIG["study"]
        case = dict(study["cases"][1], amplitude_bounds=[1, 200])
        r = np.linspace(.2, 4, 32)
        with self.assertRaisesRegex(ValueError, "No feasible amplitude"):
            pressure.fit_amplitude(r, np.ones_like(r), study["train_indices"], case, "known_pressure", study)

    def test_exact_unknown_pressure_force_ridge(self):
        predictions = []
        for dispersion in [0, 12, 18, 24]:
            case = dict(CONFIG["study"]["cases"][1], dispersion0_km_s=dispersion,
                        amplitude=400+(dispersion/1.2)**2)
            predictions.append(pressure.case_balance(self.r, case).speed())
        for pred in predictions:
            np.testing.assert_allclose(pred, 20*self.r, atol=1e-12)

    def test_independent_oracle_center_finite(self):
        for case in CONFIG["study"]["cases"]:
            with np.errstate(all="raise"):
                v2, vc2, correction = pressure.independent_truth(self.r, case)
            self.assertEqual(v2[0], 0)
            self.assertTrue(np.all(np.isfinite(v2)))

    def test_frozen_radial_partition_and_seed_independence(self):
        study = CONFIG["study"]
        train, test = set(study["train_indices"]), set(study["heldout_indices"])
        self.assertEqual(train & test, set())
        self.assertEqual(train | test, set(range(study["radius_count"])))
        seeds = [s+i*study["case_seed_stride"] for i in range(len(study["cases"])) for s in study["seeds"]]
        fresh = [s+study["fresh_seed_offset"] for s in seeds]
        self.assertEqual(len(set(seeds+fresh)), len(seeds+fresh))


if __name__ == "__main__":
    unittest.main()
