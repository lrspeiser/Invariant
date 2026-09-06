"""Scientific controls for marginal/conditional noise and response isolation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import multivariate_normal
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_mond_atlas_motion_covariance as runner
from mond_atlas_motion_controls import Geometry, Instrument, forward_cube
from mond_atlas_motion_covariance import (
    FixedPartition, GaussianBlock, ar1_from_innovations, channel_covariance,
    conditional_channels, fit_motion, forecast_evaluation, innovation_noise,
    numerical_controls, pixel_scales,
)


class CovarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.prior, cls.freeze = runner.load_frozen()
        cls.thread_limit = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.thread_limit.restore_original_limits()

    def test_immutable_prior_and_theory_only_freeze(self):
        c, p, f = runner.load_frozen()
        self.assertIsNone(c["source_dataset"])
        self.assertFalse(c["observational_scoring_allowed"])
        self.assertFalse(f["synthetic_study_generated"])
        self.assertEqual(p["disposition"], "THEORY_BENCHMARK_ONLY")

    def test_all_frozen_statistical_and_mechanics_controls(self):
        r = numerical_controls(self.config, self.prior)
        failed = [v for v in r["statistical_controls"] if not v["passed"]]
        self.assertTrue(r["all_passed"], failed)
        self.assertEqual(r["statistical_control_count"], 30)
        self.assertEqual(r["prior_forward_control_count"], 25)

    def test_non_AR_covariance_whitening_and_logpdf_against_dense(self):
        rng = np.random.default_rng(500)
        a = rng.normal(size=(5, 5))
        k = a @ a.T+0.5*np.eye(5)
        scale = np.array([0.8, 2.0, 1.2])
        r = rng.normal(size=(5, 3))
        block = GaussianBlock(k, scale)
        c = np.kron(k, np.diag(scale**2))
        expected_q = r.ravel() @ np.linalg.inv(c) @ r.ravel()
        self.assertAlmostEqual(block.score(r)["q"], expected_q, places=12)
        self.assertAlmostEqual(-block.score(r)["negative_log_likelihood"],
                               multivariate_normal.logpdf(r.ravel(), cov=c), places=11)

    def test_conditioning_matches_dense_joint_precision_in_arbitrary_order(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(7, 7))
        c = a @ a.T+np.eye(7)
        t, h = np.array([5, 2, 0]), np.array([6, 1, 4, 3])
        p = np.linalg.inv(c)
        expected_s = np.linalg.inv(p[np.ix_(h, h)])
        expected_a = -expected_s @ p[np.ix_(h, t)]
        gain, schur = conditional_channels(c, t, h)
        np.testing.assert_allclose(gain, expected_a, atol=1e-13)
        np.testing.assert_allclose(schur, expected_s, atol=1e-13)

    def test_training_uses_inverse_of_marginal_not_precision_subset(self):
        c = np.array([[1., 0.8], [0.8, 1.]])
        marginal = GaussianBlock(c[:1, :1], np.ones(1)).score(np.ones((1, 1)))["q"]
        wrong = np.linalg.inv(c)[0, 0]
        self.assertEqual(marginal, 1.)
        self.assertGreater(wrong, 2.)

    def test_diagonal_conditional_is_marginal(self):
        k = np.diag([1., 2., 4., 6.])
        gain, s = conditional_channels(k, np.array([1, 3]), np.array([2, 0]))
        np.testing.assert_array_equal(gain, 0)
        np.testing.assert_array_equal(s, np.diag([4., 1.]))

    def test_empty_conditioning_set_is_marginal(self):
        k = np.array([[2., 0.3], [0.3, 1.]])
        gain, s = conditional_channels(k, np.array([], dtype=int), np.array([1, 0]))
        self.assertEqual(gain.shape, (2, 0))
        np.testing.assert_array_equal(s, k[::-1, ::-1])

    def test_invalid_covariance_and_indices_fail_without_jitter(self):
        for c in (np.zeros((2, 2)), np.array([[1., 1], [1, 1]]),
                  np.array([[1., 2], [2, 1]]), np.array([[1., 0.2], [0.4, 1]]),
                  np.array([[float("nan")]])):
            with self.assertRaises(ValueError):
                GaussianBlock(c, np.ones(2))
        with self.assertRaises(ValueError):
            GaussianBlock(np.eye(2), np.array([1., -1.]))
        for t, h in (([0, 0], [1]), ([0], [0]), ([0], [3]), ([0.5], [1])):
            with self.assertRaises(ValueError):
                conditional_channels(np.eye(3), np.asarray(t), np.asarray(h))

    def test_AR_transform_covariance_and_diagonal_generation_limit(self):
        noise = dict(self.config["noise"], rho=0.4)
        k, s = channel_covariance(7, noise)
        basis = ar1_from_innovations(np.eye(7), noise["rho"])*s[:, None]
        np.testing.assert_allclose(basis @ basis.T, k, rtol=1e-13, atol=1e-20)
        zero = dict(noise, rho=0.)
        made = innovation_noise((7, 5, 5), zero, np.random.default_rng(8))
        z = np.random.default_rng(8).standard_normal((7, 25))
        _, s = channel_covariance(7, zero)
        expected = s[:, None]*pixel_scales(5, zero)[None, :]*z
        np.testing.assert_array_equal(made.reshape(7, 25), expected)

    def test_fixed_mask_excludes_and_partitions_all_measured_cells(self):
        p = FixedPartition.build((25, 21, 21), 0)
        m = p.masks()
        union = sum(v.astype(int) for v in m.values())
        c, y, x = np.indices(union.shape)
        np.testing.assert_array_equal(union, ((x+3*y) % 11 != 0).astype(int))
        self.assertFalse(m["train"][c % 3 == 0].any())
        self.assertFalse(m["train"][(x+2*y) % 3 == 0].any())

    def test_heldout_and_unmeasured_values_cannot_change_either_fit(self):
        prior = copy.deepcopy(self.prior)
        prior["instrument"].update(npix=9, nchannel=9, beam_half_width=2)
        ins, geometry = Instrument(**prior["instrument"]), Geometry(**prior["geometry"])
        data = forward_cube(prior["base_parameters"], geometry, ins, 6, 24)
        p = FixedPartition.build(data.shape, 0)
        bad = data.copy()
        bad[~p.masks()["train"]] = np.nan
        k, _ = channel_covariance(9, self.config["noise"])
        s = pixel_scales(9, self.config["noise"])
        for cov in (k, np.diag(np.diag(k))):
            a, ra = fit_motion(data, p, cov, s, prior, self.config, False, quadrature=(6, 24), max_nfev=6)
            b, rb = fit_motion(bad, p, cov, s, prior, self.config, False, quadrature=(6, 24), max_nfev=6)
            self.assertEqual(ra["parameters"], rb["parameters"])
            self.assertEqual(ra["training_marginal"], rb["training_marginal"])
            np.testing.assert_array_equal(a, b)

    def test_fresh_forecast_never_uses_old_noise_or_schur_covariance(self):
        shape = (9, 9, 9)
        p = FixedPartition.build(shape, 0)
        cov, _ = channel_covariance(9, self.config["noise"])
        s = pixel_scales(9, self.config["noise"])
        rng = np.random.default_rng(604)
        data = innovation_noise(shape, self.config["noise"], rng)
        fresh = [innovation_noise(shape, self.config["noise"], rng)]
        zero = np.zeros(shape)
        first = forecast_evaluation(zero, data, zero, fresh, p, cov, cov, s)
        modified = data.copy()
        modified[p.masks()["train"]] *= 100
        second = forecast_evaluation(zero, modified, zero, fresh, p, cov, cov, s)
        for split in first:
            self.assertEqual(first[split]["fresh_signal_true_marginal"], second[split]["fresh_signal_true_marginal"])
        self.assertNotEqual(first["heldout_channels"]["fresh_transferred_mean_q_per_cell"], second["heldout_channels"]["fresh_transferred_mean_q_per_cell"])
        c, px = p.blocks()["heldout_channels"]
        expected = GaussianBlock(cov[np.ix_(c, c)], s[px]).score(p.extract(fresh[0], "heldout_channels"))
        self.assertEqual(first["heldout_channels"]["fresh_signal_true_marginal"][0], expected)

    def test_independent_heldout_pixels_never_receive_a_correction(self):
        shape = (9, 9, 9)
        p = FixedPartition.build(shape, 1)
        cov, _ = channel_covariance(9, self.config["noise"])
        s = pixel_scales(9, self.config["noise"])
        data = innovation_noise(shape, self.config["noise"], np.random.default_rng(98))
        z = np.zeros(shape)
        result = forecast_evaluation(z, data, z, [data], p, cov, cov, s)
        for split in ("heldout_pixels", "heldout_joint"):
            self.assertEqual(result[split]["same_signal_assumed_marginal"], result[split]["same_conditional_assumed_distribution"])
            self.assertEqual(result[split]["noise_correction_size_true_marginal"], 0.)

    def test_linear_GLS_solution_matches_independent_normal_equations(self):
        rng = np.random.default_rng(430)
        a = rng.normal(size=(7, 7))
        k = a @ a.T+np.eye(7)
        x, y = rng.normal(size=(7, 3)), rng.normal(size=7)
        b = GaussianBlock(k, np.ones(3))
        x_white = b.whiten(x)
        y_white = GaussianBlock(k, np.ones(1)).whiten(y[:, None]).ravel()
        actual = np.linalg.lstsq(x_white, y_white, rcond=None)[0]
        inverse = np.linalg.inv(k)
        expected = np.linalg.solve(x.T @ inverse @ x, x.T @ inverse @ y)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_log_determinant_is_part_of_forecast_density(self):
        zero = np.zeros((3, 2))
        a = GaussianBlock(np.eye(3), np.ones(2)).score(zero)
        b = GaussianBlock(2*np.eye(3), np.ones(2)).score(zero)
        self.assertEqual(a["q"], b["q"])
        self.assertAlmostEqual(b["negative_log_likelihood"]-a["negative_log_likelihood"], 3*np.log(2), places=13)

    def test_input_covariance_is_copied_and_protected(self):
        source = np.eye(3)
        b = GaussianBlock(source, np.ones(2))
        source[0, 0] = -1
        self.assertEqual(b.covariance[0, 0], 1.)
        self.assertFalse(b.covariance.flags.writeable)

    def test_failed_controls_stop_before_study_response_generation(self):
        with tempfile.TemporaryDirectory(dir=runner.PRIVATE, prefix="test-gate-") as temporary:
            base = Path(temporary)
            with patch.object(runner, "REPORT", base / "reports"), \
                 patch.object(runner, "PRIVATE", base / "private"), \
                 patch.object(runner, "load_frozen", return_value=(self.config, self.prior, self.freeze)), \
                 patch.object(runner, "numerical_controls", return_value={"all_passed": False}), \
                 patch.object(runner, "direct_reference_cube", side_effect=AssertionError("opened study truth")) as generation:
                code = runner.execute("failed-control")
                self.assertEqual(code, 2)
                generation.assert_not_called()
                failed = json.loads((base / "reports/failed-control/failure.json").read_text())
                self.assertFalse(failed["synthetic_study_started"])

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=runner.PRIVATE, prefix="test-overwrite-") as temporary:
            base = Path(temporary)
            (base / "private/old").mkdir(parents=True)
            with patch.object(runner, "REPORT", base / "reports"), \
                 patch.object(runner, "PRIVATE", base / "private"), \
                 patch.object(runner, "load_frozen", return_value=(self.config, self.prior, self.freeze)):
                with self.assertRaises(FileExistsError):
                    runner.execute("old")
                self.assertFalse((base / "reports/old").exists())


if __name__ == "__main__":
    unittest.main()
