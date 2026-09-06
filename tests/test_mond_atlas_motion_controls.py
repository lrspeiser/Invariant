"""Independent scientific and held-out-access controls for the motion milestone."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mond_atlas_motion_controls import (
    Geometry, Instrument, PARAMETERS, deposit_particles, direct_reference_cube,
    fit_model, fixed_splits, forward_cube, known_noise_sigma, numerical_controls,
    observe_particles, projected_particles, ring_nodes,
)
from run_mond_atlas_motion_controls import verify_freeze


class MotionControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/mond_atlas_motion_controls_v1.json").read_text())
        cls.base = cls.config["base_parameters"]
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_frozen_theory_only_and_no_observed_source(self):
        c, freeze, hashes = verify_freeze()
        self.assertFalse(c["observational_scoring_allowed"])
        self.assertIsNone(c["source_dataset"])
        self.assertEqual(freeze["config_sha256"], hashes["config_sha256"])

    def test_all_frozen_numerical_gates(self):
        result = numerical_controls(self.config)
        self.assertTrue(result["all_passed"], [r for r in result["controls"] if not r["passed"]])

    def test_radial_annuli_have_exact_analytic_flux(self):
        g = Geometry()
        for nr in (1, 3, 48):
            radius, _, weights = ring_nodes(g, nr, 32)
            self.assertAlmostEqual(weights.sum(), 1., places=14)
            cumulative = weights[radius < g.radius_max_kpc/2].sum()
            if nr == 48:
                f = lambda r: 1-(1+r/g.scale_kpc)*np.exp(-r/g.scale_kpc)
                self.assertAlmostEqual(cumulative, f(g.radius_max_kpc/2)/f(g.radius_max_kpc), places=14)

    def test_azimuthal_emission_changes_weights_not_velocity(self):
        a = projected_particles(self.base)
        b = projected_particles(dict(self.base, asymmetry=0.5))
        for i in range(4):
            np.testing.assert_array_equal(a[i], b[i])
        self.assertAlmostEqual(b[4].sum(), 1., places=14)
        self.assertGreater(np.max(np.abs(a[4]-b[4])), 0)

    def test_face_on_speed_and_radial_are_unidentifiable(self):
        a = dict(self.base, inclination_deg=0, radial_km_s=30)
        b = dict(a, rotation_km_s=160, radial_km_s=-35)
        np.testing.assert_array_equal(forward_cube(a), forward_cube(b))

    def test_streaming_has_correct_outward_sign(self):
        p = dict(self.base, rotation_km_s=0, inclination_deg=90, position_angle_deg=0, radial_km_s=20)
        _, _, depth, los, _ = projected_particles(p, nr=3, nphi=4)
        self.assertTrue(np.all(depth*los > 0))

    def test_line_band_is_not_renormalized(self):
        ins = Instrument(npix=7, beam_sigma_kpc=0, nchannel=1,
                         channel_min_km_s=-1, channel_max_km_s=1)
        cube, loss = observe_particles(np.array([0]), np.array([0]), np.array([0]), np.array([1]), 7, ins, True)
        self.assertAlmostEqual(cube.sum(), ndtr(1/7)-ndtr(-1/7), places=7)
        self.assertGreater(loss["spectral_loss"], 0.8)

    def test_tent_partition_of_unity_and_outside_loss(self):
        ins = Instrument(npix=9, beam_half_width=0)
        values = np.array([[0.3, 0.7]])
        cube = deposit_particles(np.array([0.21, -0.75]), np.array([-0.3, 0.5]), values, ins)
        self.assertAlmostEqual(cube.sum(), 1., places=14)
        outside = deposit_particles(np.array([100., -100.]), np.zeros(2), values, ins)
        self.assertEqual(outside.sum(), 0)

    def test_independent_reference_does_not_call_production_operators(self):
        with patch("mond_atlas_motion_controls.projected_particles", side_effect=AssertionError("shared geometry")), \
             patch("mond_atlas_motion_controls.project_emission", side_effect=AssertionError("shared CDF")), \
             patch("mond_atlas_motion_controls.spatial_beam", side_effect=AssertionError("shared beam")), \
             patch("mond_atlas_motion_controls.deposit_particles", side_effect=AssertionError("shared deposition")):
            cube = direct_reference_cube(self.base, Geometry(), Instrument(npix=9), 8, 24)
        self.assertTrue(np.isfinite(cube).all())
        self.assertGreater(cube.sum(), 0)

    def test_training_partition_does_not_expose_channels_or_pixels(self):
        splits = fixed_splits((25, 21, 21))
        counts = sum(mask.astype(int) for mask in splits.values())
        np.testing.assert_array_equal(counts, 1)
        c, y, x = np.indices(counts.shape)
        self.assertFalse(splits["train"][c % 3 == 0].any())
        self.assertFalse(splits["train"][(x+2*y) % 3 == 0].any())

    def test_changing_heldout_response_cannot_change_fit(self):
        ins = Instrument(npix=9, nchannel=9, beam_half_width=2)
        g = Geometry()
        data = forward_cube(self.base, g, ins, 6, 24)
        sigma = known_noise_sigma(data.shape, 0.0002)
        mask = fixed_splits(data.shape)["train"]
        altered = data.copy()
        altered[~mask] = np.nan  # No reading held-out cells even for finite checks.
        a, ca, ra = fit_model(data, sigma, mask, g, ins, self.config, False,
                             quadrature=(6, 24), max_nfev=8)
        b, cb, rb = fit_model(altered, sigma, mask, g, ins, self.config, False,
                             quadrature=(6, 24), max_nfev=8)
        self.assertEqual(a, b)
        self.assertEqual(ra["training_q"], rb["training_q"])
        np.testing.assert_array_equal(ca, cb)

    def test_diagonal_known_covariance_is_positive_and_fixed(self):
        sigma = known_noise_sigma((25, 21, 21), 0.00012)
        self.assertTrue((sigma > 0).all())
        self.assertGreater(sigma.max(), sigma.min())
        np.testing.assert_array_equal(sigma, known_noise_sigma(sigma.shape, 0.00012))

    def test_invalid_physical_inputs_fail_closed(self):
        for p in (dict(self.base, asymmetry=1), dict(self.base, dispersion_km_s=0),
                  dict(self.base, radial_km_s=float("nan"))):
            with self.assertRaises(ValueError):
                forward_cube(p)
        for kwargs in ({"radius_max_kpc": -1}, {"scale_kpc": float("inf")}):
            with self.assertRaises(ValueError):
                Geometry(**kwargs)
        with self.assertRaises(ValueError):
            Instrument(npix=0)
        with self.assertRaises(ValueError):
            Instrument(nchannel=2.5)


if __name__ == "__main__":
    unittest.main()
