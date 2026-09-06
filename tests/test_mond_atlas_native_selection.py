"""Independent numerical and failure controls for conditional native selection."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.signal import convolve2d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_native_selection import (
    beam_covariance, gaussian_kernel, convolve_spatial, select_runs,
    spectral_matrix, integrated_gaussian, recovery, FWHM_SIGMA)


class NativeSelectionTests(unittest.TestCase):
    def test_nonwrapping_runs_against_pixel_scan(self):
        rng = np.random.default_rng(406)
        cube = rng.normal(size=(11, 5, 6)) + 1.7
        cube[:, 0, 0] = [3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3]
        cube[5, 2, 2] = np.nan
        expected = np.zeros(cube.shape, bool)
        for y in range(5):
            for x in range(6):
                for channel in range(9):
                    if all(cube[channel+i, y, x] > 2 for i in range(3)):
                        expected[channel:channel+3, y, x] = True
        np.testing.assert_array_equal(select_runs(cube, 1), expected)
        self.assertFalse(expected[:, 0, 0].any())

    def test_threshold_equality_and_channel_sigma(self):
        cube = np.ones((5, 2, 2))*2
        self.assertFalse(select_runs(cube, 1).any())
        self.assertFalse(select_runs(cube, [1, .5, 1, .5, 1]).any())
        self.assertTrue(select_runs(cube, [.5]*5).all())
        with self.assertRaises(ValueError):
            select_runs(cube, 0)

    def test_elliptical_fft_against_direct(self):
        rng = np.random.default_rng(73)
        plane = rng.normal(size=(39, 47))
        cov = beam_covariance(7.4, 6.4, 71.8, 1.5, -1.5)
        k = gaussian_kernel(cov)
        np.testing.assert_allclose(convolve_spatial(plane, k), convolve2d(plane, k, mode='same'), atol=1e-11)

    def test_beam_direction_and_conservation(self):
        cov = beam_covariance(12, 5, 25, 1.5, -1.5)
        self.assertLess(cov[0, 1], 0)
        k = gaussian_kernel(cov)
        r = len(k)//2
        grid = np.stack(np.mgrid[-r:r+1, -r:r+1], axis=-1)
        measured = np.einsum('yx,yxi,yxj->ij', k, grid, grid)
        np.testing.assert_allclose(measured, cov, rtol=0.0002, atol=0.0001)
        self.assertAlmostEqual(k.sum(), 1, places=14)
        self.assertAlmostEqual(convolve_spatial(np.pad([[1.]], 60), k).sum(), 1, places=12)

    def test_beam_addition_and_axis_reflection(self):
        native = beam_covariance(7.407, 6.42384, 71.79, 1.5, -1.5)
        target = np.eye(2)*(30/FWHM_SIGMA/1.5)**2
        k = gaussian_kernel(target-native)
        self.assertGreater(np.linalg.eigvalsh(target-native).min(), 0)
        reflected = gaussian_kernel(beam_covariance(7.407, 6.42384, 71.79, 1.5, 1.5))
        np.testing.assert_allclose(reflected, gaussian_kernel(native)[:, ::-1], atol=1e-15)
        self.assertAlmostEqual(k.sum(), 1)

    def test_hanning_covariance_analytic_lags(self):
        for branch, lag1, lag2 in [('boxcar_independent', 0, 0),
                                   ('boxcar_hanning_full', 2/3, 1/6),
                                   ('boxcar_hanning_decimated', 1/6, 0)]:
            h, _, _ = spectral_matrix(20, branch)
            covariance = h@h.T
            correlation = covariance/covariance[0, 0]
            np.testing.assert_allclose(np.diag(correlation, 1), lag1, atol=1e-15)
            np.testing.assert_allclose(np.diag(correlation, 2), lag2, atol=1e-15)
            np.testing.assert_allclose(h.sum(axis=1), 1)

    def test_integrated_source_against_independent_quadrature(self):
        for fwhm in [1, 3, 6]:
            for width in [.5, 1]:
                grid = np.arange(-4, 5, dtype=float)
                computed = integrated_gaussian(grid, .5, fwhm, width)
                reference = [quad(lambda x: np.exp(-4*np.log(2)*(x-.5)**2/fwhm**2),
                                  g-width/2, g+width/2)[0]/width for g in grid]
                np.testing.assert_allclose(computed, reference, atol=1e-12)

    def test_flux_unit_conversion_and_selection_bias_separated(self):
        source = np.ones((3, 2, 2))*4
        zero = np.zeros_like(source)
        result = recovery(zero, zero, source, source, source, 1, 1, .2)
        self.assertAlmostEqual(result['reference_flux_jy_kms'], 9.6)
        self.assertEqual(result['true_flux_fraction_retained'], 1)
        self.assertEqual(result['paired_selected_flux_difference_over_reference'], 1)
        positive_noise = np.ones_like(source)
        result = recovery(positive_noise, positive_noise, source, source, source, 1, 1, .2)
        self.assertEqual(result['true_flux_fraction_retained'], 1)
        self.assertEqual(result['selected_noisy_flux_over_reference'], 1.25)

    def test_zero_amplitude_background_and_mask_support(self):
        cube = np.ones((4, 3, 3))*4
        support = np.eye(3, dtype=bool)
        np.testing.assert_array_equal(select_runs(cube, 1, support=support), np.broadcast_to(support, cube.shape))
        with self.assertRaises(ValueError):
            gaussian_kernel(np.diag([-1, 2]))


if __name__ == '__main__':
    unittest.main()
