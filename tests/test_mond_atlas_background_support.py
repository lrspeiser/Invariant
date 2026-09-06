import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_background_support import block_fraction, dilate_disk, segment_pairs, spectral_diagnostics


class BackgroundSupportTests(unittest.TestCase):
    def test_dilation_against_integer_distance(self):
        a = np.zeros((9, 13), bool); a[0, 0] = a[4, 7] = a[8, 12] = True
        yy, xx = np.indices(a.shape); points = np.argwhere(a)
        for radius in (0, 1, 3, 15):
            expected = np.any((yy[..., None]-points[:, 0])**2+(xx[..., None]-points[:, 1])**2 <= radius**2, axis=-1)
            np.testing.assert_array_equal(dilate_disk(a, radius), expected)

    def test_block_support_conservation(self):
        a = np.random.default_rng(123).random((80, 104)) < .17
        fraction = block_fraction(a, 8)
        self.assertLess(abs(fraction.sum()*64-a.sum()), 1e-12)
        self.assertLess(abs(fraction.mean()-a.mean()), 1e-12)
        with self.assertRaises(ValueError): block_fraction(a, 3)

    def test_gapped_spectral_segments(self):
        left, right = segment_pairs([(0, 4), (16, 20)], 1, 20)
        self.assertEqual(list(zip(left, right)), [(0,1), (1,2), (2,3), (16,17), (17,18), (18,19)])
        self.assertEqual(len(segment_pairs([(0, 4), (16, 20)], 6, 20)[0]), 0)
        with self.assertRaises(ValueError): segment_pairs([(0, 5), (4, 9)], 1, 20)

    def test_excluded_values_and_calibration_protection(self):
        rng = np.random.default_rng(823); cube = rng.normal(size=(40, 12, 12))
        cal = np.zeros((12, 12), bool); test = cal.copy(); cal[:3, :] = True; test[9:, :] = True
        baseline = spectral_diagnostics(cube, cal, test, .15, [1, 2, 3, 6])
        changed = cube.copy(); changed[:, ~(cal|test)] = np.nan
        self.assertEqual(baseline, spectral_diagnostics(changed, cal, test, .15, [1, 2, 3, 6]))
        changed[:, test] *= 100
        after = spectral_diagnostics(changed, cal, test, .15, [1, 2, 3, 6])
        self.assertEqual(baseline['calibration'], after['calibration'])
        self.assertNotEqual(baseline['validation'], after['validation'])


if __name__ == '__main__': unittest.main()
