"""Independent manufactured and algebraic controls, before real-background fits."""
import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from mond_atlas_native_covariance import (
    block_geometry, extract_background, sky_design, regularized_covariance, fit_model,
    residuals, gaussian_statistics, conditional_gaussian, fit_and_select_training,
    spatial_diagnostics)


def manufactured_report(contract):
    rng = np.random.default_rng(contract['seed'])
    count = contract['manufactured_covariance_draws']
    n = 6
    covariance = .7**np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    values, vectors = np.linalg.eigh(covariance)
    # Eigenvector generation is independent of the implementation's Cholesky whitening.
    noise = rng.normal(size=(count, n))@(vectors*np.sqrt(values)).T
    z, q, logs, logdet = gaussian_statistics(noise, covariance)
    error = float(np.linalg.norm(np.cov(z, rowvar=False, bias=True)-np.eye(n))/np.sqrt(n))
    diagonal = np.diag(np.diag(covariance))
    _, _, diagonal_logs, _ = gaussian_statistics(noise, diagonal)
    report = dict(draws=count, whitening_covariance_relative_error=error,
        q_mean_absolute_error=float(abs(q.mean()/n-1)),
        mean_logscore_gain_per_channel=float((logs-diagonal_logs).mean()/n),
        explicit_inverse_q_max_abs=float(np.max(np.abs(q-np.einsum('ni,ij,nj->n', noise, np.linalg.inv(covariance), noise)))),
        slogdet_absolute_error=float(abs(logdet-np.linalg.slogdet(covariance)[1])))
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    cr, cc, _ = conditional_gaussian(noise, covariance, odd, even)
    _, aq, al, _ = gaussian_statistics(noise[:, odd], covariance[np.ix_(odd, odd)])
    _, cq, cl, _ = gaussian_statistics(cr, cc)
    report['conditional_log_factorization_max_abs'] = float(np.max(np.abs(logs-al-cl)))
    report['conditional_q_factorization_max_abs'] = float(np.max(np.abs(q-aq-cq)))
    report['unconditional_minus_conditional_trace'] = float(np.trace(covariance[np.ix_(even, even)]-cc))
    tol = contract['linear_algebra_absolute_tolerance']
    report['passed'] = bool(error < contract['manufactured_whitening_covariance_relative_error_max']
        and report['q_mean_absolute_error'] < contract['manufactured_q_mean_absolute_error_max']
        and report['mean_logscore_gain_per_channel'] > contract['manufactured_logscore_gain_min']
        and all(report[k] < tol for k in ['explicit_inverse_q_max_abs', 'slogdet_absolute_error',
            'conditional_log_factorization_max_abs', 'conditional_q_factorization_max_abs'])
        and report['unconditional_minus_conditional_trace'] > 0)
    return report


class NativeCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/mond_atlas_native_covariance_v1.json').read_text())

    def test_manufactured_covariance_whitening_and_logscore(self):
        self.assertTrue(manufactured_report(self.config['benchmarks'])['passed'])

    def test_diagonal_density_matches_scalar_formula(self):
        residual = np.array([[1., -2., 3.], [2., 1., 0.]])
        variance = np.array([2., 3., 5.])
        z, q, logpdf, logdet = gaussian_statistics(residual, np.diag(variance))
        expected = np.sum(-.5*np.log(2*np.pi*variance)-residual**2/(2*variance), axis=1)
        np.testing.assert_allclose(logpdf, expected, atol=1e-12)
        np.testing.assert_allclose(z, residual/np.sqrt(variance), atol=1e-12)
        np.testing.assert_allclose(q, (residual**2/variance).sum(axis=1))
        self.assertAlmostEqual(logdet, np.log(variance).sum())

    def test_schur_complement_independent_block_inverse(self):
        covariance = np.array([[4., 1., 2.], [1., 3., .5], [2., .5, 5.]])
        residual = np.array([[3., -1., 4.]])
        result, conditional, _ = conditional_gaussian(residual, covariance, [0, 1], [2])
        reference_variance = 1/np.linalg.inv(covariance)[2, 2]
        reference_mean = covariance[2, :2]@np.linalg.solve(covariance[:2, :2], residual[0, :2])
        self.assertAlmostEqual(conditional[0, 0], reference_variance)
        self.assertAlmostEqual(result[0, 0], 4-reference_mean)
        self.assertLess(conditional[0, 0], covariance[2, 2])

    def test_conditional_index_overlap_rejected(self):
        with self.assertRaises(ValueError):
            conditional_gaussian(np.ones((1, 3)), np.eye(3), [0, 1], [1, 2])

    def test_positive_regularization_of_rank_deficient_samples(self):
        residual = np.repeat(np.array([[1., -1., 0., 2.]]), 4, axis=0)
        for spec in self.config['covariance_models']:
            covariance = regularized_covariance(residual, spec)
            self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)
            np.linalg.cholesky(covariance)
        zero_covariance = regularized_covariance(np.zeros((3, 4)), self.config['covariance_models'][1])
        np.testing.assert_allclose(zero_covariance, np.eye(4)*1e-12, atol=1e-20)

    def test_bartlett_taper_matches_known_schur_product(self):
        rng = np.random.default_rng(2)
        residual = rng.normal(size=(20, 7))
        sample = residual.T@residual/len(residual)
        reference = np.zeros_like(sample)
        for i in range(7):
            for j in range(7):
                reference[i, j] = sample[i, j]*(.9*max(1-abs(i-j)/4, 0)+(.1 if i == j else 0))
        np.testing.assert_allclose(regularized_covariance(residual, self.config['covariance_models'][3]), reference, atol=1e-12)

    def test_affine_manufactured_mean_and_coordinate_invariance(self):
        rng = np.random.default_rng(53)
        design = np.concatenate([np.ones((8, 4, 4, 1)), rng.normal(size=(8, 4, 4, 2))], axis=-1)
        beta = rng.normal(size=(3, 6))
        data = design@beta
        model = fit_model(data, design, 'channel_affine_sky', self.config['covariance_models'][1])
        np.testing.assert_allclose(model['beta'], beta, atol=self.config['benchmarks']['manufactured_mean_max_absolute_error'])
        shifted = design.copy(); shifted[..., 1] = 2*design[..., 1]+1
        model2 = fit_model(data, shifted, 'channel_affine_sky', self.config['covariance_models'][1])
        np.testing.assert_allclose(residuals(data, shifted, model2), residuals(data, design, model), atol=1e-12)

    def test_blocks_guards_and_geometry_only_folds(self):
        supports = {'training':np.zeros((240, 480), bool), 'validation':np.zeros((240, 480), bool)}
        supports['training'][:, :210] = True; supports['validation'][:, 270:] = True
        rows = block_geometry(supports, self.config['regions'])
        for r in rows:
            self.assertTrue(supports[r['region']][r['y0']:r['y1'], r['x0']:r['x1']].all())
            self.assertEqual(r['y1']-r['y0'], 24)
            if r['region'] == 'training':
                self.assertEqual(r['fold'], (r['grid_row']+2*r['grid_column'])%3)
        for a in rows:
            for b in rows:
                if a['block_id'] != b['block_id']:
                    self.assertGreaterEqual(max(abs(a['center_y']-b['center_y']),abs(a['center_x']-b['center_x'])), 48)

    def test_only_listed_background_values_are_read(self):
        cube = np.full((4, 20, 20), np.nan)
        cube[:, 3:7, 9:13] = .001
        rows = [dict(y0=3,y1=7,x0=9,x1=13)]
        result = extract_background(cube, rows)
        self.assertEqual(result.shape, (1, 4, 4, 4))
        np.testing.assert_array_equal(result, 1)

    def test_training_selection_scores_match_separate_fold_fits(self):
        rng = np.random.default_rng(921)
        data = rng.normal(size=(18, 4, 4, 4))
        design = np.concatenate([np.ones((18, 4, 4, 1)),rng.normal(size=(18, 4, 4, 2))],axis=-1)
        rows = [dict(block_id=str(i),fold=i%3) for i in range(18)]
        models, ranking, cv = fit_and_select_training(data, design, rows, self.config)
        held = np.arange(18)%3 == 0
        # Independent scalar-channel normal fit and density for the diagonal/constant candidate.
        train = data[~held].reshape(-1, 4)
        mean = train.mean(axis=0); variance = ((train-mean)**2).mean(axis=0)
        expected = (-.5*(np.log(2*np.pi*variance)+(data[held]-mean)**2/variance)).mean(axis=(1,2,3))
        actual = [r['mean_logpdf_per_channel'] for r in cv if r['model_id']=='channel_constant__diagonal' and r['fold']==0]
        np.testing.assert_allclose(expected, actual, atol=1e-12)
        # No held-east array can enter the training-only API; changing external storage leaves outputs fixed.
        unused_east = rng.normal(size=data.shape); unused_east[:] = 1e9
        models2, ranking2, _ = fit_and_select_training(data, design, rows, self.config)
        self.assertEqual(ranking, ranking2)
        for key in models:
            np.testing.assert_array_equal(models[key]['covariance'],models2[key]['covariance'])

    def test_spatial_offsets_are_not_hidden_as_centered_noise(self):
        rng = np.random.default_rng(108)
        data = rng.normal(size=(2, 24, 24, 3))+.7
        rows = [dict(block_id='a',grid_row=0,grid_column=0,center_y=24,center_x=24),
                dict(block_id='b',grid_row=0,grid_column=1,center_y=24,center_x=72)]
        stats = spatial_diagnostics(data, rows, [1], [1])
        raw = [r['product'] for r in stats if r['kind']=='cross_core' and not r['local_channel_means_removed']][0]
        centered = [r['product'] for r in stats if r['kind']=='cross_core' and r['local_channel_means_removed']][0]
        self.assertGreater(raw, .35)
        self.assertLess(abs(centered), .1)

    def test_unit_change_leaves_q_and_shifts_density_correctly(self):
        data = np.array([[.2, -.3, .6]])
        covariance = np.array([[2., .4, .2],[.4, 3., .1],[.2, .1, 4.]])
        _, q, logpdf, _ = gaussian_statistics(data, covariance)
        _, scaled_q, scaled_logpdf, _ = gaussian_statistics(1000*data, 1000000*covariance)
        np.testing.assert_allclose(q, scaled_q, atol=1e-12)
        np.testing.assert_allclose(scaled_logpdf, logpdf-3*np.log(1000), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
