"""Execute an immutable NGC2976 conditional native-selection milestone on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import platform
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from astropy.io import fits
from scipy.integrate import quad
from scipy.ndimage import distance_transform_edt, minimum_filter
from scipy.signal import convolve2d
from threadpoolctl import threadpool_limits

from mond_atlas_native_selection import (
    FWHM_SIGMA, beam_from_history, beam_covariance, gaussian_kernel,
    convolve_spatial, select_runs, spectral_matrix, integrated_gaussian,
    robust_channels, covariance_diagnostics, source_templates, recovery, conditional_noise)
from mond_atlas_native_spectral import continuum_operator

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def controls(config, operator, provenance, native_cov, extra_cov):
    """Independent calculations with fixed tolerances, before image-value access."""
    rng = np.random.default_rng(381060)
    result = {}
    ctrl = config['controls']
    kernel = gaussian_kernel(extra_cov)
    plane = rng.normal(size=(37, 43))
    error = float(np.max(np.abs(convolve_spatial(plane, kernel)-convolve2d(plane, kernel, mode='same'))))
    result['direct_convolution_max_abs'] = error
    assert error < ctrl['convolution_direct_absolute_tolerance']
    n = operator.shape[1]
    x = np.linspace(-1, 1, n)
    design = np.column_stack([np.ones(n), x])
    cal = provenance['continuum_fit_parent_indices_zero_based']
    out = provenance['parent_channel_indices_zero_based']
    parent = rng.normal(size=(n, 15))
    reference = parent[out]-design[out]@np.linalg.lstsq(design[cal], parent[cal], rcond=None)[0]
    error = float(np.max(np.abs(operator@parent-reference)))
    result['separate_least_squares_max_abs'] = error
    result['polynomial_annihilation_max_abs'] = float(np.max(np.abs(operator@design)))
    assert max(error, result['polynomial_annihilation_max_abs']) < ctrl['continuum_reference_absolute_tolerance']
    for label, covariance in [('native', native_cov), ('extra', extra_cov)]:
        k = gaussian_kernel(covariance)
        r = len(k)//2
        grid = np.stack(np.mgrid[-r:r+1, -r:r+1], axis=-1)
        measured = np.einsum('yx,yxi,yxj->ij', k, grid, grid)
        error = float(np.linalg.norm(measured-covariance)/np.linalg.norm(covariance))
        result[label+'_beam_covariance_relative_error'] = error
        assert error < ctrl['beam_covariance_relative_tolerance']
        k6 = gaussian_kernel(covariance, 6)
        pad = (len(k6)-len(k))//2
        difference = float(np.abs(np.pad(k, pad)-k6).sum())
        result[label+'_kernel_5_vs_6_sigma_l1'] = difference
        assert difference < ctrl['kernel_truncation_5_vs_6_sigma_l1_tolerance']
    maximum_quad_error = 0
    for fwhm, width in itertools.product([1, 3, 6], [.5, 1]):
        grid = np.arange(-6, 7, dtype=float)
        analytic = integrated_gaussian(grid, .5, fwhm, width)
        numerical = np.array([quad(lambda z: np.exp(-4*np.log(2)*(z-.5)**2/fwhm**2),
                                   g-width/2, g+width/2)[0]/width for g in grid])
        maximum_quad_error = max(maximum_quad_error, float(np.max(np.abs(analytic-numerical))))
    result['spectral_source_independent_quadrature_max_abs'] = maximum_quad_error
    assert maximum_quad_error < ctrl['source_quadrature_absolute_tolerance']
    # Native pixel integration convergence: total of a compact 6-arcsec Gaussian.
    totals = []
    for refinement in [1, 2, 4]:
        step = 1.5/refinement
        xx = (np.arange(80*refinement)+.5)*step-60
        yy, xx = np.meshgrid(xx, xx)
        totals.append(float(np.exp(-4*np.log(2)*(xx**2+yy**2)/6**2).sum()*step**2))
    exact = np.pi*6**2/(4*np.log(2))
    result['spatial_source_pixel_refinement_integrals_arcsec2'] = totals
    result['spatial_source_integral_relative_errors'] = [abs(v-exact)/exact for v in totals]
    assert max(result['spatial_source_integral_relative_errors']) < ctrl['source_pixel_refinement_relative_tolerance']
    result['spectral_covariance_mc'] = {}
    for branch in config['spectral_branches']:
        h, _, _ = spectral_matrix(n, branch)
        transform = operator@h
        theoretical = transform@transform.T
        draws = transform@rng.normal(size=(h.shape[1], ctrl['spectral_covariance_mc_draws']))
        measured = np.cov(draws, bias=True)
        error = float(np.linalg.norm(measured-theoretical)/np.linalg.norm(theoretical))
        result['spectral_covariance_mc'][branch] = dict(relative_error=error, draws=draws.shape[1],
                                                        minimum_eigenvalue=float(np.linalg.eigvalsh(theoretical).min()))
        assert error < ctrl['spectral_covariance_mc_relative_tolerance']
    result['passed'] = True
    return result


def run(config_path, output, private):
    config = json.loads(config_path.read_text())
    if config['admission_disposition'] != 'SOURCE_BLOCKED' or config['observational_admission_permitted']:
        raise ValueError('conditional source/instrument development only')
    if (config['selection']['threshold_sigma'], config['selection']['consecutive_channels'],
        config['selection']['kernel_truncate_sigma']) != (2, 3, 5):
        raise ValueError('v1 implements only the frozen 2-sigma, 3-channel, 5-sigma-support operator')
    if output.exists() or private.exists():
        raise FileExistsError('new immutable result directories required')
    output.mkdir(parents=True)
    private.mkdir(parents=True)
    prior_dirs = ['mond-atlas-native-spectral-001', 'mond-atlas-preprocessing-replay-002',
                  'mond-atlas-smoothing-null-001', 'mond-atlas-mask-injection-001',
                  'mond-atlas-background-support-001', 'mond-atlas-emission-excluded-noise-001']
    bindings = [config_path, Path(__file__), ROOT/'scripts/mond_atlas_native_selection.py',
                ROOT/'tests/test_mond_atlas_native_selection.py', ROOT/'scripts/mond_atlas_native_spectral.py',
                ROOT/'scripts/mond_atlas_common.py', ROOT/config['native_history']]
    bindings += [ROOT/'work/gravity-first-principles'/name/'summary.json' for name in prior_dirs]
    acquisition = json.loads((ROOT/config['source_acquisition']).read_text())
    bindings += [ROOT/config['source_acquisition']]
    bindings += [ROOT/row['file'] for row in acquisition if row.get('file')]
    for key in ['cube', 'moment']:
        path = ROOT/config[key+'_path']
        if sha(path) != config[key+'_sha256']:
            raise ValueError('source hash mismatch: '+key)
        bindings.append(path)
    bound = {p.relative_to(ROOT).as_posix(): sha(p) for p in bindings}
    write_json(output/'prospective-bindings.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        configuration=config, bindings=bound, source_pixels_opened_for_new_selection=False,
        earlier_header_support_and_source_identity_audits=True, galaxy_motion_response_access=False))
    print('Bound configuration, sources and independent checks before new selection values.', flush=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_mond_atlas_native_selection.py')
    log = io.StringIO()
    test = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (output/'unit-tests.log').write_text(log.getvalue(), encoding='utf8')
    if not test.wasSuccessful():
        raise AssertionError('independent unit tests failed')
    history = json.loads((ROOT/config['native_history']).read_text())
    public_headers = [row for row in acquisition if row.get('file', '').endswith('archive-cube-header.bin')]
    if len(public_headers) != 1 or public_headers[0]['sha256'] != history['original_padded_header_sha256']:
        raise ValueError('public archive header identity not established')
    prov = history['provenance']
    if not prov['direct_channel_mapping']:
        raise ValueError('direct channel history required')
    out = prov['parent_channel_indices_zero_based']
    operator = continuum_operator(prov['parent_channel_count'], prov['continuum_fit_parent_indices_zero_based'],
                                  out, prov['polynomial_order'])
    header = fits.getheader(ROOT/config['cube_path'])
    moment_header = fits.getheader(ROOT/config['moment_path'])
    keys = ['NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
            'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2']
    if any(header.get(k) != moment_header.get(k) for k in keys):
        raise ValueError('moment/cube spatial grid mismatch')
    if (header['CTYPE1'], header['CTYPE2'], header['CTYPE3']) != ('RA---SIN', 'DEC--SIN', 'VELO-HEL'):
        raise ValueError('restricted declared WCS required')
    if any(header.get('CROTA'+str(i), 0) != 0 for i in [1, 2, 3]) or any(k.startswith(('CD1_', 'CD2_', 'PC1_', 'PC2_')) for k in header):
        raise ValueError('rotated or coupled WCS not supported by this pilot')
    beam = beam_from_history(header)
    dy, dx = header['CDELT2']*3600, header['CDELT1']*3600
    if not np.isclose(abs(dy), abs(dx)):
        raise ValueError('square angular pixels required')
    pixel = abs(dx)
    native_cov = beam_covariance(beam['major_arcsec'], beam['minor_arcsec'], beam['pa_deg'], dy, dx)
    target_cov = np.eye(2)*(config['selection']['target_fwhm_arcsec']/FWHM_SIGMA/pixel)**2
    extra_cov = target_cov-native_cov
    validation = controls(config, operator, prov, native_cov, extra_cov)
    validation['unit_tests_run'] = test.testsRun
    write_json(output/'controls.json', validation)
    native_kernel = gaussian_kernel(native_cov)
    extra_kernel = gaussian_kernel(extra_cov)
    margin = len(extra_kernel)//2
    shape = (header['NAXIS2'], header['NAXIS1'])
    yy, xx = np.indices(shape)
    north = (yy+1-header['CRPIX2'])*dy
    east = (xx+1-header['CRPIX1'])*dx
    radius = np.hypot(north, east)
    # Only published MOM0 geometry is inspected to freeze eligible positions.
    moment = fits.getdata(ROOT/config['moment_path']).squeeze()
    positive = np.isfinite(moment) & (moment > 0)
    distance = distance_transform_edt(~positive)*pixel
    finite_support = np.isfinite(moment)
    allowed = finite_support & (distance > config['support']['exclude_positive_moment_radius_arcsec'])
    allowed[:margin] = False; allowed[-margin:] = False
    allowed[:, :margin] = False; allowed[:, -margin:] = False
    inner, outer = config['support']['annulus_arcsec']
    annulus = (radius >= inner) & (radius <= outer)
    guard = config['support']['east_west_guard_arcsec']
    calibration = allowed & annulus & (east < -guard)
    validation_support = allowed & annulus & (east > guard)
    if min(calibration.sum(), validation_support.sum()) < config['support']['minimum_pixels_each_half']:
        raise ValueError('insufficient predeclared support')
    patch_radius = config['support']['patch_radius_pixels']
    size = 2*patch_radius+1
    eligible = minimum_filter((allowed & (east > guard)).astype(np.uint8), size=size, mode='constant') > 0
    eligible &= validation_support
    step = config['support']['center_lattice_step_pixels']
    candidates = np.argwhere(eligible & (yy % step == 0) & (xx % step == 0))
    if len(candidates) == 0:
        raise ValueError('no eligible fixed patch centers')
    chosen = np.linspace(0, len(candidates)-1, min(config['support']['maximum_empirical_positions'], len(candidates)), dtype=int)
    positions = candidates[chosen]
    write_json(output/'support.json', dict(calibration_pixels=int(calibration.sum()), validation_pixels=int(validation_support.sum()),
        moment_positive_pixels=int(positive.sum()), candidate_centers=len(candidates), selected_positions_yx_zero_based=positions.tolist(),
        geometry_only_positions=True, exact_native_line_free_support=False, local_projection='FITS SIN projection-plane angular coordinates',
        beam=beam, signed_pixel_arcsec_yx=[dy, dx], convolution_margin_pixels=margin))
    print(f'Controls passed; fixed {len(positions)} background locations, reading native cube.', flush=True)
    with fits.open(ROOT/config['cube_path'], memmap=True) as hdus:
        hdus.verify('exception')
        cube = hdus[0].data.squeeze().astype(np.float64)
        extension_summary = [dict(name=h.name, rows=int(h.header.get('NAXIS2', 0))) for h in hdus[1:]]
    if cube.shape != tuple(history['shape']) or not np.isfinite(cube).all():
        raise ValueError('fully finite native standard cube required')
    smoothed = np.stack([convolve_spatial(plane, extra_kernel) for plane in cube])
    med, sigma = robust_channels(smoothed, calibration)
    native_med, native_sigma = robust_channels(cube, calibration)
    smoothed -= med[:, None, None]
    cube -= native_med[:, None, None]
    mask = select_runs(smoothed, sigma)
    mask[:, :margin] = False; mask[:, -margin:] = False
    mask[:, :, :margin] = False; mask[:, :, -margin:] = False
    np.savez_compressed(private/'selection-and-support.npz', mask=mask, calibration=calibration,
                        validation=validation_support, sigma_jy_per_native_beam=sigma, median=med,
                        native_sigma=native_sigma, native_median=native_med)
    observed = {}
    for name, data in [('native', cube), ('detector30', smoothed)]:
        for region, support in [('west_calibration', calibration), ('east_validation', validation_support)]:
            diagnostic, covariance = covariance_diagnostics(data, support)
            observed[name+'_'+region] = diagnostic
            np.save(private/(name+'_'+region+'-covariance.npy'), covariance)
    projected_mask = mask.any(axis=0)
    joint = int((projected_mask & positive).sum())
    observed['selection'] = dict(selected_voxels=int(mask.sum()), selected_spatial_pixels=int(projected_mask.sum()),
        selected_validation_voxel_fraction=float(mask[:, validation_support].mean()),
        overlap_with_positive_moment_pixels=joint, moment_positive_pixels=int(positive.sum()),
        fraction_moment_support_covered=joint/int(positive.sum()),
        fraction_selected_spatial_support_outside_moment=float((projected_mask & ~positive).sum()/projected_mask.sum()),
        comparison_is_projected_support_only=True, publisher_mask_recovered=False)
    observed['cube_validation'] = dict(shape=list(cube.shape), finite_fraction=1.0, standard_unblanked=True,
        fits_structure_verified=True, fits_checksum_present='CHECKSUM' in header, extensions=extension_summary,
        header_sha256_matches_public_range=True, full_remote_cube_rehashed=False)
    channel_rows = [dict(stored_index=c, parent_index=out[c], velocity_radio_heliocentric_kms=(header['CRVAL3']+(c+1-header['CRPIX3'])*header['CDELT3'])/1000,
        detector_mad_jy_per_native_beam=float(sigma[c]), native_mad_jy_per_native_beam=float(native_sigma[c]),
        selected_pixels=int(mask[c].sum())) for c in range(len(cube))]
    write_csv(output/'channels.csv', channel_rows)
    write_json(output/'observed-diagnostics.json', observed)
    flux_factor = pixel**2/(np.pi*beam['major_arcsec']*beam['minor_arcsec']/(4*np.log(2)))*abs(header['CDELT3'])/1000
    inj = config['injections']
    scale = float(np.median(sigma))
    case_parameters = list(itertools.product(inj['intrinsic_spatial_fwhm_arcsec'],
        inj['intrinsic_spectral_fwhm_stored_channels'], inj['subchannel_phases'], inj['peak_detector_sigma']))
    empirical = []; synthetic = []; noise_rows = []; noiseless = []; branch_covariance = {}
    rng = np.random.default_rng(inj['seed'])
    for branch in config['spectral_branches']:
        h, _, _ = spectral_matrix(operator.shape[1], branch)
        covariance = operator@h@h.T@operator.T
        branch_covariance[branch] = covariance.tolist()
        templates = {}
        def evaluate(background_native, background_detector, rowbase, output_rows, local_sigma):
            base_mask = select_runs(background_detector, local_sigma)
            for case_index, (spatial, spectral, phase, peak) in enumerate(case_parameters):
                key = (spatial, spectral, phase)
                if key not in templates:
                    templates[key] = source_templates((size, size), pixel, spatial, spectral, phase,
                        out[inj['center_stored_channel']], branch, operator, out, native_kernel, extra_kernel)
                ns, ds, ps = templates[key]
                result = recovery(background_native, background_detector, ns, ds, ps, local_sigma,
                                  peak*scale, flux_factor, base_mask)
                output_rows.append(dict(branch=branch, case_index=case_index, spatial_fwhm_arcsec=spatial,
                    spectral_fwhm_stored_channels=spectral, subchannel_phase=phase, peak_detector_sigma=peak,
                    **rowbase, **result))
        zero = np.zeros((len(cube), size, size))
        evaluate(zero, zero, dict(realization='noiseless'), noiseless, sigma)
        for index, (y, x) in enumerate(positions):
            sl = np.s_[:, y-patch_radius:y+patch_radius+1, x-patch_radius:x+patch_radius+1]
            evaluate(cube[sl], smoothed[sl], dict(position=index, y=int(y), x=int(x)), empirical, sigma)
        predicted_sigma = scale*np.sqrt(np.diag(covariance)/np.diag(covariance).mean())
        for draw in range(inj['synthetic_draws_per_branch']):
            nb, db = conditional_noise(rng, (size, size), covariance, native_kernel, extra_kernel, scale)
            noise_rows.append(dict(branch=branch, draw=draw, detector_rms_over_scale=float(np.sqrt(np.mean(db*db))/scale),
                selected_voxel_fraction=float(select_runs(db, predicted_sigma).mean())))
            evaluate(nb, db, dict(draw=draw), synthetic, predicted_sigma)
        print(f'{branch}: {len(positions)*len(case_parameters)} actual-background and {inj["synthetic_draws_per_branch"]*len(case_parameters)} simulated trials.', flush=True)
    write_csv(output/'empirical-injections.csv', empirical)
    write_csv(output/'synthetic-injections.csv', synthetic)
    write_csv(output/'noiseless-controls.csv', noiseless)
    write_csv(output/'synthetic-noise.csv', noise_rows)
    write_json(output/'spectral-covariances.json', branch_covariance)
    case_rows = []
    for kind, rows in [('empirical', empirical), ('synthetic', synthetic)]:
        for branch, case_index in itertools.product(config['spectral_branches'], range(len(case_parameters))):
            subset = [r for r in rows if r['branch'] == branch and r['case_index'] == case_index]
            first = subset[0]
            r = {k:first[k] for k in ['branch', 'case_index', 'spatial_fwhm_arcsec', 'spectral_fwhm_stored_channels', 'subchannel_phase', 'peak_detector_sigma']}
            r.update(kind=kind, trials=len(subset))
            for metric in ['peak_selected', 'true_flux_fraction_retained', 'selected_noisy_flux_over_reference', 'paired_selected_flux_difference_over_reference']:
                values = np.array([v[metric] for v in subset], dtype=float)
                r[metric+'_mean'] = float(values.mean())
                r[metric+'_sd'] = float(values.std(ddof=1)) if len(values)>1 else 0.
            case_rows.append(r)
    write_csv(output/'case-summary.csv', case_rows)
    if any(sha(ROOT/p) != value for p, value in bound.items()):
        raise ValueError('a bound input changed during execution')
    summary = dict(status='EXECUTED_CONDITIONAL_NATIVE_SELECTION', admission_disposition='SOURCE_BLOCKED',
        galaxy=config['galaxy'], actual_cube_shape=list(cube.shape), unit_tests_passed=test.testsRun,
        analytic_numerical_controls_passed=validation['passed'], empirical_positions=len(positions),
        cases_per_branch=len(case_parameters), spectral_branches=config['spectral_branches'],
        actual_background_injection_trials=len(empirical), synthetic_injection_trials=len(synthetic),
        synthetic_noise_cubes=len(noise_rows), noiseless_controls=len(noiseless),
        detector_median_mad_jy_per_native_beam=scale, native_to_jy_kms_flux_factor=flux_factor,
        download_bytes=sum(r.get('bytes', r.get('bytes_read', 0)) for r in acquisition),
        no_gpu_used=True, runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        new_gravity_motion_scores=0, admitted_galaxy_cube_likelihoods=0,
        input_bindings_reverified=True,
        unresolved=['Exact publisher 3D mask and sigma estimator absent from linked release',
            'Online spectral taper and channel-decimation not certified by this header or object table',
            'Correlator passband, dirty beam, flags, visibility weights and nonlinear CLEAN not reconstructed',
            'Historical continuum channels and MOM0-screened background not certified line-free',
            'Actual-background injection positions overlap and are development-exposed; no population completeness inference',
            'No residual-scaling, primary-beam or absolute-flux likelihood calibration; injected flux is model-defined'],
        flux_aperture='Finite 81 by 81 native-pixel injection patch; no extrapolation of Gaussian tails or sky-integrated mass claim',
        empirical_is_conditional_on_fixed_observed_background=True,
        synthetic_is_conditional_on_declared_response_covariance=True)
    if summary['download_bytes'] > config['download_cap_bytes']:
        raise ValueError('download cap exceeded')
    write_json(output/'summary.json', summary)
    manifest = {p.relative_to(ROOT).as_posix():sha(p) for directory in [output, private] for p in sorted(directory.rglob('*')) if p.is_file()}
    write_json(output/'run-manifest.json', dict(files=manifest, note='Written once after execution; excludes itself and later human-readable report.'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/mond_atlas_native_selection_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private-output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.config.resolve(), args.output.resolve(), args.private_output.resolve())
