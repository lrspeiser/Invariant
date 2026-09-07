"""Fixed central-aperture working covariance and western-core sensitivity."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from mond_atlas_noise_scale_channel import geometry, transform
from mond_atlas_noise_mode_power import pool_power
from mond_atlas_noise_dc_regularization import candidates
from mond_atlas_native_covariance import regularized_covariance

R = Path(__file__).resolve().parents[1]
P = R/'work/gravity-first-principles/mond-atlas-aperture-covariance-001'
FROZEN = R/'work/gravity-first-principles/mond-atlas-noise-dc-regularization-001/run001/models-before-east.json'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def aperture_operator():
    a = np.zeros((24, 24))
    a[6:18, 6:18] = 1/144
    return a.ravel()


def contract(models):
    bands, u, _ = geometry(24, 24)
    weight = (aperture_operator()@u)**2
    terms = {b: float(weight[mask]@np.asarray(models[b]['power']))*np.asarray(models[b]['C'])
             for b, mask in bands.items()}
    return sum(terms.values()), terms


def fixed_refit(data, frozen):
    mean = data.mean(axis=(0, 1, 2))
    coeff = transform(data, mean)
    bands, _, _ = geometry(24, 24)
    coords = np.array([(y, x) for y in range(24) for x in range(24)])
    result = {}
    for band, mask in bands.items():
        recipe = frozen['base_models'][band]
        power, _ = pool_power(np.mean(coeff[:, mask]**2, axis=(0, 2)),
                              coords[mask], recipe['radius'])
        values = (coeff[:, mask]/np.sqrt(power)[None, :, None]).reshape(-1, 42)
        covariance = regularized_covariance(values, dict(kind='full', shrinkage=recipe['alpha']))
        result[band] = dict(power=power, C=covariance)
    dc = candidates(np.diag(result['DC']['C']), np.diag(result['low']['C']), len(data))
    result['DC']['C'] = np.diag(dc[frozen['selected']]['variance'])
    return mean, result


def manual_dense(models):
    # Independent cosine definition; no scipy DCT/band-geometry helper.
    pixel = np.arange(24)[:, None]
    mode = np.arange(24)[None, :]
    basis = np.sqrt(2/24)*np.cos(np.pi*(pixel+.5)*mode/24)
    basis[:, 0] = 1/np.sqrt(24)
    u = np.kron(basis, basis)
    r2 = np.array([y*y+x*x for y in range(24) for x in range(24)])
    masks = [r2==0, (r2>=1)&(r2<=9), (r2>=10)&(r2<=64), r2>64]
    a = np.zeros((24, 24)); a[6:18, 6:18] = 1/144
    result = np.zeros((42, 42))
    for name, mask in zip(('DC', 'low', 'middle', 'high'), masks):
        k = (u[:, mask]*np.asarray(models[name]['power']))@u[:, mask].T
        result += float(a.ravel()@k@a.ravel())*np.asarray(models[name]['C'])
    return result, float(np.max(abs(u.T@u-np.eye(576))))


def run():
    start = time.monotonic()
    out = P/'run001'; out.mkdir(exist_ok=False)
    try:
        cfg = json.loads((R/'configs/mond_atlas_aperture_noise_v1.json').read_text())
        source = R/cfg['input']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == cfg['input_sha256']
        deps = [Path(__file__), FROZEN, source, P/'PREFLIGHT.md',
                R/'scripts/mond_atlas_noise_scale_channel.py', R/'scripts/mond_atlas_noise_mode_power.py',
                R/'scripts/mond_atlas_noise_dc_regularization.py', R/'scripts/mond_atlas_native_covariance.py',
                R/'work/gravity-first-principles/mond-atlas-observation-admission-001/run004/frozen-geometric-apertures.csv']
        save(out/'bindings.json', {p.relative_to(R).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in deps})
        frozen = json.loads(FROZEN.read_text())
        models = {b: dict(power=np.asarray(m['power']), C=np.asarray(m['C'])) for b, m in frozen['base_models'].items()}
        models['DC']['C'] = np.diag(frozen['dc_candidates'][frozen['selected']]['variance'])
        covariance, terms = contract(models)
        dense, orthogonality = manual_dense(models)
        relative = np.linalg.norm(dense-covariance)/np.linalg.norm(covariance)
        assert relative < 1e-11 and orthogonality < 1e-12
        a = aperture_operator(); bands, u, _ = geometry(24, 24)
        white = np.sum((a@u)**2)
        common = (a@u[:, 0])**2*576
        assert abs(a.sum()-1)<1e-12 and abs(white-1/144)<1e-12 and abs(common-1)<1e-12
        cholesky = np.linalg.cholesky(covariance)
        save(out/'working-covariance.json', dict(mean=frozen['mean'], covariance=covariance.tolist(),
            cholesky=cholesky.tolist(), value_units='mJy/native_restoring_beam',
            covariance_units='(mJy/native_restoring_beam)^2', operation='central12x12 mean in24x24',
            channels=42, native_channel_order=True, selected_dc=frozen['selected'],
            band_trace_fraction={b:float(np.trace(v)/np.trace(covariance)) for b,v in terms.items()}))
        save(out/'pre-access-controls.json', dict(dense_relative_error=float(relative),
            orthogonality_max=orthogonality, white_variance=white, analytic_white=1/144,
            common_mode_variance=common, min_eigenvalue=float(np.linalg.eigvalsh(covariance).min())))
        with np.load(source) as packet:
            west = packet['training']
        assert west.shape == (29, 24, 24, 42) and np.isfinite(west).all()
        mean, refitted = fixed_refit(west, frozen)
        refit, _ = contract(refitted)
        error = np.linalg.norm(refit-covariance)/np.linalg.norm(covariance)
        assert error < 1e-10 and np.max(abs(mean-frozen['mean'])) < 1e-12
        variants, rows = [], []
        for index in range(len(west)):
            fitted_mean, fitted = fixed_refit(np.delete(west, index, axis=0), frozen)
            c, _ = contract(fitted)
            l = np.linalg.cholesky(c)
            residual = west[index, 6:18, 6:18].mean(axis=(0, 1))-fitted_mean
            whitened = np.linalg.solve(l, residual)
            variants.append(dict(omitted_western_index=index, mean=fitted_mean.tolist(), covariance=c.tolist()))
            rows.append(dict(omitted_western_index=index, q_per_channel=float(whitened@whitened/42),
                trace_ratio=float(np.trace(c)/np.trace(covariance)),
                relative_matrix_change=float(np.linalg.norm(c-covariance)/np.linalg.norm(covariance)),
                mean_change_rms_mJy_beam=float(np.sqrt(np.mean((fitted_mean-mean)**2))),
                minimum_eigenvalue=float(np.linalg.eigvalsh(c).min())))
        save(out/'western-leave-one-core-out.json', variants)
        with (out/'western-sensitivity.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        summary = dict(status='EXACT_MARGINAL_OF_FROZEN_WORKING_MODEL', observed_gravity_scores=0,
            source_region_reads=0, validation_array_reads=0, fixed_recipe=True, full_refit_relative_error=float(error),
            variants=len(variants), mean_loo_q_per_channel=float(np.mean([r['q_per_channel'] for r in rows])),
            trace_ratio_range=[min(r['trace_ratio'] for r in rows), max(r['trace_ratio'] for r in rows)],
            max_relative_covariance_change=max(r['relative_matrix_change'] for r in rows),
            max_mean_change_rms_mJy_beam=max(r['mean_change_rms_mJy_beam'] for r in rows),
            baseline_channel_sd_range=np.sqrt(np.diag(covariance))[[np.argmin(np.diag(covariance)), np.argmax(np.diag(covariance))]].tolist(),
            band_trace_fraction={b:float(np.trace(v)/np.trace(covariance)) for b,v in terms.items()},
            elapsed_seconds=time.monotonic()-start,
            limits='Descriptive working weights, not source-region or joint-aperture likelihood validation; omitted cores may be dependent.')
        save(out/'summary.json', summary)
        print(json.dumps(summary))
    except Exception as exc:
        save(out/'failure.json', dict(error=repr(exc)))
        raise


if __name__ == '__main__':
    run()
