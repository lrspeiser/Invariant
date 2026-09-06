"""Restricted background mean/covariance models and conditional Gaussian algebra.

Scores are per-pixel spectral composite scores. No spatial independence, pure
noise, dirty-beam model or observational cube-likelihood admission is implied.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular, cho_solve


def extract_background(cube, rows):
    """Read only listed cores from a spectral,y,x array; retain native pixels."""
    values = []
    for row in rows:
        patch = cube[:, row['y0']:row['y1'], row['x0']:row['x1']]
        values.append(np.asarray(patch, dtype=float).transpose(1, 2, 0)*1000)
    result = np.stack(values)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite values inside declared background cores')
    return result


def block_geometry(supports, contract):
    side = contract['block_core_side_native_pixels']
    step = contract['block_lattice_step_native_pixels']
    origin = contract['block_lattice_center_origin_native_pixels']
    if side % 2 or step <= side or origin < side//2:
        raise ValueError('even disjoint guarded cores required')
    shape = next(iter(supports.values())).shape
    if any(s.shape != shape for s in supports.values()):
        raise ValueError('inconsistent supports')
    if np.any(supports['training'] & supports['validation']):
        raise ValueError('training and validation overlap')
    rows = []
    for j, y in enumerate(range(origin, shape[0], step)):
        for i, x in enumerate(range(origin, shape[1], step)):
            y0, y1, x0, x1 = y-side//2, y+side//2, x-side//2, x+side//2
            if y1 > shape[0] or x1 > shape[1]:
                continue
            for region, support in supports.items():
                if support[y0:y1, x0:x1].all():
                    rows.append(dict(block_id=f'{region}_r{j:02d}_c{i:02d}', region=region,
                        grid_row=j, grid_column=i, center_y=y, center_x=x,
                        y0=y0, y1=y1, x0=x0, x1=x1,
                        fold=(j+2*i) % contract['inner_training_folds'] if region == 'training' else -1))
    return rows


def sky_design(rows, header):
    result = []
    for row in rows:
        yy, xx = np.mgrid[row['y0']:row['y1'], row['x0']:row['x1']]
        east = (xx+1-header['CRPIX1'])*header['CDELT1']*3600/600
        north = (yy+1-header['CRPIX2'])*header['CDELT2']*3600/600
        result.append(np.stack([np.ones_like(east), east, north], axis=-1))
    return np.stack(result)


def regularized_covariance(residual, spec):
    residual = np.asarray(residual, float)
    if residual.ndim != 2 or not np.isfinite(residual).all() or len(residual) < 2:
        raise ValueError('finite samples by channels required')
    sample = residual.T@residual/len(residual)
    diagonal = np.diag(sample).copy()
    floor = max(1e-12, 1e-8*float(np.median(diagonal)))
    sample += np.diag(np.maximum(diagonal, floor)-diagonal)
    diagonal = np.diag(sample).copy()
    kind, alpha = spec['kind'], spec['shrinkage']
    if not 0 < alpha <= 1:
        raise ValueError('positive diagonal shrinkage required')
    if kind == 'diagonal':
        covariance = np.diag(diagonal)
    elif kind in ('full', 'bartlett'):
        if kind == 'bartlett':
            lag = np.abs(np.subtract.outer(np.arange(len(diagonal)), np.arange(len(diagonal))))
            sample = sample*np.maximum(1-lag/(spec['max_lag']+1), 0)
        covariance = (1-alpha)*sample+alpha*np.diag(diagonal)
    else:
        raise ValueError('undeclared covariance family')
    covariance = (covariance+covariance.T)/2
    np.linalg.cholesky(covariance)
    return covariance


def fit_model(data, design, mean_kind, covariance_spec):
    data, design = np.asarray(data, float), np.asarray(design, float)
    if data.shape[:-1] != design.shape[:-1] or data.ndim != 4 or not np.isfinite(data).all():
        raise ValueError('matching block y x channel and design arrays required')
    if mean_kind not in ('channel_constant', 'channel_affine_sky'):
        raise ValueError('undeclared mean family')
    p = 1 if mean_kind == 'channel_constant' else 3
    x = design[..., :p].reshape(-1, p)
    y = data.reshape(-1, data.shape[-1])
    beta, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if rank != p:
        raise ValueError('mean design rank deficient')
    residual = y-x@beta
    covariance = regularized_covariance(residual, covariance_spec)
    return dict(mean_kind=mean_kind, covariance_id=covariance_spec['id'], beta=beta,
                covariance=covariance, training_blocks=len(data), design_rank=int(rank),
                design_condition=float(singular.max()/singular.min()))


def residuals(data, design, model):
    return np.asarray(data)-np.asarray(design)[..., :len(model['beta'])]@model['beta']


def gaussian_statistics(residual, covariance):
    residual = np.asarray(residual, float)
    covariance = np.asarray(covariance, float)
    n = residual.shape[-1]
    if covariance.shape != (n, n) or not np.isfinite(residual).all():
        raise ValueError('finite matching residual and covariance required')
    factor = np.linalg.cholesky(covariance)
    z = solve_triangular(factor, residual.reshape(-1, n).T, lower=True, check_finite=False).T.reshape(residual.shape)
    q = np.sum(z*z, axis=-1)
    logdet = float(2*np.log(np.diag(factor)).sum())
    logpdf = -.5*(q+logdet+n*np.log(2*np.pi))
    return z, q, logpdf, logdet


def conditional_gaussian(residual, covariance, observed, predicted):
    observed, predicted = np.asarray(observed, int), np.asarray(predicted, int)
    if (len(np.unique(np.r_[observed, predicted])) != len(observed)+len(predicted)
        or len(observed) == 0 or len(predicted) == 0):
        raise ValueError('nonempty disjoint channel indices required')
    aa = covariance[np.ix_(observed, observed)]
    ba = covariance[np.ix_(predicted, observed)]
    bb = covariance[np.ix_(predicted, predicted)]
    factor = np.linalg.cholesky(aa)
    gain = cho_solve((factor, True), ba.T).T
    conditional = bb-gain@ba.T
    conditional = (conditional+conditional.T)/2
    np.linalg.cholesky(conditional)
    return residual[..., predicted]-residual[..., observed]@gain.T, conditional, gain


def per_block_scores(data, design, model):
    _, q, logpdf, _ = gaussian_statistics(residuals(data, design, model), model['covariance'])
    return q.mean(axis=(1, 2))/data.shape[-1], logpdf.mean(axis=(1, 2))/data.shape[-1]


def fit_and_select_training(data, design, rows, config):
    """Only western training arrays enter this API; validation is not an argument."""
    specs = [(m, c) for m in config['mean_models'] for c in config['covariance_models']]
    fold_ids = np.array([r['fold'] for r in rows])
    cv, ranking, models = [], [], {}
    for ordinal, (mean, covariance) in enumerate(specs):
        model_id = mean+'__'+covariance['id']
        held_scores = []
        for fold in range(config['regions']['inner_training_folds']):
            train, held = fold_ids != fold, fold_ids == fold
            if train.sum() < config['regions']['minimum_inner_fit_blocks'] or held.sum() < config['regions']['minimum_inner_validation_blocks']:
                raise ValueError('insufficient geometry-defined inner split')
            fitted = fit_model(data[train], design[train], mean, covariance)
            q, scores = per_block_scores(data[held], design[held], fitted)
            for block_index, qn, score in zip(np.flatnonzero(held), q, scores):
                cv.append(dict(model_id=model_id, fold=fold, block_id=rows[block_index]['block_id'],
                    q_over_n=float(qn), mean_logpdf_per_channel=float(score)))
            held_scores.extend(scores.tolist())
        ranking.append(dict(model_id=model_id, declaration_order=ordinal,
            training_cv_mean_logpdf_per_channel=float(np.mean(held_scores)), held_western_blocks=len(held_scores)))
        models[model_id] = fit_model(data, design, mean, covariance)
    ranking.sort(key=lambda r: (-r['training_cv_mean_logpdf_per_channel'], r['declaration_order']))
    return models, ranking, cv


def product_diagnostic(a, b):
    aa, bb = np.asarray(a).ravel(), np.asarray(b).ravel()
    denom = np.sqrt(np.mean(aa*aa)*np.mean(bb*bb))
    ac, bc = aa-aa.mean(), bb-bb.mean()
    pearson_den = np.sqrt(np.mean(ac*ac)*np.mean(bc*bc))
    return dict(product=float(np.mean(aa*bb)),
                normalized_product=float(np.mean(aa*bb)/denom) if denom else 0.,
                pearson=float(np.mean(ac*bc)/pearson_den) if pearson_den else 0.)


def spatial_diagnostics(z, rows, lags, lattice_lags):
    """Products retain block channel offsets; centered versions explicitly remove them."""
    result = []
    local_means = z.mean(axis=(1, 2), keepdims=True)
    for centered in (False, True):
        values = z-local_means if centered else z
        for block_index, row in enumerate(rows):
            v = values[block_index]
            for lag in lags:
                for axis in ('y', 'x'):
                    a, b = (v[:-lag], v[lag:]) if axis == 'y' else (v[:, :-lag], v[:, lag:])
                    result.append(dict(block_a=row['block_id'], block_b=row['block_id'],
                        kind='within_core', axis=axis, lag_native_pixels=lag,
                        local_channel_means_removed=centered, **product_diagnostic(a, b)))
        lookup = {(r['grid_row'], r['grid_column']):i for i, r in enumerate(rows)}
        for i, row in enumerate(rows):
            for steps in lattice_lags:
                for axis, direction in [('y', (steps, 0)), ('x', (0, steps))]:
                    target = (row['grid_row']+direction[0], row['grid_column']+direction[1])
                    if target in lookup:
                        j = lookup[target]
                        separation = rows[j]['center_y']-row['center_y'] if axis == 'y' else rows[j]['center_x']-row['center_x']
                        result.append(dict(block_a=row['block_id'], block_b=rows[j]['block_id'],
                            kind='cross_core', axis=axis, lag_native_pixels=separation,
                            local_channel_means_removed=centered,
                            **product_diagnostic(values[i], values[j])))
    return result


def summarize_model(data, design, rows, model, config):
    residual = residuals(data, design, model)
    covariance = model['covariance']
    z, q, logpdf, logdet = gaussian_statistics(residual, covariance)
    observed, predicted = np.arange(1, data.shape[-1], 2), np.arange(0, data.shape[-1], 2)
    cr, cc, _ = conditional_gaussian(residual, covariance, observed, predicted)
    cz, cq, clog, _ = gaussian_statistics(cr, cc)
    _, uq, ulog, _ = gaussian_statistics(residual[..., predicted], covariance[np.ix_(predicted, predicted)])
    mean_prediction = data-residual
    block_rows = []
    for i, row in enumerate(rows):
        block_rows.append(dict(block_id=row['block_id'],
            q_over_n=float(q[i].mean()/data.shape[-1]), logpdf_per_channel=float(logpdf[i].mean()/data.shape[-1]),
            conditional_even_q_over_n=float(cq[i].mean()/len(predicted)), conditional_even_logpdf_per_channel=float(clog[i].mean()/len(predicted)),
            marginal_even_q_over_n=float(uq[i].mean()/len(predicted)), marginal_even_logpdf_per_channel=float(ulog[i].mean()/len(predicted)),
            whitened_mean=float(z[i].mean()), whitened_channel_mean_rms=float(np.sqrt(np.mean(z[i].mean(axis=(0, 1))**2))),
            whitened_within_block_variance=float(z[i].var(axis=(0, 1)).mean()),
            positive_3sigma_fraction=float((z[i] > 3).mean()), negative_3sigma_fraction=float((z[i] < -3).mean()),
            absolute_5sigma_fraction=float((np.abs(z[i]) > 5).mean()),
            predicted_channel_mean_rms_mjy=float(np.sqrt(np.mean(mean_prediction[i].mean(axis=(0, 1))**2))),
            residual_channel_mean_rms_mjy=float(np.sqrt(np.mean(residual[i].mean(axis=(0, 1))**2)))))
    channel_rows = []
    means = z.mean(axis=(0, 1, 2))
    sigma = np.sqrt(np.diag(covariance))
    for channel in range(data.shape[-1]):
        zz = z[..., channel]
        centered = zz-zz.mean()
        channel_rows.append(dict(channel=channel,
            fitted_sigma_mjy=float(sigma[channel]),
            raw_channel_mean_mjy=float(data[..., channel].mean()),
            predicted_channel_mean_mjy=float(mean_prediction[..., channel].mean()),
            residual_channel_mean_mjy=float(residual[..., channel].mean()),
            standardized_residual_mean=float(residual[..., channel].mean()/sigma[channel]),
            whitened_mean=float(zz.mean()), whitened_variance=float(zz.var()),
            whitened_skewness=float(np.mean(centered**3)/np.mean(centered**2)**1.5),
            whitened_q01=float(np.quantile(zz, .01)), whitened_q50=float(np.quantile(zz, .5)), whitened_q99=float(np.quantile(zz, .99)),
            positive_3sigma_fraction=float((zz > 3).mean()), negative_3sigma_fraction=float((zz < -3).mean())))
    qblock = np.array([b['q_over_n'] for b in block_rows])
    north = np.array([float(design[i, ..., 2].mean()) >= 0 for i in range(len(rows))])
    ratio = float(qblock[north].mean()/qblock[~north].mean()) if north.any() and (~north).any() else None
    spatial = spatial_diagnostics(z, rows, config['validation']['spatial_lags_native_pixels'], config['validation']['cross_block_lattice_lags'])
    groups = []
    for kind, axis, lag, centered in sorted(set((r['kind'], r['axis'], r['lag_native_pixels'], r['local_channel_means_removed']) for r in spatial)):
        group = [r for r in spatial if (r['kind'], r['axis'], r['lag_native_pixels'], r['local_channel_means_removed']) == (kind, axis, lag, centered)]
        groups.append(dict(kind=kind, axis=axis, lag_native_pixels=lag, local_channel_means_removed=centered,
            block_or_pair_count=len(group), mean_product=float(np.mean([r['product'] for r in group])),
            mean_normalized_product=float(np.mean([r['normalized_product'] for r in group])),
            mean_pearson=float(np.mean([r['pearson'] for r in group]))))
    summary = dict(blocks=len(rows), vectors_per_block=int(np.prod(data.shape[1:3])), channels=data.shape[-1],
        mean_q_over_n=float(qblock.mean()), block_q_sd=float(qblock.std(ddof=1)),
        block_q_min=float(qblock.min()), block_q_max=float(qblock.max()),
        mean_logpdf_per_channel=float(np.mean([r['logpdf_per_channel'] for r in block_rows])),
        mean_conditional_even_q_over_n=float(cq.mean()/len(predicted)), mean_conditional_even_logpdf_per_channel=float(clog.mean()/len(predicted)),
        mean_marginal_even_q_over_n=float(uq.mean()/len(predicted)), mean_marginal_even_logpdf_per_channel=float(ulog.mean()/len(predicted)),
        whitened_channel_mean_rms=float(np.sqrt(np.mean(means**2))),
        positive_3sigma_fraction=float((z > 3).mean()), negative_3sigma_fraction=float((z < -3).mean()),
        absolute_5sigma_fraction=float((np.abs(z) > 5).mean()), north_south_block_q_ratio=ratio,
        channel_logdet_fixed_mjy_units=logdet, covariance_condition=float(np.linalg.cond(covariance)),
        mean_whitened_block_mean_square=float(np.mean(z.mean(axis=(1, 2))**2)),
        mean_whitened_within_block_variance=float(z.var(axis=(1, 2)).mean()), spatial=groups)
    flags = config['validation']['descriptive_transfer_flags']
    summary['descriptive_checks'] = dict(
        q_mean_in_range=flags['mean_q_over_n_range'][0] <= summary['mean_q_over_n'] <= flags['mean_q_over_n_range'][1],
        mean_residual_small=summary['whitened_channel_mean_rms'] <= flags['whitened_channel_mean_rms_max'],
        absolute_3sigma_tails_bounded=summary['positive_3sigma_fraction']+summary['negative_3sigma_fraction'] <= flags['absolute_whitened_3sigma_tail_fraction_max'],
        signed_3sigma_asymmetry_bounded=abs(summary['positive_3sigma_fraction']-summary['negative_3sigma_fraction']) <= flags['signed_3sigma_tail_fraction_difference_abs_max'],
        north_south_q_in_range=ratio is not None and flags['north_south_block_q_ratio_range'][0] <= ratio <= flags['north_south_block_q_ratio_range'][1])
    summary['all_descriptive_checks_pass'] = bool(all(summary['descriptive_checks'].values()))
    cross = [abs(r['mean_product']) for r in groups if r['kind'] == 'cross_core' and not r['local_channel_means_removed']]
    summary['cross_block_dependence_warning'] = bool(max(cross, default=float('inf')) > flags['cross_block_uncentered_product_abs_warning'])
    return summary, block_rows, channel_rows, spatial
