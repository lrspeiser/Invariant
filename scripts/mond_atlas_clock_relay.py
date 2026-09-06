"""Frozen radial empirical candidates; no observational I/O or fitting on import."""
import itertools
import numpy as np

G = 4.30091727003628e-6  # kpc (km/s)^2 / Msun
A0 = 1.2e-10 / (1e6 / 3.085677581491367e19)  # (km/s)^2/kpc


def candidate_grid(config):
    result = []
    for family in config['families']:
        if family in ('newton_fixed', 'mond_fixed'):
            result.append(dict(family=family, mf=1.0))
            continue
        options = {'mf': config['mass_factors']}
        if family == 'mond_adjusted':
            options['a0_factor'] = config['a0_factors']
        elif family == 'absorption_proxy':
            options['kappa'] = config['opacities_per_100_msun_pc2']
        elif family == 'surface_relay':
            options.update(beta=config['strengths'], sigma0=config['surface_scales_msun_pc2'])
        elif family == 'clock_potential':
            options.update(beta=config['strengths'], clock_factor=config['clock_factors'])
        elif family in ('kernel_point', 'finite_p2', 'finite_p3', 'finite_mixture'):
            options.update(eta=config['strengths'], length_factor=config['length_factors'])
            if family == 'finite_mixture':
                options['q'] = config['core_mixture_weights']
        elif family != 'newton_ml':
            raise ValueError(f'Unknown family: {family}')
        for values in itertools.product(*options.values()):
            row = dict(family=family, **dict(zip(options, values)))
            if family == 'kernel_point':
                row['cutoff'] = config['kernel_cutoff_over_L']
            result.append(row)
    return result


def nfw_mass_shape(x, xp=np):
    x = xp.asarray(x)
    # Avoid cancellation at the origin. Series is evaluated only on small arguments.
    t = xp.minimum(x, 1e-3)
    series = sum((-1)**k * (k-1)/k * t**k for k in range(2, 10))
    return xp.where(x < 1e-3, series, xp.log1p(x)-x/(1+x))


def predict_logv(sources, candidate, xp=np):
    r, gas, disk, bulge, sb, luminosity, hi, rd = (
        xp.asarray(sources[key]) for key in ('r', 'gas', 'disk', 'bulge', 'sb', 'luminosity', 'hi', 'rd'))
    mf = candidate['mf']
    vb2 = gas*xp.abs(gas) + mf*(0.5*disk**2+0.7*bulge**2)
    gb = vb2/r
    family = candidate['family']
    if family in ('newton_fixed', 'newton_ml'):
        g = gb
    elif family in ('mond_fixed', 'mond_adjusted'):
        a0 = A0*candidate.get('a0_factor', 1.0)
        g = 0.5*(gb+xp.sqrt(gb**2+4*a0*gb))
    elif family == 'absorption_proxy':
        sigma = 0.5*mf*xp.maximum(sb, 0)
        g = gb*xp.exp(-candidate['kappa']*sigma/100)
    elif family == 'surface_relay':
        sigma = 0.5*mf*xp.maximum(sb, 0)
        g = gb*(1+candidate['beta']/(1+sigma/candidate['sigma0']))
    else:
        GM = G*1e9*(0.5*mf*luminosity+1.33*hi)
        if family == 'clock_potential':
            psi0 = candidate['clock_factor']*A0*rd
            extra = candidate['beta']*GM/((r+rd)*(r+rd+GM/psi0))
        elif family in ('kernel_point', 'finite_p2', 'finite_p3', 'finite_mixture'):
            L = candidate['length_factor']*rd
            x = r/L
            if family == 'kernel_point':
                extra = candidate['eta']*GM/r**2*nfw_mass_shape(xp.minimum(x,candidate['cutoff']),xp)
            else:
                p2 = 1/(1+x)**2
                p3 = x/(1+x)**3
                shape = p2 if family == 'finite_p2' else p3
                if family == 'finite_mixture':
                    shape = (1-candidate['q'])*p2+candidate['q']*p3
                extra = candidate['eta']*GM/L**2*shape
        else:
            raise ValueError(f'Unknown family: {family}')
        g = gb+extra
    return 0.5*xp.log10(r*g)


def loss_select(loss_by_candidate_galaxy, train_mask):
    loss = np.asarray(loss_by_candidate_galaxy)
    mask = np.asarray(train_mask, dtype=bool)
    if loss.ndim != 2 or mask.shape != (loss.shape[1],) or not mask.any():
        raise ValueError('Expected candidate by galaxy matrix and nonempty training mask')
    training_loss = loss[:, mask]
    if not np.isfinite(training_loss).all():
        raise ValueError('Nonfinite training loss')
    return int(np.argmin(training_loss.mean(axis=1)))
