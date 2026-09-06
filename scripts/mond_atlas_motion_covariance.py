"""Known correlated channel noise for the frozen theory-only motion operator.

The Gaussian formulas are exact for a fixed mean and supplied covariance.
Fitted nonlinear means use plug-in forecasts, without parameter integration.
No observational data adapter or source/pressure/dynamics closure is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal

from mond_atlas_motion_controls import (
    CIRCULAR_PARAMETERS, PARAMETERS, Geometry, Instrument, forward_cube,
    numerical_controls as forward_numerical_controls,
)


def validated_covariance(covariance):
    c = np.array(covariance, dtype=float, copy=True)
    if (c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0
            or not np.isfinite(c).all()):
        raise ValueError("finite nonempty square covariance required")
    scale = max(float(np.max(np.abs(c))), np.finfo(float).tiny)
    if np.max(np.abs(c-c.T)) > 1e-12*scale:
        raise ValueError("symmetric covariance required")
    # Do not repair a declared covariance, add jitter or use a pseudo-inverse.
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("strictly positive definite covariance required") from exc
    c.flags.writeable = False
    chol.flags.writeable = False
    return c, chol


class GaussianBlock:
    """C = K_channel kron diag(pixel_scale**2), channel-major vector order."""

    def __init__(self, channel_covariance, pixel_scale):
        self.covariance, self.cholesky = validated_covariance(channel_covariance)
        self.pixel_scale = np.array(pixel_scale, dtype=float, copy=True)
        if (self.pixel_scale.ndim != 1 or not self.pixel_scale.size
                or not np.isfinite(self.pixel_scale).all() or np.any(self.pixel_scale <= 0)):
            raise ValueError("positive finite per-pixel standard-deviation scales required")
        self.pixel_scale.flags.writeable = False
        self.shape = (len(self.covariance), len(self.pixel_scale))
        self.n = int(np.prod(self.shape))
        self.logdet = float(2*self.shape[1]*np.log(np.diag(self.cholesky)).sum()
                            +2*self.shape[0]*np.log(self.pixel_scale).sum())

    def whiten(self, residual):
        r = np.asarray(residual, dtype=float)
        if r.shape != self.shape or not np.isfinite(r).all():
            raise ValueError("finite residual with matching channel/pixel shape required")
        return solve_triangular(self.cholesky, r/self.pixel_scale[None, :],
                                lower=True, check_finite=False)

    def score(self, residual):
        white = self.whiten(residual)
        q = float(np.sum(white**2))
        nll = 0.5*(q+self.logdet+self.n*np.log(2*np.pi))
        return {"n": self.n, "q": q, "q_per_cell": q/self.n, "log_determinant": self.logdet,
                "negative_log_likelihood": float(nll), "nll_per_cell": float(nll/self.n)}


def channel_indices(indices, size, allow_empty=False):
    raw = np.asarray(indices)
    if (raw.ndim != 1 or (not allow_empty and not raw.size)
            or not np.issubdtype(raw.dtype, np.integer)
            or np.any(raw < 0) or np.any(raw >= size) or len(np.unique(raw)) != len(raw)):
        raise ValueError("unique in-range integer channel indices required")
    return raw.astype(int, copy=True)


def conditional_channels(covariance, training_channels, heldout_channels):
    """A=K_HT K_TT^-1, S=K_HH-A K_TH, retaining the supplied index order."""
    cov, _ = validated_covariance(covariance)
    t = channel_indices(training_channels, len(cov), allow_empty=True)
    h = channel_indices(heldout_channels, len(cov))
    if np.intersect1d(t, h).size:
        raise ValueError("conditioning and held-out channels must be disjoint")
    hh = cov[np.ix_(h, h)]
    if not t.size:
        return np.zeros((len(h), 0)), hh.copy()
    tt, lt = validated_covariance(cov[np.ix_(t, t)])
    th = cov[np.ix_(t, h)]
    gain = cho_solve((lt, True), th, check_finite=False).T
    schur = hh-gain @ th
    # Only remove roundoff asymmetry arising in this arithmetic, never change eigenvalues.
    schur = 0.5*(schur+schur.T)
    validated_covariance(schur)
    return gain, schur


def channel_covariance(nchannel, noise):
    if int(nchannel) != nchannel or nchannel < 2:
        raise ValueError("at least two integer channels required")
    rho, amplitude, modulation = (noise[k] for k in ("rho", "sigma_flux", "channel_modulation"))
    if (not np.isfinite([rho, amplitude, modulation]).all() or abs(rho) >= 1
            or amplitude <= 0 or abs(modulation) >= 1):
        raise ValueError("stationary |rho|<1 and positive channel noise scale required")
    channel = np.arange(nchannel)
    sigma = amplitude*(1+modulation*np.cos(2*np.pi*channel/nchannel))
    covariance = sigma[:, None]*sigma[None, :]*rho**np.abs(channel[:, None]-channel[None, :])
    validated_covariance(covariance)
    return covariance, sigma


def pixel_scales(npix, noise):
    amplitude = noise["pixel_x_modulation"]
    if int(npix) != npix or npix < 2 or not np.isfinite(amplitude) or amplitude <= -1:
        raise ValueError("invalid pixel scale geometry")
    return np.broadcast_to(1+amplitude*np.arange(npix)/(npix-1), (npix, npix)).copy().ravel()


def ar1_from_innovations(innovations, rho):
    """Stationary AR transform, independent of covariance-matrix factorization."""
    z = np.array(innovations, dtype=float, copy=True)
    if z.ndim != 2 or not z.size or not np.isfinite(z).all() or not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("finite innovations and stationary |rho|<1 required")
    for c in range(1, z.shape[0]):
        z[c] = rho*z[c-1]+np.sqrt(1-rho*rho)*z[c]
    return z


def innovation_noise(shape, noise, rng):
    """Generate with scalar AR innovations, without using a covariance factor."""
    if len(shape) != 3 or shape[1] != shape[2]:
        raise ValueError("square sky cube shape required")
    _, sigma = channel_covariance(shape[0], noise)
    scale = pixel_scales(shape[1], noise)
    z = ar1_from_innovations(rng.standard_normal((shape[0], scale.size)), noise["rho"])
    return (sigma[:, None]*scale[None, :]*z).reshape(shape)


@dataclass(frozen=True)
class FixedPartition:
    shape: tuple
    fold: int
    train_channels: np.ndarray
    test_channels: np.ndarray
    train_pixels: np.ndarray
    test_pixels: np.ndarray
    measurement_pixels: np.ndarray

    @classmethod
    def build(cls, shape, fold):
        if len(shape) != 3 or shape[1] != shape[2] or fold not in (0, 1, 2):
            raise ValueError("square cube and fold 0,1,2 required")
        c = np.arange(shape[0])
        y, x = np.indices(shape[1:])
        measured = (x+3*y) % 11 != 0
        held = (x+2*y) % 3 == fold
        return cls(tuple(shape), fold, c[c % 3 != fold], c[c % 3 == fold],
                   np.flatnonzero(measured & ~held), np.flatnonzero(measured & held),
                   np.flatnonzero(measured))

    def blocks(self):
        return {"train": (self.train_channels, self.train_pixels),
                "heldout_channels": (self.test_channels, self.train_pixels),
                "heldout_pixels": (self.train_channels, self.test_pixels),
                "heldout_joint": (self.test_channels, self.test_pixels)}

    def extract(self, cube, name):
        a = np.asarray(cube)
        if a.shape != self.shape:
            raise ValueError("cube shape differs from frozen partition")
        c, p = self.blocks()[name]
        return a.reshape(self.shape[0], -1)[np.ix_(c, p)]

    def masks(self):
        result = {}
        for name, (c, p) in self.blocks().items():
            a = np.zeros((self.shape[0], self.shape[1]*self.shape[2]), dtype=bool)
            a[np.ix_(c, p)] = True
            result[name] = a.reshape(self.shape)
        return result


def fit_motion(data, partition, covariance, spatial_scale, prior, config, expanded,
               fixed=None, quadrature=None, max_nfev=None):
    """Fit the training MARGINAL; copy only training response values into closure."""
    fixed = fixed or {}
    cov, _ = validated_covariance(covariance)
    if cov.shape != (partition.shape[0],)*2 or len(spatial_scale) != np.prod(partition.shape[1:]):
        raise ValueError("covariance/partition mismatch")
    t, p = partition.train_channels, partition.train_pixels
    block = GaussianBlock(cov[np.ix_(t, t)], np.asarray(spatial_scale)[p])
    target = partition.extract(data, "train").copy()
    if not np.isfinite(target).all():
        raise ValueError("training response must be finite")
    geometry, instrument = Geometry(**prior["geometry"]), Instrument(**prior["instrument"])
    if partition.shape != (instrument.nchannel, instrument.npix, instrument.npix):
        raise ValueError("instrument/partition mismatch")
    fit_config = config["study"]["fit"]
    nr, nphi = quadrature or fit_config["quadrature"]
    names = [k for k in (PARAMETERS if expanded else CIRCULAR_PARAMETERS) if k not in fixed]
    lower = np.array([prior["parameter_bounds"][k][0] for k in names])
    widths = np.array([prior["parameter_bounds"][k][1] for k in names])-lower

    def parameters(u):
        params = dict(prior["base_parameters"])
        params.update({k: 0.0 for k in PARAMETERS[5:]})
        params.update(zip(names, lower+widths*u))
        params.update(fixed)
        return params

    def residual(u):
        prediction = forward_cube(parameters(u), geometry, instrument, nr, nphi)
        return block.whiten(partition.extract(prediction, "train")-target).ravel()

    fits, receipts = [], []
    for start in prior["study"]["starts"]:
        values = dict(zip(PARAMETERS, start))
        initial = (np.array([values[k] for k in names])-lower)/widths
        fitted = least_squares(residual, initial, bounds=(np.zeros(len(names)), np.ones(len(names))),
                               max_nfev=max_nfev or fit_config["max_nfev"],
                               ftol=fit_config["ftol"], xtol=fit_config["xtol"], gtol=fit_config["gtol"])
        fits.append(fitted)
        receipts.append({"parameters": parameters(fitted.x), "training_q": float(2*fitted.cost),
                         "nfev": int(fitted.nfev), "status": int(fitted.status),
                         "success": bool(fitted.success), "message": str(fitted.message)})
    selected = int(np.argmin([r["training_q"] for r in receipts]))
    best = fits[selected]
    params = parameters(best.x)
    singular = np.linalg.svd(best.jac, compute_uv=False)
    rank = int(np.count_nonzero(singular > max(float(singular[0])*1e-8, 1e-10)))
    norms = np.linalg.norm(best.jac, axis=0)
    cosines = []
    for i in range(len(names)):
        for j in range(i):
            if norms[i]*norms[j] > 1e-14:
                cosines.append({"parameters": [names[j], names[i]],
                                "cosine": float(np.dot(best.jac[:, j], best.jac[:, i])/(norms[i]*norms[j]))})
    cosines.sort(key=lambda a: abs(a["cosine"]), reverse=True)
    prediction = forward_cube(params, geometry, instrument, nr, nphi)
    receipt = {"parameters": params, "free_parameters": names, "fixed_parameters": fixed,
               "selected_start": selected, "starts": receipts, "optimizer_success": bool(best.success),
               "training_marginal": block.score(partition.extract(prediction, "train")-target),
               "jacobian_rank": rank, "jacobian_singular_values": singular.tolist(),
               "jacobian_column_norms": dict(zip(names, norms.tolist())),
               "strongest_sensitivity_cosines": cosines[:5],
               "bound_contacts": [k for k, u in zip(names, best.x) if min(u, 1-u) < 1e-4],
               "parameter_uncertainty_integrated": False}
    return prediction, receipt


def forecast_evaluation(prediction, data, truth, fresh_data, partition,
                        true_covariance, assumed_covariance, spatial_scale):
    """Keep signal errors separate from same-noise conditional interpolation."""
    train_residual = partition.extract(data, "train")-partition.extract(prediction, "train")
    evaluation = {}
    for name, (channels, pixels) in partition.blocks().items():
        if name == "train":
            continue
        m = partition.extract(prediction, name)
        y = partition.extract(data, name)
        exact = partition.extract(truth, name)
        kt = true_covariance[np.ix_(channels, channels)]
        ka = assumed_covariance[np.ix_(channels, channels)]
        marginal_true = GaussianBlock(kt, spatial_scale[pixels])
        marginal_assumed = GaussianBlock(ka, spatial_scale[pixels])
        if name == "heldout_channels":
            a, sa = conditional_channels(assumed_covariance, partition.train_channels, channels)
            _, st = conditional_channels(true_covariance, partition.train_channels, channels)
            correction = a @ train_residual
        else:
            # No spatial covariance: held-out pixels have no observed conditioning cells.
            sa, st, correction = ka, kt, np.zeros_like(m)
        cond_assumed = GaussianBlock(sa, spatial_scale[pixels])
        cond_true = GaussianBlock(st, spatial_scale[pixels])
        fresh = [partition.extract(f, name) for f in fresh_data]
        row = {
            "n": marginal_true.n,
            "signal_noiseless_true_marginal": marginal_true.score(m-exact),
            "same_signal_true_marginal": marginal_true.score(y-m),
            "same_signal_assumed_marginal": marginal_assumed.score(y-m),
            "same_conditional_assumed_distribution": cond_assumed.score(y-m-correction),
            "same_conditional_common_true_schur": cond_true.score(y-m-correction),
            "same_conditional_common_true_marginal": marginal_true.score(y-m-correction),
            "fresh_signal_true_marginal": [marginal_true.score(f-m) for f in fresh],
            "fresh_signal_assumed_marginal": [marginal_assumed.score(f-m) for f in fresh],
            "fresh_transferred_noise_control_true_marginal": [marginal_true.score(f-m-correction) for f in fresh],
            "noiseless_transferred_noise_control": marginal_true.score(m+correction-exact),
            "noise_correction_size_true_marginal": marginal_true.score(correction)["q_per_cell"],
            "conditional_to_marginal_noise_variance_ratio": float(np.trace(st)/np.trace(kt)),
            "same_noise_conditioning_active": name == "heldout_channels",
            "fresh_noise_cross_covariance_with_training": 0.0,
        }
        row["fresh_signal_mean_q_per_cell"] = float(np.mean([v["q_per_cell"] for v in row["fresh_signal_true_marginal"]]))
        row["fresh_transferred_mean_q_per_cell"] = float(np.mean([v["q_per_cell"] for v in row["fresh_transferred_noise_control_true_marginal"]]))
        evaluation[name] = row
    return evaluation


def predictive_pass(evaluation):
    return all(row["fresh_signal_mean_q_per_cell"] <= 1.25
               and row["signal_noiseless_true_marginal"]["q_per_cell"] <= 0.25
               for row in evaluation.values())


def numerical_controls(config, prior):
    """Pre-response matrix/manufactured controls, retaining every gate value."""
    import copy
    limits = config["controls"]
    rng = np.random.default_rng(limits["seed"])
    records = []

    def gate(name, error, bound, **detail):
        records.append({"name": name, "error": float(error), "tolerance": float(bound),
                        "passed": bool(np.isfinite(error) and error <= bound), **detail})

    def relative(a, b):
        return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))/max(float(np.max(np.abs(b))), 1e-300))

    # Six channels and three independent pixels, but non-diagonal channel covariance.
    fixture = dict(config["noise"], sigma_flux=0.8)
    k, sig = channel_covariance(6, fixture)
    scales = np.array([0.7, 1., 1.4])
    block = GaussianBlock(k, scales)
    dense = np.kron(k, np.diag(scales**2))
    residual = rng.normal(size=block.shape)
    vector = residual.ravel()
    independent_inverse = np.linalg.inv(dense)
    dense_q = float(vector @ independent_inverse @ vector)
    sign, dense_logdet = np.linalg.slogdet(dense)
    result = block.score(residual)
    gate("whitening_quadratic_vs_dense_inverse", abs(result["q"]-dense_q)/dense_q, limits["relative_quadratic_tolerance"])
    gate("whitening_vector_vs_dense_cholesky", relative(block.whiten(residual).ravel(), np.linalg.solve(np.linalg.cholesky(dense), vector)), limits["relative_matrix_tolerance"])
    gate("logdet_vs_dense_determinant", abs(result["log_determinant"]-dense_logdet), limits["log_likelihood_absolute_tolerance"])
    independent_nll = -float(multivariate_normal.logpdf(vector, mean=np.zeros(vector.size), cov=dense, allow_singular=False))
    gate("gaussian_logpdf_vs_scipy", abs(result["negative_log_likelihood"]-independent_nll), limits["log_likelihood_absolute_tolerance"])
    eigenvalues = np.linalg.eigvalsh(k)
    records.append({"name": "strict_positive_definiteness", "passed": bool(eigenvalues.min() > 0 and sign == 1),
                    "minimum_eigenvalue": float(eigenvalues.min()), "condition_number": float(eigenvalues.max()/eigenvalues.min())})

    t, h = np.array([4, 1]), np.array([5, 0, 3, 2])  # Non-sorted ordering must survive.
    gain, schur = conditional_channels(k, t, h)
    tt = np.kron(k[np.ix_(t, t)], np.diag(scales**2))
    tindex = (t[:, None]*3+np.arange(3)).ravel()
    hindex = (h[:, None]*3+np.arange(3)).ravel()
    phh = independent_inverse[np.ix_(hindex, hindex)]
    pht = independent_inverse[np.ix_(hindex, tindex)]
    s_dense = np.linalg.inv(phh)
    a_dense = -s_dense @ pht
    gate("conditional_schur_vs_joint_precision", relative(np.kron(schur, np.diag(scales**2)), s_dense), limits["relative_matrix_tolerance"])
    rt, rh = residual[t], residual[h]
    gate("conditional_mean_vs_joint_precision", relative((gain @ rt).ravel(), a_dense @ rt.ravel()), limits["relative_matrix_tolerance"])
    block_t, block_h = GaussianBlock(k[np.ix_(t, t)], scales), GaussianBlock(schur, scales)
    factorized_nll = block_t.score(rt)["negative_log_likelihood"]+block_h.score(rh-gain @ rt)["negative_log_likelihood"]
    gate("joint_density_factorization", abs(factorized_nll-result["negative_log_likelihood"]), limits["log_likelihood_absolute_tolerance"])
    marginal_q = float(rt.ravel() @ np.linalg.inv(tt) @ rt.ravel())
    gate("training_marginal_vs_dense_subcovariance", abs(block_t.score(rt)["q"]-marginal_q)/marginal_q, limits["relative_quadratic_tolerance"])
    wrong_q = float(rt.ravel() @ independent_inverse[np.ix_(tindex, tindex)] @ rt.ravel())
    records.append({"name": "full_precision_subset_is_not_marginal", "passed": bool(abs(wrong_q-marginal_q)/marginal_q > limits["relative_quadratic_tolerance"]),
                    "correct_marginal_q": marginal_q, "wrong_precision_subset_q": wrong_q})
    gate("conditional_noise_uncorrelated_with_training", np.max(np.abs(k[np.ix_(h, t)]-gain @ k[np.ix_(t, t)])), limits["relative_matrix_tolerance"])

    # Innovation response to independent unit columns yields its exact covariance.
    transform = ar1_from_innovations(np.eye(6), fixture["rho"])*sig[:, None]
    gate("AR_innovations_exact_covariance", relative(transform @ transform.T, k), limits["relative_matrix_tolerance"])
    diagonal, diag_sig = channel_covariance(6, dict(fixture, rho=0.0))
    ad, sd = conditional_channels(diagonal, t, h)
    gate("diagonal_conditional_gain_zero", np.max(np.abs(ad)), limits["diagonal_limit_tolerance"])
    gate("diagonal_schur_equals_marginal", relative(sd, diagonal[np.ix_(h, h)]), limits["diagonal_limit_tolerance"])
    diag_q = np.sum((residual/(diag_sig[:, None]*scales))**2)
    gate("diagonal_whitening_limit", abs(GaussianBlock(diagonal, scales).score(residual)["q"]-diag_q)/diag_q, limits["diagonal_limit_tolerance"])

    # Independent replica is a distinct covariance block, even at identical channels/pixels.
    replica_cov = np.zeros((12, 12))
    replica_cov[:6, :6], replica_cov[6:, 6:] = k, k
    af, sf = conditional_channels(replica_cov, t, h+6)
    gate("fresh_replica_zero_conditional_gain", np.max(np.abs(af)), limits["diagonal_limit_tolerance"])
    gate("fresh_replica_marginal_covariance", relative(sf, k[np.ix_(h, h)]), limits["diagonal_limit_tolerance"])
    # Two pixels with independent noise have identical zero cross-condition behavior.
    p_cov = np.kron(np.eye(2), k)
    ap, sp = conditional_channels(p_cov, t, h+6)
    gate("heldout_pixel_zero_conditional_gain", np.max(np.abs(ap)), limits["diagonal_limit_tolerance"])
    gate("heldout_pixel_marginal_covariance", relative(sp, k[np.ix_(h, h)]), limits["diagonal_limit_tolerance"])

    invalid = [np.zeros((2, 2)), np.array([[1., 2], [2, 1]]),
               np.array([[1., 0.1], [0.3, 1]]), np.array([[np.nan]]), np.array([1., 2.])]
    rejected = 0
    for value in invalid:
        try:
            GaussianBlock(value, np.ones(1))
        except ValueError:
            rejected += 1
    records.append({"name": "invalid_covariance_rejected_without_repair", "passed": rejected == len(invalid), "rejected": rejected, "total": len(invalid)})

    draws = limits["oracle_monte_carlo_draws"]
    noise = ar1_from_innovations(rng.normal(size=(6, draws)), fixture["rho"])*sig[:, None]
    fresh = ar1_from_innovations(rng.normal(size=(6, draws)), fixture["rho"])*sig[:, None]
    oracle_correction = gain @ noise[t]
    errors = noise[h]-oracle_correction
    white = GaussianBlock(schur, np.ones(draws)).whiten(errors)
    gate("oracle_conditional_whitened_mean", np.max(np.abs(white.mean(axis=1))), limits["oracle_whitened_mean_absolute_max"])
    gate("oracle_conditional_whitened_covariance", np.max(np.abs(np.cov(white)-np.eye(len(h)))), limits["oracle_whitened_covariance_absolute_max"])
    gate("oracle_conditional_q_expectation", abs(float(np.mean(white**2))-1), limits["oracle_q_per_dimension_error_max"])
    marginal = GaussianBlock(k[np.ix_(h, h)], np.ones(draws))
    expected_remaining = float(np.trace(np.linalg.solve(k[np.ix_(h, h)], schur))/len(h))
    same_q = marginal.score(errors)["q_per_cell"]
    fresh_q = marginal.score(fresh[h])["q_per_cell"]
    transferred_q = marginal.score(fresh[h]-oracle_correction)["q_per_cell"]
    gate("oracle_same_noise_interpolation_expectation", abs(same_q-expected_remaining), limits["oracle_q_per_dimension_error_max"], measured_q_per_cell=same_q, expected_q_per_cell=expected_remaining)
    gate("oracle_fresh_noise_marginal_expectation", abs(fresh_q-1), limits["oracle_q_per_dimension_error_max"])
    expected_transferred = 2-expected_remaining
    gate("oracle_fresh_noise_transfer_penalty", abs(transferred_q-expected_transferred), limits["oracle_q_per_dimension_error_max"], measured_q_per_cell=transferred_q, expected_q_per_cell=expected_transferred)

    # Irregular measured mask: every retained cell belongs to exactly one partition.
    partition = FixedPartition.build((9, 9, 9), 1)
    masks = partition.masks()
    union = sum(m.astype(int) for m in masks.values())
    target_mask = np.zeros((9, 81), dtype=int)
    target_mask[:, partition.measurement_pixels] = 1
    gate("fixed_mask_disjoint_complete_measured_partition", np.max(np.abs(union-target_mask.reshape(union.shape))), 0.0)
    subprior = copy.deepcopy(prior)
    subprior["instrument"].update(npix=9, nchannel=9, beam_half_width=2)
    data = forward_cube(subprior["base_parameters"], Geometry(**subprior["geometry"]),
                        Instrument(**subprior["instrument"]), 6, 24)
    noise = innovation_noise(data.shape, config["noise"], rng)
    data = data+noise
    changed = data.copy()
    changed[~masks["train"]] = np.nan
    cov, _ = channel_covariance(9, config["noise"])
    scale = pixel_scales(9, config["noise"])
    for label, c in (("correct", cov), ("diagonal", np.diag(np.diag(cov)))):
        a, ra = fit_motion(data, partition, c, scale, subprior, config, True, quadrature=(6, 24), max_nfev=8)
        b, rb = fit_motion(changed, partition, c, scale, subprior, config, True, quadrature=(6, 24), max_nfev=8)
        delta = max(np.max(np.abs(a-b)), abs(ra["training_marginal"]["q"]-rb["training_marginal"]["q"]))
        gate(f"heldout_and_masked_response_cannot_change_{label}_fit", delta, limits["heldout_fit_change_tolerance"], control_optimizer_budget=8)
    # Changing residual values affects the conditional mean, never the covariance.
    _, same_schur = conditional_channels(k, t, h)
    gate("conditional_covariance_response_independent", relative(same_schur, schur), 0.0)
    physical = forward_numerical_controls(prior)
    return {"disposition": "THEORY_BENCHMARK_ONLY" if all(r["passed"] for r in records) and physical["all_passed"] else "BENCHMARK_FAILED",
            "all_passed": all(r["passed"] for r in records) and physical["all_passed"],
            "statistical_controls": records, "prior_forward_controls": physical,
            "statistical_control_count": len(records), "prior_forward_control_count": len(physical["controls"]),
            "oracle_control_draws": draws}
