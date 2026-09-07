"""Restricted three-nuisance fitting; evaluation consumes frozen predictions."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular

LOW = np.array([-30., 3., .5])
HIGH = np.array([30., 20., 2.])
STARTS = ([0., 6., 1.], [0., 10., 1.], [0., 16., 1.])


def arrays(values, mean, covariance):
    values, mean, covariance = [np.asarray(a, float) for a in (values, mean, covariance)]
    if (values.ndim != 2 or values.shape[1] != 42 or values.shape[0] == 0
            or mean.shape != (42,) or covariance.shape != (42, 42)
            or not all(np.isfinite(a).all() for a in (values, mean, covariance))
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-15)):
        raise ValueError('Finite aperture-by42 spectra, western mean and symmetric covariance required')
    return values, mean, np.linalg.cholesky(covariance)


def training_fit(predict, aperture_indices, observed_training, western_mean, covariance):
    data, mean, lower = arrays(observed_training, western_mean, covariance)
    indices = tuple(aperture_indices)
    if len(indices) != len(data) or len(set(indices)) != len(indices):
        raise ValueError('Unique indices must match training spectra')
    def residual(parameters):
        values = np.asarray(predict(indices, np.asarray(parameters).copy()), float)
        if values.shape != data.shape or not np.isfinite(values).all():
            raise ValueError('Invalid predicted training spectra')
        return (solve_triangular(lower, (values+mean-data).T, lower=True).T/
                np.sqrt(data.size)).ravel()
    starts = []
    valid = []
    for initial in STARTS:
        try:
            fit = least_squares(residual, initial, bounds=(LOW, HIGH),
                                ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=300)
            loss = float(fit.fun@fit.fun)
            entry = dict(initial=list(initial), success=bool(fit.success), status=int(fit.status),
                         message=fit.message, nfev=int(fit.nfev), parameters=fit.x.tolist(),
                         training_mean_q_per_channel=loss,
                         bound_contacts=np.flatnonzero((fit.x-LOW<1e-6)|(HIGH-fit.x<1e-6)).tolist())
            if fit.success and np.isfinite(loss):
                valid.append((loss, fit.x.copy()))
        except (ValueError, np.linalg.LinAlgError) as exc:
            entry = dict(initial=list(initial), success=False, error=repr(exc))
        starts.append(entry)
    if not valid:
        return dict(status='ALL_STARTS_FAILED', starts=starts, evaluation_allowed=False)
    loss, parameters = min(valid, key=lambda item: item[0])
    jacobian = []
    for j, scale in enumerate(HIGH-LOW):
        minus, plus = parameters.copy(), parameters.copy()
        minus[j] = max(LOW[j], parameters[j]-1e-5*scale)
        plus[j] = min(HIGH[j], parameters[j]+1e-5*scale)
        jacobian.append((residual(plus)-residual(minus))*scale/(plus[j]-minus[j]))
    singular = np.linalg.svd(np.array(jacobian).T, compute_uv=False)
    ratio = float(singular[-1]/singular[0]) if singular[0]>0 else 0.
    admitted = bool(ratio>1e-6)
    return dict(status='TRAINING_FIT_IDENTIFIABLE' if admitted else 'TRAINING_RANK_FAILED',
                evaluation_allowed=admitted, parameters=parameters.tolist(),
                training_indices=list(indices), training_mean_q_per_channel=loss,
                scaled_singular_values=singular.tolist(), scaled_singular_ratio=ratio, starts=starts)


@dataclass(frozen=True)
class FrozenPrediction:
    indices: tuple
    values: np.ndarray


def freeze_predictions(predict, fit, indices):
    if fit.get('evaluation_allowed') is not True:
        raise ValueError('Training identifiability must pass first')
    indices = tuple(indices)
    if (not indices or len(set(indices)) != len(indices)
            or set(indices)&set(fit['training_indices'])):
        raise ValueError('Evaluation indices must be unique and disjoint from training')
    values = np.asarray(predict(indices, np.array(fit['parameters'])), float).copy()
    if values.shape != (len(indices), 42) or not np.isfinite(values).all():
        raise ValueError('Invalid evaluation predictions')
    values.flags.writeable = False
    return FrozenPrediction(indices, values)


def evaluate(frozen_prediction, observed_evaluation, western_mean, covariance):
    if not isinstance(frozen_prediction, FrozenPrediction):
        raise TypeError('Numeric frozen predictions required, never an optimizer')
    data, mean, lower = arrays(observed_evaluation, western_mean, covariance)
    if data.shape != frozen_prediction.values.shape:
        raise ValueError('Prediction and evaluation shape differ')
    residual = data-mean-frozen_prediction.values
    white = solve_triangular(lower, residual.T, lower=True).T
    q = np.sum(white*white, axis=1)/42
    return dict(indices=list(frozen_prediction.indices), per_aperture_q_per_channel=q.tolist(),
                mean_q_per_channel=float(q.mean()), raw_rmse_mJy_beam=float(np.sqrt(np.mean(residual**2))),
                residuals=residual.tolist(), interpretation='descriptive conditional discrepancy')
