"""Small, auditable nested regression with interchangeable NumPy/CuPy backends."""
from __future__ import annotations
import hashlib
import numpy as np


def galaxy_folds(names, seed, count=5):
    if len(set(names)) != len(names):
        raise ValueError('One row per independent galaxy identifier required')
    order = sorted(names, key=lambda n: hashlib.sha256(f'{seed}|{n}'.encode()).digest())
    mapping = {name: i % count for i, name in enumerate(order)}
    return np.array([mapping[n] for n in names])


def standardize(train, evaluate):
    center = np.mean(train, axis=0)
    scale = np.maximum(np.std(train, axis=0), 1e-12)
    return (train-center)/scale, (evaluate-center)/scale


def predict(train_x, train_y, test_x, estimator, alpha, gamma_multiplier=1., xp=np):
    a, b = standardize(np.asarray(train_x), np.asarray(test_x))
    mean = float(np.mean(train_y))
    a, b, y = xp.asarray(a), xp.asarray(b), xp.asarray(train_y-mean)
    if estimator == 'linear_ridge':
        coefficients = xp.linalg.solve(a.T@a + alpha*xp.eye(a.shape[1]), a.T@y)
        result = b@coefficients + mean
    elif estimator == 'rbf_kernel_ridge':
        def kernel(x, z):
            d2 = xp.maximum(xp.sum(x*x, axis=1)[:, None] + xp.sum(z*z, axis=1)[None, :] - 2*x@z.T, 0.)
            return xp.exp(-gamma_multiplier/a.shape[1]*d2)
        coefficients = xp.linalg.solve(kernel(a,a) + alpha*xp.eye(len(a)), y)
        result = kernel(b,a)@coefficients + mean
    else:
        raise ValueError('unknown estimator')
    return np.asarray(result) if xp is np else xp.asnumpy(result)


def parameter_grid(estimator, config):
    gamma = [1.] if estimator == 'linear_ridge' else config['rbf_gamma_multipliers']
    return [(alpha,g) for alpha in config['ridge_penalties'] for g in gamma]


def outer_prediction(x, y, folds, held_fold, estimator, config, xp=np):
    train, test = folds != held_fold, folds == held_fold
    assert train.any() and test.any() and not np.any(train & test)
    params = parameter_grid(estimator, config)
    losses = []
    for alpha,gamma in params:
        errors = []
        for inner in sorted(set(folds[train])):
            fit, validate = train & (folds != inner), train & (folds == inner)
            pred = predict(x[fit], y[fit], x[validate], estimator, alpha, gamma, xp)
            errors.extend((pred-y[validate])**2)
        losses.append(float(np.mean(errors)))
    choice = int(np.argmin(losses)); alpha,gamma = params[choice]
    prediction = predict(x[train], y[train], x[test], estimator, alpha, gamma, xp)
    return prediction, dict(fold=int(held_fold), train_count=int(train.sum()), test_count=int(test.sum()),
        alpha=alpha, gamma_multiplier=gamma, inner_mse=losses[choice])


def nested_predictions(x, y, folds, estimator, config, xp=np):
    result = np.full(len(y), np.nan); selections = []
    for held in sorted(set(folds)):
        prediction, selected = outer_prediction(x, y, folds, held, estimator, config, xp)
        result[folds == held] = prediction; selections.append(selected)
    assert np.isfinite(result).all()
    return result, selections


def synthetic_controls(xp=np):
    """A held-out nonlinear signal and independent implementation control."""
    from sklearn.kernel_ridge import KernelRidge
    rng = np.random.default_rng(60701)
    x = rng.uniform(-2, 2, (220, 3))
    y = np.sin(1.5*x[:,0]) + .3*x[:,1]**2 + rng.normal(0,.05,len(x))
    tr, te = np.arange(150), np.arange(150,220)
    cpu = predict(x[tr],y[tr],x[te],'rbf_kernel_ridge',.1,1.)
    tested = predict(x[tr],y[tr],x[te],'rbf_kernel_ridge',.1,1.,xp)
    a,b = standardize(x[tr],x[te]); center = y[tr].mean()
    reference = KernelRidge(alpha=.1,kernel='rbf',gamma=1/3).fit(a,y[tr]-center).predict(b)+center
    return dict(cpu_backend_max_abs=float(np.max(np.abs(tested-cpu))),
        sklearn_max_abs=float(np.max(np.abs(tested-reference))),
        positive_control_rmse_ratio=float(np.sqrt(np.mean((tested-y[te])**2)/np.mean((center-y[te])**2))),
        synthetic_training_galaxies=len(tr),synthetic_test_galaxies=len(te))
