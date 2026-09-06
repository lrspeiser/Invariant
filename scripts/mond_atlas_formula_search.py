"""Training-only greedy sparse residual expressions; no physical force operator."""
from __future__ import annotations
import numpy as np


def library(z):
    terms = [(i,) for i in range(4, 8)]
    terms += [(i, i) for i in range(4, 8)]
    terms += [(i, j) for i in range(4, 8) for j in range(i+1, 8)]
    terms += [(i, j) for i in range(4) for j in range(4, 8)]
    return np.column_stack([np.prod(z[:, t], axis=1) for t in terms]), terms


def design(train, evaluate):
    if train.ndim != 2 or train.shape[1] != 8 or evaluate.ndim != 2 or evaluate.shape[1] != 8:
        raise ValueError('Eight declared inputs required')
    if not np.isfinite(train).all() or not np.isfinite(evaluate).all() or len(train) < 2:
        raise ValueError('Finite inputs and at least two training rows required')
    center = train.mean(axis=0); scale = np.maximum(train.std(axis=0), 1e-12)
    a = (train-center)/scale; b = (evaluate-center)/scale
    la, terms = library(a); lb, _ = library(b)
    lc = la.mean(axis=0); ls = np.maximum(la.std(axis=0), 1e-12)
    return np.column_stack([a[:, :4], (la-lc)/ls]), np.column_stack([b[:, :4], (lb-lc)/ls]), dict(
        input_center=center.tolist(), input_scale=scale.tolist(), expression_center=lc.tolist(),
        expression_scale=ls.tolist(), terms=[list(t) for t in terms])


def host(value, xp):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def paths(x, y, evaluate, alpha, depth, xp=np):
    """At each step test all remaining expressions; retain exact fitted formula."""
    if not np.isfinite(y).all() or y.shape != (len(x),) or alpha <= 0 or depth not in range(31):
        raise ValueError('Invalid target, penalty or depth')
    a, b, transform = design(x, evaluate)
    a, b = xp.asarray(a), xp.asarray(b)
    mean = float(y.mean()); yc = xp.asarray(y-mean)
    gram = a.T@a; rhs = a.T@yc
    selected = list(range(4)); result = []
    def fit(cols):
        ix = xp.asarray(cols)
        coef = xp.linalg.solve(gram[ix[:, None], ix]+alpha*xp.eye(len(cols)), rhs[ix])
        # At optimum SSE + alpha*||coef||² equals y'y - coef'X'y.
        objective = float(host(yc@yc-coef@rhs[ix], xp))
        return coef, objective
    for step in range(depth+1):
        coef, objective = fit(selected)
        result.append(dict(prediction=host(b[:, selected]@coef+mean, xp),
            formula=dict(transform=transform, columns=selected.copy(), coefficients=host(coef, xp).tolist(),
                         intercept=mean, alpha=alpha, added_terms=step), objective=objective))
        if step < depth:
            trials = [(fit(selected+[j])[1], j) for j in range(4, a.shape[1]) if j not in selected]
            selected.append(min(trials)[1])
    return result


def replay(x, formula):
    t = formula['transform']; z = (x-np.array(t['input_center']))/np.array(t['input_scale'])
    expressions, _ = library(z)
    a = np.column_stack([z[:, :4], (expressions-np.array(t['expression_center']))/np.array(t['expression_scale'])])
    return a[:, formula['columns']]@np.array(formula['coefficients'])+formula['intercept']


def outer(x, y, folds, held, config, xp=np):
    train = folds != held; test = ~train
    if not train.any() or not test.any() or len(set(folds[train])) < 2:
        raise ValueError('Need an outer test and at least two inner folds')
    depth = config['maximum_added_terms']; alphas = config['ridge_penalties']
    scores = []
    for alpha in alphas:
        errors = [[] for _ in range(depth+1)]
        for inner in sorted(set(folds[train])):
            fit = train & (folds != inner); valid = train & (folds == inner)
            for k, p in enumerate(paths(x[fit], y[fit], x[valid], alpha, depth, xp)):
                errors[k].extend((p['prediction']-y[valid])**2)
        scores.extend(dict(alpha=alpha, depth=k, mse=float(np.mean(e))) for k, e in enumerate(errors))
    key = lambda r: (r['mse'], r['depth'], -r['alpha'])
    chosen = min(scores, key=key); baseline = min((r for r in scores if r['depth']==0), key=key)
    out = {}
    for name, choice in [('adaptive', chosen), ('baseline', baseline)]:
        p = paths(x[train], y[train], x[test], choice['alpha'], choice['depth'], xp)[-1]
        out[name] = p
    out['selection'] = dict(fold=int(held), train_count=int(train.sum()), test_count=int(test.sum()),
                            adaptive=chosen, baseline=baseline, inner_scores=scores)
    return out


def nested(x, y, folds, config, xp=np):
    predictions = {k: np.full(len(y), np.nan) for k in ('adaptive', 'baseline')}; records = []
    for held in sorted(set(folds)):
        result = outer(x, y, folds, held, config, xp)
        record = result['selection']; record['formulas'] = {}
        for name in predictions:
            predictions[name][folds==held] = result[name]['prediction']
            record['formulas'][name] = result[name]['formula']
        records.append(record)
    return predictions, records


def controls(xp=np):
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(901); x = rng.normal(size=(160, 8))
    y = .2*x[:, 0]+.8*x[:, 4]*x[:, 7]+rng.normal(0, .01, 160)
    a, b, _ = design(x[:110], x[110:])
    tested = paths(x[:110], y[:110], x[110:], .1, 1, xp)
    cpu = paths(x[:110], y[:110], x[110:], .1, 1)
    ref = Ridge(alpha=.1, fit_intercept=False).fit(a[:, :4], y[:110]-y[:110].mean()).predict(b[:, :4])+y[:110].mean()
    return dict(cpu_gpu_max_abs=float(max(np.max(abs(t['prediction']-c['prediction'])) for t,c in zip(tested,cpu))),
        independent_ridge_max_abs=float(np.max(abs(ref-tested[0]['prediction']))),
        planted_rmse_ratio=float(np.sqrt(np.mean((tested[1]['prediction']-y[110:])**2)/np.mean((tested[0]['prediction']-y[110:])**2))),
        replay_max_abs=float(np.max(abs(replay(x[110:], tested[1]['formula'])-tested[1]['prediction']))))
