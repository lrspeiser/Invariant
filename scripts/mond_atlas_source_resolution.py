"""Fixed observed-cell integrals of a refinable nonnegative bilinear source."""
from __future__ import annotations
import numpy as np


def cell_projection_matrix(cells, cell_width, nodes, node_spacing, blur_scale):
    """Average a unit-height tent of width 2*node_spacing over each cell.

    The tent is convolved with a normalized Laplace kernel. All lengths must
    use the same unit. Array bounds are finite; no periodic wrapping occurs.
    """
    cells = np.asarray(cells, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    d, h, b = float(cell_width), float(node_spacing), float(blur_scale)
    if (cells.ndim != 1 or nodes.ndim != 1 or not cells.size or not nodes.size
            or not np.isfinite(cells).all() or not np.isfinite(nodes).all()
            or not np.isfinite([d, h, b]).all() or min(d, h) <= 0 or b < 0):
        raise ValueError('invalid cell or source geometry')
    distance = np.abs(cells[:, None] - nodes[None, :])
    hi, lo = distance + d / 2, distance - d / 2
    if b == 0:
        def primitive(x):
            return np.where(x <= -h, 0, np.where(x < 0, (x+h)**2/(2*h),
                   np.where(x < h, h-(h-x)**2/(2*h), h)))
        a = (primitive(hi) - primitive(lo)) / d
    else:
        if min(d, h) / b < .001:
            raise ValueError('extreme kernel ratio needs a higher precision expansion')
        far = distance >= h + d / 2
        a = np.zeros_like(distance)
        # Stable tail: no exp(+large) or cancellation of polynomial terms.
        a[far] = (b*b/(2*h*d) * (-np.expm1(-h/b))**2 * (-np.expm1(-d/b))
                  * np.exp(-(distance[far]-d/2-h)/b))
        def primitive(x):
            return .5*(np.maximum(x, 0)**2
                       + b*b*np.sign(x)*(-np.expm1(-np.abs(x)/b)))
        def second_difference(x):
            return primitive(x+h) - 2*primitive(x) + primitive(x-h)
        near = ~far
        a[near] = (second_difference(hi[near])-second_difference(lo[near]))/(h*d)
    if not np.isfinite(a).all() or a.min() < -2e-12 or a.max() > 1+2e-12:
        raise ArithmeticError('invalid analytic cell coefficient')
    return np.maximum(a, 0)


def project(surface, left, right):
    return left @ surface @ right.T


def adjoint(image, left, right):
    return left.T @ image @ right


def roughness_gradient(surface, xp=np):
    out = xp.zeros_like(surface)
    dx = surface[1:, :] - surface[:-1, :]
    dy = surface[:, 1:] - surface[:, :-1]
    out[1:, :] += dx
    out[:-1, :] -= dx
    out[:, 1:] += dy
    out[:, :-1] -= dy
    return out


def fit_fixed_image(target, weight, left, right, support, refinement=1,
                    regularization=1e-4, max_iterations=12000, tolerance=1e-6,
                    backend='numpy'):
    """Nonnegative FISTA with a safe rectangular-operator Lipschitz bound.

    Coverage weights define a diagnostic objective, NOT a noise likelihood.
    CPU/GPU use the same float64 equations, but independent tests compare the
    optimizer with an augmented-design constrained least-squares reference.
    """
    target, weight = np.asarray(target, float), np.asarray(weight, float)
    left, right = np.asarray(left, float), np.asarray(right, float)
    support = np.asarray(support, bool)
    if (target.ndim != 2 or target.shape != weight.shape or left.ndim != 2
            or right.ndim != 2 or support.shape != (left.shape[1], right.shape[1])
            or target.shape != (left.shape[0], right.shape[0])):
        raise ValueError('rectangular projection shape mismatch')
    if (not all(np.isfinite(a).all() for a in (target, weight, left, right))
            or weight.min() < 0 or weight.max() > 1 or not np.any(weight > 0)
            or not np.any(support) or min(left.min(), right.min()) < 0
            or not np.isfinite([regularization, tolerance, refinement]).all()
            or regularization < 0 or tolerance < 0 or refinement < 1
            or not isinstance(max_iterations, int) or max_iterations < 1):
        raise ValueError('invalid fixed-image fit')
    if backend == 'cupy':
        import cupy as xp
        xp.get_default_memory_pool().set_limit(size=1024**3)
        to_host = xp.asnumpy
    elif backend == 'numpy':
        xp = np
        to_host = np.asarray
    else:
        raise ValueError('unsupported backend')
    op_bound = float(np.max(left.sum(0))*np.max(left.sum(1))
                     *np.max(right.sum(0))*np.max(right.sum(1)))
    lipschitz = op_bound + 8*regularization
    if lipschitz <= 0:
        raise ValueError('empty projection objective')
    # Zero-weight values do not affect initialization, normalization or gradients.
    scale = max(float(np.sqrt(np.sum(weight*target**2)/weight.sum())), 1e-12)
    data = xp.asarray(np.where(weight > 0, target/scale, 0))
    w, a, b = xp.asarray(weight), xp.asarray(left), xp.asarray(right)
    mask = xp.asarray(support)
    current = xp.zeros(support.shape, dtype=xp.float64)
    extrapolated = current.copy()
    t = 1.
    history = []
    converged = False

    def gradient(s):
        return adjoint(w*(project(s, a, b)-data), a, b) + regularization*roughness_gradient(s, xp)

    for iteration in range(1, max_iterations+1):
        new = xp.where(mask, xp.maximum(extrapolated-gradient(extrapolated)/lipschitz, 0), 0)
        tnew = (1+np.sqrt(1+4*t*t))/2
        extrapolated = new+(t-1)/tnew*(new-current)
        current, t = new, tnew
        if iteration % 50 == 0 or iteration == max_iterations:
            residual = project(current, a, b)-data
            step = current-xp.where(mask, xp.maximum(current-gradient(current)/lipschitz, 0), 0)
            stationarity = float(xp.sqrt(xp.mean(step**2)))*lipschitz*refinement**2
            objective = .5*xp.sum(w*residual**2) + .5*regularization*sum(
                xp.sum(xp.diff(current, axis=axis)**2) for axis in (0, 1))
            history.append(dict(iteration=iteration, objective=float(objective),
                                scaled_projected_gradient_rms=stationarity))
            if stationarity < tolerance:
                converged = True
                break
    return to_host(current)*scale, dict(converged=converged, iterations=iteration,
        scaled_projected_gradient_rms=stationarity, normalizing_intensity=scale,
        lipschitz_bound=lipschitz, backend=backend, history=history)
