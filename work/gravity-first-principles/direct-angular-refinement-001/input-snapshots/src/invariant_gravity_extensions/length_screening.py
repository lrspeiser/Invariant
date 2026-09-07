"""Bounded GQUMOND length response, with its full second-gradient variation.

P=x+x*K(x+h), K(u)=[Q_saturated(u)-u]/u. This is a static action ansatz
within Milgrom's arXiv:2305.01589 framework, not a covariant admission.
The generic flux uses physical units consistently; point-source routines use
GM=a0=1 and ell measured in the MOND radius sqrt(GM/a0).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import pairwise

import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=12)
def nodes(n):
    if type(n) is not int or not 16 <= n <= 2048:
        raise ValueError('16 to 2048 Gauss nodes required')
    return leggauss(n)


@dataclass(frozen=True)
class LengthScreening:
    shape: float = 1.0
    epsilon: float = 1e-6

    def __post_init__(self):
        if self.shape not in {.5, 1, 2} or not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError('registered shape and finite positive epsilon required')

    def excess_derivatives(self, u):
        """E', E'', E''' for E=Q-u, with positive regularized argument."""
        v = np.asarray(u)+self.epsilon**2
        log_v = np.log(v)
        t = np.exp(-np.logaddexp(0, -self.shape*log_v))
        d1 = np.exp(-.25*log_v-(1+.75/self.shape)*np.logaddexp(0, self.shape*log_v))
        c = .25+(self.shape+.75)*t
        d2 = -d1*c/v
        d3 = d1*(c*c+c-self.shape*(self.shape+.75)*t*(1-t))/v**2
        return d1, d2, d3

    def kernel(self, u):
        """K, u K', u^2 K'', E', avoiding origin and high-gradient cancellation."""
        u = np.asarray(u, dtype=float)
        if np.any(~np.isfinite(u)) or np.any(u < 0):
            raise ValueError('finite nonnegative invariant required')
        original_shape = u.shape
        flat = u.ravel()
        k, k1, k2 = [np.empty_like(flat) for _ in range(3)]
        d1, d2, _ = self.excess_derivatives(flat)
        small = flat < self.epsilon**2
        if np.any(small):
            # K^(j)(u)=integral_0^1 t^j E^(j+1)(tu) dt. Here the
            # regularized argument varies by less than a factor of two.
            abscissa, weights = nodes(16)
            t, w = (abscissa+1)/2, weights/2
            v = flat[small, None]
            e1, e2, e3 = self.excess_derivatives(v*t)
            k[small] = e1@w
            k1[small] = flat[small]*((e2*t)@w)
            k2[small] = flat[small]**2*((e3*t*t)@w)
        large = ~small
        if np.any(large):
            v = flat[large]+self.epsilon**2
            log_s = -.75/self.shape*np.logaddexp(0, -self.shape*np.log(v))
            log_s0 = -.75/self.shape*np.logaddexp(0, -2*self.shape*np.log(self.epsilon))
            excess = (4/3)*np.exp(log_s)*(-np.expm1(log_s0-log_s))
            k[large] = excess/flat[large]
            k1[large] = d1[large]-k[large]
            k2[large] = flat[large]*d2[large]-2*k1[large]
        values = tuple(a.reshape(original_shape) for a in (k, k1, k2, d1))
        if not all(np.all(np.isfinite(a)) for a in values):
            raise FloatingPointError('nonfinite length kernel')
        return values

    def partials(self, x, h):
        """Return P_x-1, P_h and scaled derivatives of K."""
        x, h = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(h, dtype=float))
        if np.any(x < 0) or np.any(h < 0):
            raise ValueError('nonnegative x and h required')
        u = x+h
        k, k1, k2, d1 = self.kernel(u)
        fraction = np.divide(x, u, out=np.zeros_like(u), where=u > 0)
        complement = np.divide(h, u, out=np.zeros_like(u), where=u > 0)
        px = complement*k+fraction*d1
        px = np.where(u == 0, d1, px)
        return px, fraction*k1, k1, k2, fraction

    def value(self, x, h):
        x, h = np.broadcast_arrays(x, h)
        return x*(1+self.kernel(x+h)[0])

    def card(self, length_pc, a0_m_s2):
        if not np.isfinite(length_pc) or length_pc < 0 or not np.isfinite(a0_m_s2) or a0_m_s2 <= 0:
            raise ValueError('nonnegative physical length and positive acceleration required')
        row = {'schema': 'bounded-length-action-1', 'shape': self.shape, 'epsilon': self.epsilon,
               'length_pc': length_pc, 'a0_m_s2': a0_m_s2,
               'action': 'P=x+x*(Q_m(x+h)-(x+h))/(x+h), with removable origin',
               'flux': 'J_i=P_x*psi_i-ell^2*partial_j(P_h*psi_ij)',
               'prior_art': 'https://arxiv.org/html/2305.01589v2',
               'scope': 'nonrelativistic_static_ansatz', 'photon_sector': 'not_derived',
               'historical_novelty_claimed': False, 'empirical_admission': False}
        raw = json.dumps(row, sort_keys=True, separators=(',', ':'))
        return {**row, 'card_sha256': sha256(raw.encode()).hexdigest()}


def anomalous_flux(spec, gradient, hessian, gradient_hessian_norm_squared, gradient_laplacian, length, a0=1.0):
    """Full Cartesian flux; H must include all three spatial directions.

    gradient_hessian_norm_squared is grad(sum_ij H_ij^2), NOT just a
    meridional contraction. Source-density gradients enter gradient_laplacian.
    """
    p, H, dH2, dlap = map(np.asarray, (gradient, hessian, gradient_hessian_norm_squared, gradient_laplacian))
    if (p.shape[0] != 3 or H.shape != (3, *p.shape) or dH2.shape != p.shape or dlap.shape != p.shape
            or any(np.any(~np.isfinite(v)) for v in (p, H, dH2, dlap))
            or not np.isfinite(length) or length < 0 or not np.isfinite(a0) or a0 <= 0):
        raise ValueError('consistent finite Cartesian fields and physical units required')
    x = np.sum(p*p, axis=0)/a0**2
    h = length**2*np.sum(H*H, axis=(0, 1))/a0**2
    px, ph, k1, k2, fraction = spec.partials(x, h)
    if length == 0:
        return px*p
    dx = 2*np.einsum('ij...,j...->i...', H, p)/a0**2
    dh = length**2*dH2/a0**2
    u = x+h
    dph = np.divide((k1+fraction*k2)*dx+fraction*k2*dh, u,
                     out=np.zeros_like(p, dtype=float), where=u > 0)
    result = px*p-length**2*(np.einsum('ij...,j...->i...', H, dph)+ph*dlap)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError('nonfinite variational flux')
    return result


def point_external_flux(spec, radius, mu, eta_newtonian, length):
    """Analytic point-source Hessian in a constant field; returns polar flux."""
    r, mu = np.broadcast_arrays(np.asarray(radius, dtype=float), np.asarray(mu, dtype=float))
    if (np.any(r <= 0) or np.any(~np.isfinite(r)) or np.any(~np.isfinite(mu)) or np.any(abs(mu) > 1)
            or not np.isfinite(eta_newtonian) or eta_newtonian < 0 or not np.isfinite(length) or length < 0):
        raise ValueError('positive finite radius, physical angle, background and length required')
    sine = np.sqrt(np.maximum(0, 1-mu*mu))
    pr, pt = 1/r**2-eta_newtonian*mu, eta_newtonian*sine
    x = pr*pr+pt*pt
    h = 6*length**2/r**6
    px, ph, k1, k2, fraction = spec.partials(x, h)
    radial, polar = px*pr, px*pt
    if length > 0:
        u = x+h
        dph_r = ((k1+fraction*k2)*(-4*pr/r**3)+fraction*k2*(-6*h/r))/u
        dph_t = (k1+fraction*k2)*(2*pt/r**3)/u
        radial = radial+2*length**2*dph_r/r**3
        polar = polar-length**2*dph_t/r**3
    return {'radial': radial, 'polar': polar, 'first_radial': px*pr, 'first_polar': px*pt,
            'P_h': ph, 'radius': r}


def point_monopole_delta(spec, y, length):
    """Fractional inward anomaly at g_N/a0=y, length in this mass's MOND radius."""
    y = np.asarray(y, dtype=float)
    if np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError('positive finite Newtonian acceleration ratio required')
    return point_external_flux(spec, 1/np.sqrt(y), np.zeros_like(y), 0, length)['radial']/y


def point_quadrupole(spec, eta_newtonian, length, *, quadrature_nodes=128):
    """Infinite-domain flux and independently integrated action representations.

    Q2_dim=4.5 integral dv dmu [J_r P2+J_theta mu sin(theta)].
    Integrating the higher-derivative action term twice against P2/r^3
    instead adds 54 ell^2 integral dv v^4 dmu P_h P2 to the first-gradient
    flux. Their surface terms vanish at infinity and the central point for
    this bounded kernel. Both representations are retained and compared.
    """
    if not np.isfinite(eta_newtonian) or eta_newtonian < 0 or not np.isfinite(length) or length < 0:
        raise ValueError('nonnegative finite background and length required')
    abscissa, weights = nodes(quadrature_nodes)
    if eta_newtonian == 0:
        return {'Q2_flux': 0.0, 'Q2_action': 0.0, 'absolute_agreement': 0.0, 'segments': []}
    t, w = (abscissa+1)/2, weights/2
    mu = t[None, :]
    P2, angular = (3*mu*mu-1)/2, mu*np.sqrt(1-mu*mu)
    # v=1/r. Split at the external saddle and action length scales; no
    # empirical result changes these deterministic integration intervals.
    scales = [1.0, np.sqrt(eta_newtonian)]
    if length > 0:
        scales += [(1/(6*length**2))**(1/6), (eta_newtonian**2/(6*length**2))**(1/6)]
    cuts = np.unique([0.0, *scales])
    intervals = [(lo+(hi-lo)*t, (hi-lo)*w) for lo, hi in pairwise(cuts)]
    intervals.append((cuts[-1]/(1-t), cuts[-1]*w/(1-t)**2))
    totals = np.zeros(2)
    segments = []
    for v, vw in intervals:
        radius = 1/v[:, None]
        plus = point_external_flux(spec, radius, mu, eta_newtonian, length)
        minus = point_external_flux(spec, radius, -mu, eta_newtonian, length)
        middle = point_external_flux(spec, radius, 0, eta_newtonian, length)
        # Subtract angular constants whose P2 integral is zero. Pair +/-mu
        # before integration so a uniform background cannot leave a tail.
        flux = (plus['radial']+minus['radial']-2*middle['radial'])*P2+(plus['polar']-minus['polar'])*angular
        first = (plus['first_radial']+minus['first_radial']-2*middle['first_radial'])*P2+(plus['first_polar']-minus['first_polar'])*angular
        tensor = (plus['P_h']+minus['P_h']-2*middle['P_h'])*P2
        values = np.array([4.5*vw@(flux@w), 4.5*vw@(first@w)+54*length**2*vw@(v**4*(tensor@w))])
        if not np.all(np.isfinite(values)):
            raise FloatingPointError('nonfinite quadrupole integral')
        totals += values
        segments.append(values.tolist())
    return {'Q2_flux': float(totals[0]), 'Q2_action': float(totals[1]),
            'absolute_agreement': float(abs(totals[0]-totals[1])), 'segments': segments}
