"""Derivative-consistent isolated axisymmetric fields for the length action.

Exact partial Green integrals of one cubic source interpolant supply the
potential and all needed derivatives. No repeated differentiation of an
approximate potential spline occurs. Density projection, interpolation and
boundary errors must be measured separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import gammainc, hyp1f1

from .isolated_axisymmetric import MultipolePotential, solve_poisson
from .length_screening import anomalous_flux
from .reconstructed_axisymmetric import SurfaceDensityDisk


@dataclass
class RegularSurfaceDensityDisk(SurfaceDensityDisk):
    """C1 nonnegative source with a declared inner continuation.

    Match the first measured surface density and its PCHIP slope using a
    quadratic log density in R below the first radius. The derivative vanishes
    at the axis. All measured knots and the old outer taper remain unchanged.
    This is an unmeasured central-source assumption, not extra observations.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.radius[0] <= 0 or self.surface_density[0] <= 0:
            raise ValueError('this registered inner continuation requires a positive first radius and density')
        self.core_coefficient = float(self.interpolator(self.radius[0], 1)/(2*self.radius[0]*self.surface_density[0]))

    def surface_and_derivative(self, R):
        R = np.asarray(R, float)
        if np.any(~np.isfinite(R)) or np.any(R < 0):
            raise ValueError('finite nonnegative radius required')
        r0 = self.radius[0]
        inner_r = np.minimum(R, r0)
        core = self.surface_density[0]*np.exp(self.core_coefficient*(inner_r**2-r0**2))
        sample = np.clip(R, r0, self.radius[-1])
        raw = self.interpolator(sample)
        raw_derivative = self.interpolator(sample, 1)
        phase = np.clip((R-(self.outer_radius-self.taper_width))/self.taper_width, 0, 1)
        taper = .5+.5*np.cos(np.pi*phase)
        taper_derivative = np.where((phase > 0) & (phase < 1), -np.pi/(2*self.taper_width)*np.sin(np.pi*phase), 0.)
        surface = np.where(R < r0, core, raw*taper)
        derivative = np.where(R < r0, 2*self.core_coefficient*R*core, raw_derivative*taper+raw*taper_derivative)
        if np.any(surface < 0) or np.any(~np.isfinite(surface)):
            raise FloatingPointError('nonfinite or negative reconstructed source')
        return surface, derivative

    def surface(self, R):
        return self.surface_and_derivative(R)[0]

    def density_and_gradient(self, R, z):
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        if np.any(~np.isfinite(z)):
            raise ValueError('finite vertical coordinates required')
        surface, derivative = self.surface_and_derivative(R)
        t = np.exp(-2*abs(z)/self.height)
        vertical = 2*t/(1+t)**2/self.height
        density = surface*vertical
        return density, np.array([derivative*vertical, -2*np.tanh(z/self.height)*density/self.height])


class GreenRadialInterpolator:
    """Exact finite-shell Green solution of a C2 source interpolant S_l(log r).

    Cubic exponential moments extend the precomputed inner/outer integrals to
    arbitrary t. The ODE derivatives are exact derivatives of those integrals,
    not independent replacements for derivatives of an approximate potential.
    The absence of potential-value differencing avoids losing tiny inner forces
    to an arbitrary, large monopole potential zero.
    """

    def __init__(self, t, coefficients):
        self.x = np.asarray(t, float)
        self.source = CubicSpline(t, coefficients, axis=0, extrapolate=False)
        self.orders = np.arange(coefficients.shape[1])
        # Factor r^2 analytically. Interpolating r^2*S instead would produce a
        # spurious derivative of a constant-density core after dividing by r^2.
        self.inner = self.convolved(t, coefficients, self.orders+3)
        self.outer = self.convolved(-t[::-1], coefficients[::-1], self.orders-2)[::-1]

    @staticmethod
    def moments(decay, span):
        lam = np.asarray(decay)
        u = np.asarray(span)[:, None]
        answer = []
        for j, factorial in enumerate([1, 1, 2, 6]):
            moment = np.empty((len(span), len(lam)))
            positive, zero, negative = lam > 0, lam == 0, lam < 0
            moment[:, positive] = factorial*gammainc(j+1, lam[positive]*u)/lam[positive]**(j+1)
            moment[:, zero] = u**(j+1)/(j+1)
            moment[:, negative] = u**(j+1)/(j+1)*hyp1f1(j+1, j+2, -lam[negative]*u)
            answer.append(moment)
        return answer

    @classmethod
    def convolved(cls, t, values, decay):
        spline = CubicSpline(t, values, axis=0)
        h = np.diff(t)
        a, b, c, _ = spline.c
        coefficients = [values[1:], -(3*a*h[:, None]**2+2*b*h[:, None]+c), 3*a*h[:, None]+b, -a]
        increments = sum(coefficient*moment for coefficient, moment in zip(coefficients, cls.moments(decay, h), strict=True))
        factors = np.exp(-h[:, None]*decay)
        integral = np.zeros_like(values)
        for i in range(len(t)-1):
            integral[i+1] = factors[i]*integral[i]+increments[i]
        return integral

    def jet(self, t):
        original = np.asarray(t, float)
        if np.any(~np.isfinite(original)) or np.any(original < self.x[0]) or np.any(original > self.x[-1]):
            raise ValueError('log-radius inside declared Green integration domain required')
        t = original.ravel()
        index = np.clip(np.searchsorted(self.x, t, side='right')-1, 0, len(self.x)-2)
        u = t-self.x[index]
        v = self.x[index+1]-t
        a, b, c, d = self.source.c[:, index, :]
        f = ((a*u[:, None]+b)*u[:, None]+c)*u[:, None]+d
        fp = (3*a*u[:, None]+2*b)*u[:, None]+c
        half_fpp = 3*a*u[:, None]+b
        moments_i = self.moments(self.orders+3, u)
        moments_o = self.moments(self.orders-2, v)
        inner = np.exp(-u[:, None]*(self.orders+3))*self.inner[index]
        outer = np.exp(-v[:, None]*(self.orders-2))*self.outer[index+1]
        for j, coefficient in enumerate([f, fp, half_fpp, a]):
            inner += (-1)**j*coefficient*moments_i[j]
            outer += coefficient*moments_o[j]
        r2 = np.exp(2*t[:, None])
        F = -r2*(inner+outer)/(2*self.orders+1)
        Ft = r2*((self.orders+1)*inner-self.orders*outer)/(2*self.orders+1)
        L = self.orders*(self.orders+1)
        Ftt = r2*f-Ft+L*F
        Fttt = r2*(2*f+fp)-Ftt+L*Ft
        shape = original.shape+(len(self.orders),)
        return [value.reshape(shape) for value in [F, Ft, Ftt, Fttt]]

    def __call__(self, t, nu=0):
        if type(nu) is not int or not 0 <= nu <= 3:
            raise ValueError('only the four analytically implemented radial jets are available')
        return self.jet(t)[nu]


@dataclass
class C3MultipolePotential(MultipolePotential):
    radial_source_spline: CubicSpline

    @classmethod
    def build(cls, grid, source):
        base = solve_poisson(grid, source)
        return cls.from_green_solution(base)

    @classmethod
    def from_green_solution(cls, base):
        t = base.spline.x
        spline = GreenRadialInterpolator(t, base.source_coefficients)
        return cls(base.grid, spline, base.source_coefficients, spline.source)

    def fields(self, R, z, *, batch_size=4096):
        """Return physical components in the orthonormal (r,theta,phi) basis.

        Hessian norm and trace are scalars. Their gradients can be computed by
        differentiating the spherical components; derivatives of the rotating
        basis cancel in these invariants. No density-based replacement of the
        trace or its derivative is made inside the action flux.
        """
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        shape = R.shape
        radii = np.hypot(R, z)
        if (np.any(R < 0) or np.any(~np.isfinite(radii)) or
                np.any(radii < self.grid.r_min*(1-1e-12)) or np.any(radii > self.grid.r_max*(1+1e-12))):
            raise ValueError('coordinates inside the declared finite radial domain required')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('positive batch size required')
        flat_r = np.clip(radii, self.grid.r_min, self.grid.r_max).ravel()
        flat_R, flat_z = R.ravel(), z.ravel()
        values = np.empty((12, len(flat_r)))
        for start in range(0, len(flat_r), batch_size):
            end = min(start+batch_size, len(flat_r))
            r = flat_r[start:end]
            s, mu = flat_R[start:end]/r, flat_z[start:end]/r
            t = np.log(r)
            unique_t, inverse = np.unique(t, return_inverse=True)
            f, ft, ftt, fttt = [value[inverse] for value in self.spline.jet(unique_t)]
            sums = np.zeros((10, len(r)))
            p, dp, ddp, dddp = np.ones_like(r), np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)
            old, dold, ddold, dddold = [np.zeros_like(r) for _ in range(4)]
            for order in range(self.grid.l_max+1):
                sums += np.array([f[:, order]*p, ft[:, order]*p, ftt[:, order]*p, fttt[:, order]*p,
                                  f[:, order]*dp, ft[:, order]*dp, ftt[:, order]*dp,
                                  f[:, order]*ddp, ft[:, order]*ddp, f[:, order]*dddp])
                k = 2*order+1
                nxt = (k*mu*p-order*old)/(order+1)
                dnxt = (k*(p+mu*dp)-order*dold)/(order+1)
                ddnxt = (k*(2*dp+mu*ddp)-order*ddold)/(order+1)
                dddnxt = (k*(3*ddp+mu*dddp)-order*dddold)/(order+1)
                old, p, dold, dp, ddold, ddp, dddold, dddp = p, nxt, dp, dnxt, ddp, ddnxt, dddp, dddnxt
            psi, B, C, D, E, F, Q, H, I, J = sums
            hrr, hrt = (C-B)/r**2, -s*(F-E)/r**2
            htt, hpp = (B+s*s*H-mu*E)/r**2, (B-mu*E)/r**2
            trr, trt = (D-3*C+2*B)/r**2, -s*(Q-3*F+2*E)/r**2
            ttt, tpp = (C-2*B+s*s*(I-2*H)-mu*(F-2*E))/r**2, (C-2*B-mu*(F-2*E))/r**2
            arr, art = -s*(Q-F)/r**2, (-mu*(F-E)+s*s*(I-H))/r**2
            att, app = -s*(F+s*s*J-3*mu*H-E)/r**2, -s*(F-E-mu*H)/r**2
            grad_norm_r = 2*(hrr*trr+2*hrt*trt+htt*ttt+hpp*tpp)/r
            grad_norm_t = 2*(hrr*arr+2*hrt*art+htt*att+hpp*app)/r
            values[:, start:end] = [psi, B/r, -s*E/r, hrr, hrt, htt, hpp,
                                    grad_norm_r, grad_norm_t, (trr+ttt+tpp)/r, (arr+att+app)/r,
                                    hrr+htt+hpp]
        a = values.reshape((12,)+shape)
        return {'potential': a[0], 'gradient_r_theta': a[1:3], 'hessian_rr_rt_tt_pp': a[3:7],
                'gradient_hessian_norm_r_theta': a[7:9], 'gradient_laplacian_r_theta': a[9:11], 'laplacian': a[11]}


def full_length_flux(fields, spec, length, a0):
    """Full anomalous flux, using two tangential directions in three dimensions."""
    p2 = fields['gradient_r_theta']
    p = np.concatenate([p2, np.zeros_like(p2[:1])])
    H = np.zeros((3, 3)+p2.shape[1:])
    hrr, hrt, htt, hpp = fields['hessian_rr_rt_tt_pp']
    H[0, 0], H[0, 1], H[1, 0], H[1, 1], H[2, 2] = hrr, hrt, hrt, htt, hpp
    hn = fields['gradient_hessian_norm_r_theta']
    lap = fields['gradient_laplacian_r_theta']
    return anomalous_flux(spec, p, H, np.concatenate([hn, np.zeros_like(hn[:1])]),
                           np.concatenate([lap, np.zeros_like(lap[:1])]), length, a0)[:2]
