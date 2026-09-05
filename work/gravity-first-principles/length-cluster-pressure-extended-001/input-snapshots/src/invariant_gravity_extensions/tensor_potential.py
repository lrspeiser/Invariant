"""C3 tensor Hermite potential with regular cylindrical axis derivatives.

All field components are derivatives of one piecewise polynomial. Source
identities and interpolation error still require independent validation.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .potential_join import cartesian_tensors, pack_cartesian

_Q = [[1., 4., 10., 20.], [0., 1., 4., 10.], [0., 0., .5, 2.], [0., 0., 0., 1/6]]


def _coefficient_table():
    left = [[Fraction(v) for v in row] for row in [
        [1, 0, 0, 0, -35, 84, -70, 20], [0, 1, 0, 0, -20, 45, -36, 10],
        [0, 0, Fraction(1, 2), 0, -5, 10, Fraction(-15, 2), 2],
        [0, 0, 0, Fraction(1, 6), Fraction(-2, 3), 1, Fraction(-2, 3), Fraction(1, 6)]]]
    right = [[sum((-1)**(j+a)*math.comb(n, a)*row[n] for n in range(a, 8)) for a in range(8)]
             for j, row in enumerate(left)]
    return np.array([left, right], float)


_COEFFICIENTS = _coefficient_table()


def hermite_basis(t, width):
    """Physical derivative order, endpoint, stored derivative order, point.

Factoring the fourth-order endpoint zero avoids cancellation near a cell edge.
"""
    result = np.empty((4, 2, 4, len(t)))
    for endpoint, q in enumerate([t, 1-t]):
        derivatives = [[np.polynomial.polynomial.polyval(q, np.polynomial.polynomial.polyder(row, m))
                        for m in range(4)] for row in _Q]
        for order in range(4):
            for j in range(4):
                value = sum(math.comb(order, k)*(-1)**k*math.factorial(4)/math.factorial(4-k)*
                    (1-q)**(4-k)*derivatives[j][order-k] for k in range(order+1))
                result[order, endpoint, j] = value*width**(j-order)*((-1)**(j+order) if endpoint else 1)
    return result


class C3TensorPotential:
    """Even-in-z axisymmetric potential on a nonuniform first-quadrant grid.

mixed[i,j,a,b] is the physical partial R^i z^j of the potential, i,j=0..3.
Exact odd-derivative zeros on both symmetry axes are required, not inferred
by subtracting nearly equal field values during evaluation.
"""

    def __init__(self, radius, height, mixed):
        self.radius, self.height, self.mixed = [np.array(a, dtype=float, copy=True) for a in (radius, height, mixed)]
        for grid in [self.radius, self.height]:
            if grid.ndim != 1 or len(grid) < 2 or grid[0] != 0 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
                raise ValueError('finite strictly increasing grids starting at zero required')
        if self.mixed.shape != (4, 4, len(self.radius), len(self.height)) or np.any(~np.isfinite(self.mixed)):
            raise ValueError('finite aligned mixed derivatives through order three in each coordinate required')
        if np.any(self.mixed[1::2, :, 0, :] != 0) or np.any(self.mixed[:, 1::2, :, 0] != 0):
            raise ValueError('exact odd-derivative symmetry zeros required at R=0 and z=0')

    def fields(self, radius, height, *, batch_size=4096):
        R, z = np.broadcast_arrays(np.asarray(radius, float), np.asarray(height, float))
        if (np.any(~np.isfinite(R)) or np.any(~np.isfinite(z)) or np.any(R < 0) or
                np.any(R > self.radius[-1]) or np.any(abs(z) > self.height[-1])):
            raise ValueError('finite coordinates inside the declared interpolation domain required')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('positive integer batch size required')
        shape = R.shape
        rr, zz = R.ravel(), z.ravel()
        values = np.empty((13, rr.size))
        for start in range(0, rr.size, batch_size):
            stop = min(start+batch_size, rr.size)
            r, h = rr[start:stop], abs(zz[start:stop])
            sign = np.where(zz[start:stop] < 0, -1., 1.)
            ir = np.clip(np.searchsorted(self.radius, r, side='right')-1, 0, len(self.radius)-2)
            iz = np.clip(np.searchsorted(self.height, h, side='right')-1, 0, len(self.height)-2)
            dr, dz = np.diff(self.radius)[ir], np.diff(self.height)[iz]
            tr, tz = (r-self.radius[ir])/dr, (h-self.height[iz])/dz
            br, bz = hermite_basis(tr, dr), hermite_basis(tz, dz)
            data = np.array([[self.mixed[:, :, ir+e, iz+f] for f in range(2)] for e in range(2)])
            anchor = data[0, 0, 0, 0].copy()
            data[:, :, 0, 0] -= anchor

            def partial(i, j, br=br, bz=bz, data=data, sign=sign, anchor=anchor):
                value = np.einsum('ein,fjn,efijn->n', br[i], bz[j], data, optimize=True)*sign**j
                return value+anchor if i == j == 0 else value

            psi, pr, pz = partial(0, 0), partial(1, 0), partial(0, 1)
            hrr, hrz, hzz = partial(2, 0), partial(1, 1), partial(0, 2)
            trrr, trrz, trzz, tzzz = partial(3, 0), partial(2, 1), partial(1, 2), partial(0, 3)
            hpp = np.divide(pr, r, out=np.zeros_like(r), where=r > 0)
            trpp = np.divide(hrr-hpp, r, out=np.zeros_like(r), where=r > 0)
            tzpp = np.divide(hrz, r, out=np.zeros_like(r), where=r > 0)
            first = ir == 0
            if np.any(first):
                # In the first radial cell C1=C3=0 identically. Evaluate the
                # quotient polynomials directly instead of cancelling Hrr-Hpp.
                coeff = np.einsum('eia,in,fjn,efijn->an', _COEFFICIENTS,
                    dr[None, :]**np.arange(4)[:, None], bz[0], data, optimize=True)
                coeff_z = np.einsum('eia,in,fjn,efijn->an', _COEFFICIENTS,
                    dr[None, :]**np.arange(4)[:, None], bz[1], data, optimize=True)
                hpp[first] = sum(a*coeff[a, first]*tr[first]**(a-2) for a in range(2, 8))/dr[first]**2
                trpp[first] = sum(a*(a-2)*coeff[a, first]*tr[first]**(a-3) for a in range(4, 8))/dr[first]**3
                tzpp[first] = sum(a*coeff_z[a, first]*tr[first]**(a-2) for a in range(2, 8))/dr[first]**2*sign[first]
            values[:, start:stop] = [psi, pr, pz, hrr, hrz, hzz, hpp, trrr, trrz, trzz, tzzz, trpp, tzpp]
        v = values.reshape((13,)+shape)
        temporary = {'potential': v[0], 'gradient_R_z': v[1:3], 'hessian_RR_Rz_zz_pp': v[3:7],
                     'third_RRR_RRz_Rzz_zzz_Rpp_zpp': v[7:13]}
        return pack_cartesian(*cartesian_tensors(temporary), R, z)
