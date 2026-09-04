"""Dimensional algebra used as an EXECUTABLE screen, not as documentation.

The point of this module is that the candidate laws' kernels are written once,
generically over a numeric backend `xp`, and then run twice:

    * with ``xp = numpy``            -> ordinary floating point evaluation
    * with ``xp = dimx``             -> every intermediate carries an (M,L,T)
                                       dimension vector, and ``exp``/``log``
                                       raise ``DimensionError`` the moment a
                                       dimensionful argument reaches them.

So "dimensional consistency" is not asserted by a human reading the formula, it
is decided by running the same code that the solver runs. A law that hides a
dimensionful argument inside an exponential cannot pass by accident.

Convention: dimension vectors are (mass, length, time) exponents, stored as
Fractions so that half-integer powers (sqrt of a squared quantity) stay exact
and do not accumulate float noise.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Tuple

import numpy as np


class DimensionError(ValueError):
    pass


DimTuple = Tuple[Fraction, Fraction, Fraction]


def _dim(m=0, l=0, t=0) -> DimTuple:
    return (Fraction(m), Fraction(l), Fraction(t))


DIMLESS = _dim()
MASS = _dim(m=1)
LENGTH = _dim(l=1)
TIME = _dim(t=1)
ACCEL = _dim(l=1, t=-2)
POTENTIAL = _dim(l=2, t=-2)
DENSITY = _dim(m=1, l=-3)
G_DIM = _dim(m=-1, l=3, t=-2)
TIDAL = _dim(t=-2)          # d^2 Phi / dx^2


def _fmt(d: DimTuple) -> str:
    names = ("M", "L", "T")
    parts = [f"{n}^{e}" for n, e in zip(names, d) if e != 0]
    return "1" if not parts else " ".join(parts)


class Q:
    """A float carrying an (M, L, T) dimension vector."""

    __slots__ = ("v", "d", "tag")

    def __init__(self, v, d: DimTuple = DIMLESS, tag: str = ""):
        self.v = float(v)
        self.d = tuple(Fraction(x) for x in d)
        self.tag = tag

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _coerce(o):
        if isinstance(o, Q):
            return o
        return Q(o, DIMLESS)

    def _same(self, o, op):
        if self.d != o.d:
            raise DimensionError(
                f"cannot {op} [{_fmt(self.d)}]{' ' + self.tag if self.tag else ''}"
                f" and [{_fmt(o.d)}]{' ' + o.tag if o.tag else ''}")

    def __repr__(self):
        return f"Q({self.v:.6g}, [{_fmt(self.d)}]{', ' + self.tag if self.tag else ''})"

    # ---------------------------------------------------------- algebra
    def __add__(self, o):
        o = self._coerce(o)
        self._same(o, "add")
        return Q(self.v + o.v, self.d)

    __radd__ = __add__

    def __sub__(self, o):
        o = self._coerce(o)
        self._same(o, "subtract")
        return Q(self.v - o.v, self.d)

    def __rsub__(self, o):
        return Q._coerce(o).__sub__(self)

    def __neg__(self):
        return Q(-self.v, self.d)

    def __mul__(self, o):
        o = self._coerce(o)
        return Q(self.v * o.v, tuple(a + b for a, b in zip(self.d, o.d)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._coerce(o)
        return Q(self.v / o.v, tuple(a - b for a, b in zip(self.d, o.d)))

    def __rtruediv__(self, o):
        return Q._coerce(o).__truediv__(self)

    def __pow__(self, e):
        if isinstance(e, Q):
            if e.d != DIMLESS:
                raise DimensionError(
                    f"exponent must be dimensionless, got [{_fmt(e.d)}]")
            e = e.v
        f = Fraction(e).limit_denominator(10 ** 6)
        return Q(self.v ** float(e), tuple(a * f for a in self.d))

    def __abs__(self):
        return Q(abs(self.v), self.d)

    def is_dimensionless(self):
        return self.d == DIMLESS


# ---------------------------------------------------------------- backend
# The functions below give `dimx` the same surface as numpy for the small set
# of operations the candidate kernels are allowed to use.

def _guard(x, fn):
    x = Q._coerce(x)
    if x.d != DIMLESS:
        raise DimensionError(
            f"{fn} received a dimensionful argument [{_fmt(x.d)}]"
            f"{' ' + x.tag if x.tag else ''}; interpolation-function arguments "
            f"must be dimensionless")
    return x


def exp(x):
    return Q(np.exp(_guard(x, "exp").v), DIMLESS)


def log(x):
    return Q(np.log(_guard(x, "log").v), DIMLESS)


def tanh(x):
    return Q(np.tanh(_guard(x, "tanh").v), DIMLESS)


def sqrt(x):
    x = Q._coerce(x)
    return x ** Fraction(1, 2)


def abs_(x):
    return abs(Q._coerce(x))


def maximum(a, b):
    a, b = Q._coerce(a), Q._coerce(b)
    a._same(b, "compare")
    return Q(max(a.v, b.v), a.d)


def where(cond, a, b):
    a, b = Q._coerce(a), Q._coerce(b)
    a._same(b, "select between")
    return a if cond else b


# numpy exposes these names too, so a kernel written as `xp.exp(...)` works
# unchanged with xp=np and xp=dimx.
__all__ = ["Q", "DimensionError", "exp", "log", "tanh", "sqrt", "maximum",
           "where", "DIMLESS", "MASS", "LENGTH", "TIME", "ACCEL", "POTENTIAL",
           "DENSITY", "G_DIM", "TIDAL", "_fmt", "_dim"]
