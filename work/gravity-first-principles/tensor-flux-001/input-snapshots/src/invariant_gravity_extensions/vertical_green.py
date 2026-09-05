"""Stable off-plane vertical Green jets of one explicit C1 source.

A cubic spline approximates the positive even sech-squared profile inside a
declared dimensionless height. A C1 exponential continuation supplies infinite
tails. Exact polynomial Green moments and weak source derivatives supply the
first three derivatives without high-wavenumber contact-term cancellation.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import gammainc


def exponential_moments(a, span):
    """Integral_0^span t^j exp(-a*t) dt, j=0,...,3, including zero a/span."""
    a = np.asarray(a, float)
    if a.ndim != 1 or np.any(~np.isfinite(a)) or np.any(a < 0) or not np.isfinite(span) or span < 0:
        raise ValueError('nonnegative finite decays and interval length required')
    result = np.empty((4, len(a)))
    positive = a > 0
    for j, factorial in enumerate([1., 1., 2., 6.]):
        result[j, positive] = factorial*gammainc(j+1, a[positive]*span)/a[positive]**(j+1)
        result[j, ~positive] = span**(j+1)/(j+1)
    return result


def exponential_convolution(a, rate, span):
    """Integral_0^span exp(-rate*t-a*(span-t)) dt, stable at a=rate."""
    a = np.asarray(a)
    difference = abs(a-rate)
    divided = np.divide(-np.expm1(-difference*span), difference,
                        out=np.full_like(a, span, dtype=float), where=difference > 0)
    return np.exp(-np.minimum(a, rate)*span)*divided


class Sech2VerticalGreen:
    """Dimensionless source f(u) ~ sech²(u)/2, with exactly normalized mass.

    `jet(a,u)` returns convolution of f with exp(-a*abs(u-v)) and its first
    three u derivatives. Positive-half source knots avoid numerical parity
    leakage. Negative u is evaluated by exact reflection.
    """

    def __init__(self, *, intervals=2400, extent=24.):
        if type(intervals) is not int or intervals < 32 or not np.isfinite(extent) or extent < 4:
            raise ValueError('at least 32 intervals and a finite extent >=4 required')
        self.nodes = np.linspace(0., extent, intervals+1)
        self.extent = float(extent)
        t = np.exp(-2*self.nodes)
        density = 2*t/(1+t)**2
        self.tail_rate = float(2*np.tanh(extent))
        self.spline = CubicSpline(self.nodes, density, bc_type=((1, 0.), (1, -self.tail_rate*density[-1])))
        self.unnormalized_mass = float(2*(self.spline.integrate(0, extent)+density[-1]/self.tail_rate))
        self.spline.c /= self.unnormalized_mass
        self.tail_density = float(density[-1]/self.unnormalized_mass)
        self.third_derivative_jump = self.tail_rate**2*self.tail_density-float(self.spline(extent, 2))
        # The spline and tail join in f and f'. The weak third derivative also
        # contains opposite point terms at +/-extent from the small jump in f''.

    def source(self, u):
        u = np.asarray(u, float)
        if np.any(~np.isfinite(u)):
            raise ValueError('finite dimensionless heights required')
        q = abs(u)
        clipped = np.minimum(q, self.extent)
        outside = q > self.extent
        tail = self.tail_density*np.exp(-self.tail_rate*np.maximum(q-self.extent, 0))
        density = np.where(outside, tail, self.spline(clipped))
        derivative = np.where(outside, -self.tail_rate*tail, self.spline(clipped, 1))*np.sign(u)
        return density, derivative

    @staticmethod
    def local_contribution(derivatives, moments, direction):
        result = np.zeros_like(moments)
        for n in range(4):
            for j, factorial in enumerate([1., 1., 2., 6.][:4-n]):
                result[n] += direction**j*derivatives[n+j]*moments[j]/factorial
        return result

    def polynomial_jet(self, index, distance):
        a, b, c, d = self.spline.c[:, index]
        return np.array([((a*distance+b)*distance+c)*distance+d,
                         (3*a*distance+2*b)*distance+c, 6*a*distance+2*b, 6*a])

    def jet(self, a, u):
        a, u = np.asarray(a, float), np.asarray(u, float)
        if (a.ndim != 1 or u.ndim != 1 or np.any(~np.isfinite(a)) or np.any(a < 0) or np.any(~np.isfinite(u))):
            raise ValueError('finite nonnegative decay vector and height vector required')
        unique, inverse = np.unique(abs(u), return_inverse=True)
        inside = unique <= self.extent
        bins = np.minimum(np.searchsorted(self.nodes, unique, side='right')-1, len(self.nodes)-2)
        by_bin = {}
        for index in np.flatnonzero(inside):
            by_bin.setdefault(bins[index], []).append(index)
        answer = np.zeros((4, len(unique), len(a)))
        current = (-self.tail_rate)**np.arange(4)[:, None]*self.tail_density/(a+self.tail_rate)
        cache = {}

        def moments(span):
            if span not in cache:
                cache[span] = exponential_moments(a, span)
            return cache[span]

        for j in range(len(self.nodes)-2, -1, -1):
            left, right = self.nodes[j:j+2]
            span = right-left
            for index in by_bin.get(j, []):
                distance = unique[index]-left
                remainder = right-unique[index]
                answer[:, index] = np.exp(-a*remainder)*current+self.local_contribution(
                    self.polynomial_jet(j, distance), moments(remainder), 1)
            current = np.exp(-a*span)*current+self.local_contribution(self.polynomial_jet(j, 0.), moments(span), 1)
        current = current*np.array([1., -1., 1., -1.])[:, None]
        for j in range(len(self.nodes)-1):
            left, right = self.nodes[j:j+2]
            span = right-left
            for index in by_bin.get(j, []):
                distance = unique[index]-left
                answer[:, index] += np.exp(-a*distance)*current+self.local_contribution(
                    self.polynomial_jet(j, distance), moments(distance), -1)
            current = np.exp(-a*span)*current+self.local_contribution(self.polynomial_jet(j, span), moments(span), -1)
        for index in np.flatnonzero(~inside):
            distance = unique[index]-self.extent
            tail_integral = (exponential_convolution(a, self.tail_rate, distance)+
                             np.exp(-self.tail_rate*distance)/(a+self.tail_rate))
            answer[:, index] = np.exp(-a*distance)*current+(-self.tail_rate)**np.arange(4)[:, None]*self.tail_density*tail_integral
        answer[3] += self.third_derivative_jump*(np.exp(-a[None, :]*abs(unique[:, None]-self.extent))-
                                                  np.exp(-a[None, :]*(unique[:, None]+self.extent)))
        answer = answer[:, inverse]
        answer[1::2] *= np.sign(u)[None, :, None]
        return answer
