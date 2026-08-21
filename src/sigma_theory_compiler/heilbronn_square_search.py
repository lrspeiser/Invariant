"""Exact-rational search for Heilbronn configurations of points in the unit square.

The Heilbronn problem for squares asks for ``n`` points in the unit square maximising the
area ``A`` of the *smallest* triangle any three of them span.  Best-known values are
catalogued by Erich Friedman, *Erich's Packing Center -- The Heilbronn Problem for Squares*
(https://erich-friedman.github.io/packing/heilbronn/).  Those values are lower bounds
established by explicit configurations, so a better configuration settles a better bound and
the configuration is its own proof.  This module is built to produce such a proof, or to fail
loudly and report the gap.

**Why this objective is exactly checkable.**  Put every point on the lattice
``{0, 1, ..., D}^2`` and read it as the rational point ``(a/D, b/D)``.  For three lattice
points the doubled, ``D^2``-scaled area

    ``det = (a_j - a_i) * (b_k - b_i) - (a_k - a_i) * (b_j - b_i)``

is an **integer**, and the triangle area is exactly ``|det| / (2 D^2)``.  The score of a
configuration is therefore

    ``A = min_{i<j<k} |det_ijk| / (2 D^2)``

-- a rational number computed from integers only.  :func:`certify` evaluates it with Python's
unbounded integers over *all* ``C(n, 3)`` triples, with no sampling and no floating point
anywhere on the verification path, and compares it to the published record as an exact
:class:`~fractions.Fraction`.  A search that cheats cannot survive the certificate, because
the certificate never consults the search.

**What the search does.**  Two stages, neither of which is trusted.

*Stage one -- sequential linear programming in the reals.*  Maximising the smallest triangle
is ``max t  s.t.  s_ijk * det_ijk(P) >= t``, where ``s_ijk`` is the (locally constant) sign of
the triple's determinant.  Each ``det`` is a quadratic in the coordinates with an exact
gradient, so linearising every near-active triple about the current configuration and adding
a trust region gives a small sparse LP in ``2n + 1`` variables.  Solving it, stepping, and
re-linearising is the standard sequential-LP treatment of a maximin problem and it converges
to sharp, many-triples-tight configurations that coordinate descent never finds.

*Stage two -- projection onto the lattice and integer polish.*  The real configuration is
rounded to ``{0..D}^2`` and then refined by a feasibility ladder: fix an integer target ``T``,
minimise ``sum over triples of max(0, 1 - |det| / T)^2`` (zero exactly when ``A >= T/(2D^2)``),
raise ``T`` on success and back off on failure.  Moving one point touches only the
``C(n-1, 2)`` triples through it, so a whole batch of candidate positions is scored in one
vectorised integer pass.  Lattice arithmetic uses ``int64`` under an explicit overflow guard
(:func:`_check_lattice_width`) -- but nothing depends on that, since only :func:`certify`
speaks, and it speaks in unbounded Python integers.

**Honest limits.**  The search is a heuristic; it establishes lower bounds and never upper
bounds.  Failing to reach the published record says nothing about the record.  Absence of a
better configuration from this search is not evidence that none exists, and this module never
reports one.  ``beats_record`` is decided by exact rational comparison against
:data:`PUBLISHED_RECORDS`, whose entries carry the printed value, finder and year so the
comparison can be audited against the source page.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .sigma_core import canonical_sha256

__all__ = [
    "PUBLISHED_RECORDS",
    "HeilbronnCertificate",
    "PublishedRecord",
    "certify",
    "decimal_floor",
    "exact_min_double_area",
    "search_configuration",
]

RECORD_SOURCE = (
    "Erich Friedman, Erich's Packing Center, 'The Heilbronn Problem for Squares', "
    "https://erich-friedman.github.io/packing/heilbronn/ (page read 2026-08-20)"
)


@dataclass(frozen=True)
class PublishedRecord:
    """One row of the published best-known table, kept verbatim so it can be audited.

    ``printed`` is the string as it appears on the source page.

    Two cases, and the difference decides what counts as a win.  When the page gives the
    record as an *exact* rational (``1/2``, ``1/8``, ``1/27``, ``7/341``) the record is known
    to the last bit, so ``strict`` is set and beating it means ``area > lower_bound`` --
    reproducing ``1/27`` for ``n = 11`` is a reproduction, not a record.  Otherwise the digits
    are a truncation flagged with ``+``, so the record lies in ``[digits, digits + 10**-d)``
    and only ``area >= digits + 10**-d`` is *certainly* an improvement.  Both rules refuse to
    call a tie a win, which is the point.

    ``lower_bound`` and ``beat_threshold`` are strings that :class:`~fractions.Fraction`
    parses exactly, so no comparison ever passes through a float.
    """

    n: int
    printed: str
    lower_bound: str
    beat_threshold: str
    strict: bool
    finder: str
    year: int

    @property
    def lower_bound_fraction(self) -> Fraction:
        return Fraction(self.lower_bound)

    @property
    def beat_threshold_fraction(self) -> Fraction:
        return Fraction(self.beat_threshold)

    def is_beaten_by(self, area: Fraction) -> bool:
        if self.strict:
            return area > self.lower_bound_fraction
        return area >= self.beat_threshold_fraction


def _record(
    n: int, printed: str, finder: str, year: int, exact: str | None = None
) -> PublishedRecord:
    """Build a record row, deriving the strict-improvement threshold from printed precision."""
    if exact is not None:
        return PublishedRecord(
            n=n,
            printed=printed,
            lower_bound=exact,
            beat_threshold=exact,
            strict=True,
            finder=finder,
            year=year,
        )
    digits = printed.rstrip("+")
    decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
    lower = Fraction(digits)
    threshold = lower + Fraction(1, 10**decimals)
    return PublishedRecord(
        n=n,
        printed=printed,
        lower_bound=decimal_floor(lower, decimals),
        beat_threshold=decimal_floor(threshold, decimals + 1),
        strict=False,
        finder=finder,
        year=year,
    )


def decimal_floor(value: Fraction, digits: int) -> str:
    """Exact decimal string, rounded **down** to ``digits`` places.

    Flooring rather than rounding keeps the string a valid lower bound on ``value`` at every
    sign, so a printed score can be quoted without ever inflating it.
    """
    if digits < 0:
        raise ValueError("digits must be non-negative")
    scaled = value * 10**digits
    whole = scaled.numerator // scaled.denominator
    sign = "-" if whole < 0 else ""
    whole = abs(whole)
    text = str(whole).rjust(digits + 1, "0")
    if digits == 0:
        return f"{sign}{text}"
    return f"{sign}{text[:-digits]}.{text[-digits:]}"


# Transcribed from RECORD_SOURCE.  n = 16 is the exact algebraic value 7/341 given there.
PUBLISHED_RECORDS: dict[int, PublishedRecord] = {
    row.n: row
    for row in (
        _record(3, "1/2 = .5", "trivial", 0, exact="1/2"),
        _record(4, "1/2 = .5", "trivial", 0, exact="1/2"),
        _record(5, ".19245+", "Yang Lu, Zhang Jingzhong, Zeng Zhenbing", 1991),
        _record(6, "1/8 = .125", "A. Dress, L. Yang, Z. B. Zeng", 1995, exact="1/8"),
        _record(7, ".08386+", "F. Comellas, J. Yebra", 2001),
        _record(8, ".07237+", "F. Comellas, J. Yebra", 2001),
        _record(9, ".05487+", "F. Comellas, J. Yebra", 2001),
        _record(10, ".04654+", "F. Comellas, J. Yebra", 2001),
        _record(11, "1/27 = .03704+", "Michael Goldberg", 1972, exact="1/27"),
        _record(12, ".03260+", "F. Comellas, J. Yebra", 2001),
        _record(13, ".02702+", "Peter Karpov", 2011),
        _record(14, ".02430+", "Mark Beyleveld", 2006),
        _record(15, ".02121+", "Nathan Sudermann-Merx", 2026),
        _record(16, "7/341 = .02053+", "Mark Beyleveld", 2006, exact="7/341"),
        _record(17, ".016481+", "Tej Stead", 2026),
        _record(18, ".01459+", "Nathan Sudermann-Merx", 2026),
        _record(19, ".01338+", "William Shanley", 2026),
        _record(20, ".01291+", "William Shanley", 2026),
        _record(21, ".010814+", "Tej Stead", 2026),
        _record(22, ".009569+", "Nathan Sudermann-Merx", 2026),
        _record(23, ".008812+", "Tej Stead", 2026),
        _record(24, ".008495+", "Marc-Emmanuel Coupvent des Graviers", 2026),
        _record(25, ".007330+", "Nathan Sudermann-Merx", 2026),
        _record(26, ".006945+", "Nathan Sudermann-Merx", 2026),
        _record(27, ".006790+", "Marc-Emmanuel Coupvent des Graviers", 2026),
        _record(28, ".006775+", "Marc-Emmanuel Coupvent des Graviers", 2026),
        _record(29, ".005621+", "Nathan Sudermann-Merx", 2026),
        _record(30, ".005445+", "Tej Stead", 2026),
        _record(31, ".005362+", "Nathan Sudermann-Merx", 2026),
        _record(32, ".004690+", "Tej Stead", 2026),
        _record(33, ".004146+", "Tej Stead", 2026),
        _record(34, ".004004+", "Tej Stead", 2026),
        _record(35, ".003702+", "Tej Stead", 2026),
    )
}


# --------------------------------------------------------------------------------------
# Exact verification.  Python integers only; no numpy, no floats, no sampling.
# --------------------------------------------------------------------------------------


def exact_min_double_area(points: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Exhaustive minimum of ``|det|`` over every one of the ``C(n, 3)`` triples.

    Returns ``(minimum, attaining_count)``.  ``minimum`` is ``2 * D^2`` times the smallest
    triangle area for lattice points read as ``(a/D, b/D)``; ``attaining_count`` says how many
    triples sit on that minimum, which is the honest measure of how tight a configuration is.

    Pure Python integer arithmetic throughout, and every triple is visited.
    """
    n = len(points)
    if n < 3:
        raise ValueError("a triangle needs at least three points")
    best = None
    count = 0
    for i, j, k in combinations(range(n), 3):
        ax, ay = points[i]
        bx, by = points[j]
        cx, cy = points[k]
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if det < 0:
            det = -det
        if best is None or det < best:
            best = det
            count = 1
        elif det == best:
            count += 1
    assert best is not None
    return best, count


@dataclass(frozen=True)
class HeilbronnCertificate:
    """A configuration together with everything needed to re-check it by hand."""

    n: int
    denominator: int
    points: tuple[tuple[int, int], ...]
    min_double_area: int
    attaining_triples: int
    triples_checked: int
    area_numerator: int
    area_denominator: int
    area_decimal: str
    record_printed: str
    record_lower_bound: str
    record_beat_threshold: str
    record_strict: bool
    record_finder: str
    record_year: int
    beats_record: bool
    shortfall_ratio_decimal: str

    @property
    def area(self) -> Fraction:
        return Fraction(self.area_numerator, self.area_denominator)

    def to_payload(self) -> dict[str, Any]:
        """Canonical-JSON-safe payload: integers and decimal strings, never floats."""
        return {
            "problem": "heilbronn_unit_square_min_triangle_area",
            "record_source": RECORD_SOURCE,
            "n": self.n,
            "denominator": self.denominator,
            "points": [list(point) for point in self.points],
            "min_double_area": self.min_double_area,
            "attaining_triples": self.attaining_triples,
            "triples_checked": self.triples_checked,
            "area_numerator": self.area_numerator,
            "area_denominator": self.area_denominator,
            "area_decimal": self.area_decimal,
            "record_printed": self.record_printed,
            "record_lower_bound": self.record_lower_bound,
            "record_beat_threshold": self.record_beat_threshold,
            "record_comparison_is_strict": self.record_strict,
            "record_finder": self.record_finder,
            "record_year": self.record_year,
            "beats_record": self.beats_record,
            "achieved_over_record_lower_bound": self.shortfall_ratio_decimal,
            "verification": "exhaustive over all C(n,3) triples, exact integer arithmetic",
            "absence_establishes_novelty": False,
        }

    def digest(self) -> str:
        return canonical_sha256(self.to_payload())


def certify(
    points: Sequence[tuple[int, int]],
    denominator: int,
    record: PublishedRecord | None = None,
) -> HeilbronnCertificate:
    """Verify a lattice configuration exactly and compare it with the published record.

    Raises if the configuration is not a legal instance: points must be distinct lattice
    points inside ``[0, D]^2`` and no three may be collinear (a degenerate triple has area
    zero and would make the score meaningless rather than merely bad).
    """
    n = len(points)
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError("lattice coordinates must be Python ints")
        if not (0 <= x <= denominator and 0 <= y <= denominator):
            raise ValueError("point outside the unit square")
    if len(set(map(tuple, points))) != n:
        raise ValueError("points must be distinct")

    minimum, attaining = exact_min_double_area(points)
    if minimum == 0:
        raise ValueError("configuration has three collinear points")

    area = Fraction(minimum, 2 * denominator * denominator)
    record = record if record is not None else PUBLISHED_RECORDS.get(n)
    if record is None:
        raise KeyError(f"no published record transcribed for n={n}")

    beats = record.is_beaten_by(area)
    ratio = area / record.lower_bound_fraction
    return HeilbronnCertificate(
        n=n,
        denominator=denominator,
        points=tuple((int(x), int(y)) for x, y in points),
        min_double_area=minimum,
        attaining_triples=attaining,
        triples_checked=math.comb(n, 3),
        area_numerator=area.numerator,
        area_denominator=area.denominator,
        area_decimal=decimal_floor(area, 18),
        record_printed=record.printed,
        record_lower_bound=record.lower_bound,
        record_beat_threshold=record.beat_threshold,
        record_strict=record.strict,
        record_finder=record.finder,
        record_year=record.year,
        beats_record=beats,
        shortfall_ratio_decimal=decimal_floor(ratio, 9),
    )


# --------------------------------------------------------------------------------------
# Search.  Heuristic, vectorised, and irrelevant to the certificate above.
# --------------------------------------------------------------------------------------


def _check_lattice_width(denominator: int) -> None:
    """Guard the int64 search path: ``2 * D^2`` must stay inside a signed 64-bit word."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if 2 * denominator * denominator >= 2**63:
        raise ValueError("denominator too large for exact int64 search arithmetic")


class _LatticeSearcher:
    """Batched single-point relocation on the ``{0..D}^2`` lattice.

    All determinants are ``int64`` and exact under :func:`_check_lattice_width`; the soft
    penalty is float only because it ranks candidates and never certifies anything.
    """

    def __init__(self, n: int, denominator: int, rng: random.Random) -> None:
        if n < 3:
            raise ValueError("need at least three points")
        _check_lattice_width(denominator)
        self.n = n
        self.d = denominator
        self.rng = rng
        self.np_rng = np.random.default_rng(rng.randrange(2**63))
        rows, cols = np.triu_indices(n - 1, k=1)
        self._rows = rows
        self._cols = cols
        tri = np.array(list(combinations(range(n), 3)), dtype=np.int64)
        self._tri = (tri[:, 0], tri[:, 1], tri[:, 2])

    # -- evaluation -------------------------------------------------------------------

    def min_det(self, pts: np.ndarray) -> int:
        """Minimum ``|det|`` over all triples, int64 exact.  Used only to steer the search."""
        i, j, k = self._tri
        ax = pts[i, 0]
        ay = pts[i, 1]
        det = (pts[j, 0] - ax) * (pts[k, 1] - ay) - (pts[k, 0] - ax) * (pts[j, 1] - ay)
        return int(np.abs(det).min())

    def _candidate_scores(
        self, pts: np.ndarray, index: int, cands: np.ndarray, target: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Penalty and minimum ``|det|`` restricted to triples through ``index``.

        ``cands`` has shape ``(C, 2)``.  Returns ``(penalty[C], min_det[C])``.  Triples that
        avoid ``index`` are untouched by the move, so they are excluded here and folded back
        in by the caller.
        """
        others = np.delete(pts, index, axis=0)
        vx = others[None, :, 0] - cands[:, 0, None]
        vy = others[None, :, 1] - cands[:, 1, None]
        cross = vx[:, :, None] * vy[:, None, :] - vx[:, None, :] * vy[:, :, None]
        dets = np.abs(cross[:, self._rows, self._cols])
        slack = np.maximum(0.0, 1.0 - dets / float(target))
        return np.einsum("ij,ij->i", slack, slack), dets.min(axis=1)

    # -- proposals --------------------------------------------------------------------

    def _propose(self, pts: np.ndarray, index: int, sigma: float, width: int) -> np.ndarray:
        """Candidate positions for one point: stay, local jitter, boundary snap, or restart.

        Boundary snaps are in the mix because optimal Heilbronn configurations put many
        points exactly on the edges of the square, and a purely Gaussian proposal reaches an
        edge only by luck.
        """
        d = self.d
        cur = pts[index]
        blocks = [cur[None, :]]
        n_local = max(1, width // 2)
        n_fine = max(1, width // 4)
        n_far = max(1, width // 8)
        n_edge = max(1, width // 8)

        for count, scale in ((n_local, sigma), (n_fine, max(1.0, sigma / 12.0))):
            step = self.np_rng.normal(0.0, scale, size=(count, 2))
            blocks.append(cur[None, :] + np.rint(step).astype(np.int64))

        blocks.append(self.np_rng.integers(0, d + 1, size=(n_far, 2)))

        edge = cur[None, :] + np.rint(
            self.np_rng.normal(0.0, max(1.0, sigma), size=(n_edge, 2))
        ).astype(np.int64)
        which = self.np_rng.integers(0, 4, size=n_edge)
        edge[which == 0, 0] = 0
        edge[which == 1, 0] = d
        edge[which == 2, 1] = 0
        edge[which == 3, 1] = d
        blocks.append(edge)

        cands = np.concatenate(blocks, axis=0)
        np.clip(cands, 0, d, out=cands)
        return cands

    # -- optimisation -----------------------------------------------------------------

    def _sweep(self, pts: np.ndarray, target: int, sigma: float, width: int) -> np.ndarray:
        order = list(range(self.n))
        self.rng.shuffle(order)
        for index in order:
            cands = self._propose(pts, index, sigma, width)
            penalty, _ = self._candidate_scores(pts, index, cands, target)
            best = int(np.argmin(penalty))
            if penalty[best] < penalty[0]:
                pts[index] = cands[best]
        return pts


def _distinct(pts: np.ndarray) -> bool:
    return len({(int(x), int(y)) for x, y in pts}) == len(pts)


class _RealRefiner:
    """Sequential linear programming for ``max t  s.t. |det_ijk(P)| >= t, P in [0,1]^2n``.

    Working in the reals here is a search convenience, not a shortcut: the output is rounded
    onto the lattice and re-scored by exact integer arithmetic before anything is claimed.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        tri = np.array(list(combinations(range(n), 3)), dtype=np.int32)
        self.i = tri[:, 0]
        self.j = tri[:, 1]
        self.k = tri[:, 2]
        self.count = tri.shape[0]

    def dets(self, pts: np.ndarray) -> np.ndarray:
        i, j, k = self.i, self.j, self.k
        ax, ay = pts[i, 0], pts[i, 1]
        return (pts[j, 0] - ax) * (pts[k, 1] - ay) - (pts[k, 0] - ax) * (pts[j, 1] - ay)

    def gradients(self, pts: np.ndarray) -> np.ndarray:
        """Rows of ``d det_ijk / d(coords)`` restricted to the six coordinates it depends on."""
        i, j, k = self.i, self.j, self.k
        xi, yi = pts[i, 0], pts[i, 1]
        xj, yj = pts[j, 0], pts[j, 1]
        xk, yk = pts[k, 0], pts[k, 1]
        return np.stack(
            [
                yj - yk,  # d/dx_i
                xk - xj,  # d/dy_i
                yk - yi,  # d/dx_j
                xi - xk,  # d/dy_j
                yi - yj,  # d/dx_k
                xj - xi,  # d/dy_k
            ],
            axis=1,
        )

    def _scatter(self, rows: np.ndarray) -> np.ndarray:
        """Accumulate per-triple, per-slot values onto the ``2n`` coordinate axes."""
        n = self.n
        columns = (self.i, self.i + n, self.j, self.j + n, self.k, self.k + n)
        out = np.zeros(2 * n)
        for slot, col in enumerate(columns):
            out += np.bincount(col, weights=rows[:, slot], minlength=2 * n)
        return out

    def soft_refine(
        self,
        pts: np.ndarray,
        *,
        rounds: int = 9,
        beta0: float = 400.0,
        growth: float = 2.6,
        maxiter: int = 400,
    ) -> tuple[np.ndarray, float]:
        """Ascend the soft minimum ``-(1/b) log sum exp(-b |det|)`` with ``b`` annealed upward.

        The hard minimum is non-smooth exactly where Heilbronn optima live -- many triples
        tight at once -- which is precisely where a subgradient method dithers.  The
        log-sum-exp softening is smooth everywhere, has a cheap exact gradient, and tends to
        the hard minimum as ``b`` grows, so a short L-BFGS-B run per temperature walks into
        the right basin instead of stalling on the first active constraint.
        """
        from scipy.optimize import minimize

        n = self.n
        flat = np.clip(np.asarray(pts, dtype=np.float64), 0.0, 1.0).T.ravel().copy()
        bounds = [(0.0, 1.0)] * (2 * n)
        beta = beta0

        def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
            points = np.stack([z[:n], z[n:]], axis=1)
            det = self.dets(points)
            sign = np.where(det >= 0.0, 1.0, -1.0)
            absdet = np.abs(det)
            floor = absdet.min()
            weights = np.exp(-beta * (absdet - floor))
            total = weights.sum()
            value = floor - math.log(total) / beta
            grad = self._scatter(self.gradients(points) * (sign * weights / total)[:, None])
            return -value, -grad

        for _ in range(rounds):
            result = minimize(
                objective, flat, jac=True, bounds=bounds, method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-14},
            )
            flat = np.clip(result.x, 0.0, 1.0)
            beta *= growth

        points = np.stack([flat[:n], flat[n:]], axis=1)
        return points, float(np.abs(self.dets(points)).min())

    def refine(
        self, pts: np.ndarray, iterations: int = 260, radius: float = 0.05
    ) -> tuple[np.ndarray, float]:
        """Iterate trust-region LP steps; return the best configuration and its min ``|det|``."""
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix

        n = self.n
        pts = np.clip(np.asarray(pts, dtype=np.float64), 0.0, 1.0)
        best = pts.copy()
        best_val = float(np.abs(self.dets(pts)).min())
        cost = np.zeros(2 * n + 1)
        cost[-1] = -1.0

        for _ in range(iterations):
            if radius < 1e-13:
                break
            det = self.dets(pts)
            absdet = np.abs(det)
            current = float(absdet.min())
            # Only triples that the trust region could push below the incumbent matter.
            keep = np.flatnonzero(absdet <= max(current * 6.0, current + 24.0 * radius))
            if keep.size == 0:
                break
            sign = np.where(det[keep] >= 0.0, 1.0, -1.0)
            grad = self.gradients(pts)[keep] * sign[:, None]

            rows = np.repeat(np.arange(keep.size), 7)
            cidx = np.stack(
                [
                    self.i[keep],
                    self.i[keep] + n,
                    self.j[keep],
                    self.j[keep] + n,
                    self.k[keep],
                    self.k[keep] + n,
                    np.full(keep.size, 2 * n, dtype=np.int32),
                ],
                axis=1,
            ).ravel()
            vals = np.concatenate(
                [-grad, np.ones((keep.size, 1))], axis=1
            ).ravel()
            matrix = csr_matrix(
                (vals, (rows, cidx)), shape=(keep.size, 2 * n + 1)
            )
            rhs = absdet[keep]

            flat = np.concatenate([pts[:, 0], pts[:, 1]])
            lower = np.concatenate([np.maximum(-radius, -flat), [0.0]])
            upper = np.concatenate([np.minimum(radius, 1.0 - flat), [np.inf]])
            result = linprog(
                cost,
                A_ub=matrix,
                b_ub=rhs,
                bounds=list(zip(lower, upper, strict=True)),
                method="highs",
            )
            if not result.success:
                radius *= 0.5
                continue

            step = result.x[: 2 * n]
            delta = np.stack([step[:n], step[n:]], axis=1)
            # The LP model is only locally valid, so walk back along its direction rather
            # than discarding the whole step the moment the quadratic truth disagrees.
            chosen, value = None, current
            for fraction in (1.0, 0.55, 0.3, 0.15, 0.07, 0.03):
                moved = np.clip(pts + fraction * delta, 0.0, 1.0)
                trial = float(np.abs(self.dets(moved)).min())
                if trial > value:
                    chosen, value = moved, trial
                    break
            if chosen is None:
                radius *= 0.5
                continue
            pts = chosen
            radius = min(radius * 1.35, 0.3)
            if value > best_val:
                best_val = value
                best = pts.copy()
        return best, best_val


_SQUARE_GROUP: dict[str, tuple[tuple[tuple[int, int, int, int], tuple[int, int]], ...]] = {
    # ((m00, m01, m10, m11), (t0, t1)) acting as  (x, y) -> M (u, v) + t  on the unit square.
    "identity": ((((1, 0), (0, 1)), (0, 0)),),
    "c2": ((((1, 0), (0, 1)), (0, 0)), (((-1, 0), (0, -1)), (1, 1))),
    "c4": (
        (((1, 0), (0, 1)), (0, 0)),
        (((0, -1), (1, 0)), (1, 0)),
        (((-1, 0), (0, -1)), (1, 1)),
        (((0, 1), (-1, 0)), (0, 1)),
    ),
    "d2": (
        (((1, 0), (0, 1)), (0, 0)),
        (((-1, 0), (0, 1)), (1, 0)),
        (((1, 0), (0, -1)), (0, 1)),
        (((-1, 0), (0, -1)), (1, 1)),
    ),
    "d2diag": (
        (((1, 0), (0, 1)), (0, 0)),
        (((0, 1), (1, 0)), (0, 0)),
        (((0, -1), (-1, 0)), (1, 1)),
        (((-1, 0), (0, -1)), (1, 1)),
    ),
    "d4": (
        (((1, 0), (0, 1)), (0, 0)),
        (((0, -1), (1, 0)), (1, 0)),
        (((-1, 0), (0, -1)), (1, 1)),
        (((0, 1), (-1, 0)), (0, 1)),
        (((-1, 0), (0, 1)), (1, 0)),
        (((1, 0), (0, -1)), (0, 1)),
        (((0, 1), (1, 0)), (0, 0)),
        (((0, -1), (-1, 0)), (1, 1)),
    ),
}

# Each orbit type says how a free point is parametrised: ``select`` maps local parameters to
# ``(u, v)`` and ``const`` is the fixed part.  Applying the whole group and discarding repeats
# then yields the orbit, so a point pinned to a mirror line automatically produces a short one.
_ORBIT_TYPES: dict[str, tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, Fraction]]] = {
    "generic": (((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))), (Fraction(0),) * 2),
    "vertical": (((Fraction(0),), (Fraction(1),)), (Fraction(1, 2), Fraction(0))),
    "horizontal": (((Fraction(1),), (Fraction(0),)), (Fraction(0), Fraction(1, 2))),
    "diagonal": (((Fraction(1),), (Fraction(1),)), (Fraction(0), Fraction(0))),
    "antidiagonal": (((Fraction(-1),), (Fraction(1),)), (Fraction(1), Fraction(0))),
    "centre": ((), (Fraction(1, 2), Fraction(1, 2))),
}


@dataclass(frozen=True)
class SymmetryLayout:
    """An affine reparametrisation ``P = matrix @ z + offset`` that forces a symmetry.

    ``matrix`` has shape ``(2n, d)`` and ``offset`` shape ``(2n,)``, with the flat point vector
    ordered ``[x_0..x_{n-1}, y_0..y_{n-1}]``.  Because the map is affine with entries in
    ``{0, +-1}`` and ``z`` is boxed in ``[0, 1]^d``, every generated point is automatically
    inside the square -- no projection, no constraint handling.

    Searching a symmetry class is a genuine restriction: the best symmetric configuration can
    be worse than the best asymmetric one.  It buys a search space of dimension ``d`` instead
    of ``2n``, which for ``d4`` is a factor of eight, and several of the published records in
    this range are themselves symmetric.
    """

    group: str
    n: int
    orbits: tuple[str, ...]
    matrix: np.ndarray
    offset: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1])

    def points(self, z: np.ndarray) -> np.ndarray:
        flat = self.matrix @ np.asarray(z, dtype=float) + self.offset
        return np.stack([flat[: self.n], flat[self.n :]], axis=1)


def _orbit_images(
    group: str, spec: str
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Images of one parametrised free point under the group, with duplicates removed."""
    select_rows, const = _ORBIT_TYPES[spec]
    local = len(select_rows[0]) if select_rows else 0
    select = np.array([[float(v) for v in row] for row in select_rows], dtype=float)
    if local == 0:
        select = np.zeros((2, 0))
    shift = np.array([float(const[0]), float(const[1])])

    images: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple] = set()
    for rows, translate in _SQUARE_GROUP[group]:
        m = np.array(rows, dtype=float)
        a = m @ select
        b = m @ shift + np.array(translate, dtype=float)
        key = (tuple(a.ravel()), tuple(b))
        if key in seen:
            continue
        seen.add(key)
        images.append((a, b))
    return images, local


def orbit_sizes(group: str) -> dict[str, int]:
    """Orbit length of each admissible free-point type under ``group``."""
    sizes = {}
    for spec in _ORBIT_TYPES:
        try:
            images, _ = _orbit_images(group, spec)
        except KeyError:  # pragma: no cover - defensive
            continue
        sizes[spec] = len(images)
    return sizes


def enumerate_layouts(group: str, n: int, limit: int = 64) -> list[tuple[str, ...]]:
    """All orbit compositions of ``n`` points under ``group``, shortest parametrisation first.

    An empty result is a real answer: ``n`` points simply cannot be arranged with that
    symmetry (there is no ``d4``-symmetric 27-point set, for instance, because
    ``8a + 4b + 1`` never equals 27).
    """
    sizes = orbit_sizes(group)
    specs = [
        s
        for s in ("generic", "vertical", "horizontal", "diagonal", "antidiagonal")
        if s in sizes
    ]
    # Under a group containing a diagonal reflection the two axis mirrors are conjugate, as are
    # the two diagonal mirrors, so those orbit families generate the same layouts; collapsing
    # them stops the search from spending its budget re-running identical configurations.
    fused = {"d4": {"horizontal": "vertical", "antidiagonal": "diagonal"},
             "c4": {"horizontal": "vertical", "antidiagonal": "diagonal"}}.get(group, {})
    results: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def record_combo(combo: tuple[str, ...]) -> None:
        key = tuple(sorted(fused.get(spec, spec) for spec in combo))
        if key in seen:
            return
        seen.add(key)
        results.append(combo)

    def walk(remaining: int, index: int, chosen: tuple[str, ...]) -> None:
        if len(results) >= limit:
            return
        if remaining == 0:
            record_combo(chosen)
            return
        if index >= len(specs):
            if remaining == 1 and "centre" not in chosen:
                record_combo(chosen + ("centre",))
            return
        spec = specs[index]
        size = sizes[spec]
        cap = remaining // size
        for count in range(cap, -1, -1):
            walk(remaining - count * size, index + 1, chosen + (spec,) * count)

    walk(n, 0, ())
    results.sort(key=lambda combo: (len(combo), combo))
    return results[:limit]


def build_layout(group: str, orbits: Sequence[str]) -> SymmetryLayout:
    """Assemble the affine map for a chosen orbit composition."""
    xs: list[list[tuple[int, float]]] = []
    ys: list[list[tuple[int, float]]] = []
    offx: list[float] = []
    offy: list[float] = []
    dimension = 0
    for spec in orbits:
        images, local = _orbit_images(group, spec)
        slots = list(range(dimension, dimension + local))
        dimension += local
        for a, b in images:
            xs.append([(slots[c], a[0, c]) for c in range(local) if a[0, c] != 0.0])
            ys.append([(slots[c], a[1, c]) for c in range(local) if a[1, c] != 0.0])
            offx.append(float(b[0]))
            offy.append(float(b[1]))

    n = len(offx)
    matrix = np.zeros((2 * n, dimension))
    for row, entries in enumerate(xs):
        for col, value in entries:
            matrix[row, col] = value
    for row, entries in enumerate(ys):
        for col, value in entries:
            matrix[n + row, col] = value
    offset = np.concatenate([np.array(offx), np.array(offy)])
    return SymmetryLayout(
        group=group, n=n, orbits=tuple(orbits), matrix=matrix, offset=offset
    )


class _SymmetricRefiner:
    """Soft-min ascent carried out in the reduced coordinates of a :class:`SymmetryLayout`."""

    def __init__(self, refiner: _RealRefiner, layout: SymmetryLayout) -> None:
        if refiner.n != layout.n:
            raise ValueError("layout and refiner disagree about n")
        self.refiner = refiner
        self.layout = layout

    def value(self, z: np.ndarray) -> float:
        return float(np.abs(self.refiner.dets(self.layout.points(z))).min())

    def ascend(
        self, z: np.ndarray, *, rounds: int = 9, low: float = 8.0,
        high: float = 4000.0, maxiter: int = 200,
    ) -> tuple[np.ndarray, float]:
        from scipy.optimize import minimize

        layout = self.layout
        refiner = self.refiner
        bounds = [(0.0, 1.0)] * layout.dimension
        z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
        floor = max(self.value(z), 1e-12)
        beta = low / floor
        growth = (high / low) ** (1.0 / max(1, rounds - 1)) if rounds > 1 else 1.0

        def objective(vec: np.ndarray) -> tuple[float, np.ndarray]:
            points = layout.points(vec)
            det = refiner.dets(points)
            sign = np.where(det >= 0.0, 1.0, -1.0)
            absdet = np.abs(det)
            base = absdet.min()
            weights = np.exp(-beta * (absdet - base))
            total = weights.sum()
            grad_p = refiner._scatter(
                refiner.gradients(points) * (sign * weights / total)[:, None]
            )
            return -(base - math.log(total) / beta), -(layout.matrix.T @ grad_p)

        for _ in range(rounds):
            result = minimize(
                objective, z, jac=True, bounds=bounds, method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-14},
            )
            z = np.clip(result.x, 0.0, 1.0)
            beta *= growth
        return z, self.value(z)


def search_symmetric(
    n: int,
    group: str,
    *,
    seconds: float = 30.0,
    seed: int = 0,
    layouts: Sequence[Sequence[str]] | None = None,
) -> tuple[np.ndarray, float] | None:
    """Multistart ascent inside every orbit composition; ``None`` if the symmetry cannot hold.

    Distinct free points can collide onto one another, which collapses the configuration; such
    candidates are rejected outright rather than scored, since a repeated point is not a legal
    configuration at all.
    """
    import time

    combos = list(layouts) if layouts is not None else enumerate_layouts(group, n)
    if not combos:
        return None
    rng = np.random.default_rng(seed)
    refiner = _RealRefiner(n)
    built = [build_layout(group, combo) for combo in combos]
    built = [layout for layout in built if layout.n == n and layout.dimension > 0]
    if not built:
        return None

    deadline = time.monotonic() + seconds
    best_points: np.ndarray | None = None
    best_value = -1.0
    while time.monotonic() < deadline:
        layout = built[int(rng.integers(0, len(built)))]
        symmetric = _SymmetricRefiner(refiner, layout)
        z, value = symmetric.ascend(rng.random(layout.dimension))
        points = layout.points(z)
        if len({(round(x, 12), round(y, 12)) for x, y in points}) != n:
            continue
        if value > best_value:
            best_points, best_value = points, value
    if best_points is None:
        return None
    return best_points, best_value


def _anneal(
    refiner: _RealRefiner,
    pts: np.ndarray,
    rounds: int,
    low: float,
    high: float,
    maxiter: int,
) -> tuple[np.ndarray, float]:
    """Soft-min ascent with the temperature ladder scaled to the incumbent minimum.

    A fixed ``beta`` is wrong at every scale: the softening only tracks the hard minimum when
    ``beta * |det|`` is large, and ``|det|`` shrinks like ``1/n^2``.  Anchoring the ladder to
    the current minimum makes one schedule work for every ``n``.
    """
    floor = max(float(np.abs(refiner.dets(pts)).min()), 1e-12)
    growth = (high / low) ** (1.0 / max(1, rounds - 1)) if rounds > 1 else 1.0
    return refiner.soft_refine(
        pts, rounds=rounds, beta0=low / floor, growth=growth, maxiter=maxiter
    )


def _deep_polish(refiner: _RealRefiner, pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Long anneal followed by sequential-LP sharpening; used only on promising candidates."""
    annealed, _ = _anneal(refiner, pts, 9, 8.0, 4000.0, 220)
    return refiner.refine(annealed, iterations=150, radius=0.005)


def _perturb(
    refiner: _RealRefiner, pts: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One kick of the iterated local search, biased toward the triple that is currently worst.

    Local optima of a maximin problem are pinned by their tight triples, so relocating a
    vertex of the *binding* triangle is the move most likely to reach a different basin; the
    other three modes keep the walk from specialising on that one idea.
    """
    n = refiner.n
    out = pts.copy()
    worst = int(np.argmin(np.abs(refiner.dets(out))))
    triple = np.array([refiner.i[worst], refiner.j[worst], refiner.k[worst]])
    mode = int(rng.integers(0, 4))
    if mode == 0:
        out[rng.choice(triple)] = rng.random(2)
    elif mode == 1:
        count = int(rng.integers(1, max(2, n // 5)))
        out[rng.choice(n, size=count, replace=False)] = rng.random((count, 2))
    elif mode == 2:
        spread = float(rng.choice(np.array([0.008, 0.03, 0.08])))
        out = np.clip(out + rng.normal(0.0, spread, (n, 2)), 0.0, 1.0)
    else:
        a, b = rng.choice(n, size=2, replace=False)
        out[[a, b]] = np.clip(out[[b, a]] + rng.normal(0.0, 0.02, (2, 2)), 0.0, 1.0)
    return out


def search_real_configuration(
    n: int,
    *,
    seconds: float = 60.0,
    seed: int = 0,
    starts: Iterable[Sequence[Sequence[float]]] = (),
    on_improve: Any = None,
    restart_after: int = 900,
) -> tuple[np.ndarray, float]:
    """Iterated local search in the reals; returns ``(points, min |det|)``.

    The walk is: kick the incumbent (:func:`_perturb`), re-converge cheaply, promote anything
    within a hair of the record-so-far to a deep polish, and restart from scratch after a long
    barren stretch.  Nothing here is trusted -- the output is a *proposal* that the lattice
    stage and then :func:`certify` have to survive.
    """
    import time

    rng = np.random.default_rng(seed)
    refiner = _RealRefiner(n)
    deadline = time.monotonic() + seconds

    queue = [np.asarray(item, dtype=float) for item in starts]
    current, current_value = _deep_polish(
        refiner, queue.pop() if queue else rng.random((n, 2))
    )
    best, best_value = current.copy(), current_value
    barren = 0

    while time.monotonic() < deadline:
        if queue:
            candidate, value = _deep_polish(refiner, np.clip(queue.pop(), 0.0, 1.0))
        else:
            candidate, value = _anneal(
                refiner, _perturb(refiner, current, rng), 3, 25.0, 3000.0, 70
            )
            if value > best_value * 0.999:
                candidate, value = _deep_polish(refiner, candidate)
        barren += 1

        if value > current_value or rng.random() < 0.03:
            current, current_value = candidate, value
        if value > best_value:
            best, best_value = candidate.copy(), value
            current, current_value = candidate.copy(), value
            barren = 0
            if on_improve is not None:
                on_improve(best, best_value)
        if barren > restart_after:
            current, current_value = _deep_polish(refiner, rng.random((n, 2)))
            barren = 0

    return best, best_value


def lattice_polish(
    real_points: np.ndarray,
    denominator: int,
    *,
    seed: int = 0,
    width: int = 96,
    rounds: int = 40,
) -> tuple[list[tuple[int, int]], int]:
    """Round a real configuration onto ``{0..D}^2`` and climb the exact integer ladder.

    Rounding costs a little of the real objective; the ladder wins it back and often more,
    because the lattice search is free to move points the continuous optimiser had pinned.
    """
    rng = random.Random(seed)
    n = int(real_points.shape[0])
    searcher = _LatticeSearcher(n, denominator, rng)
    pts = np.clip(np.rint(np.asarray(real_points, float) * denominator), 0, denominator)
    pts = pts.astype(np.int64)
    if not _distinct(pts):
        raise ValueError("rounding collapsed two points; use a finer denominator")

    current = searcher.min_det(pts)
    best_pts, best_min = pts.copy(), current
    step = 1e-5
    sigma = 8.0
    stall = 0
    for _ in range(rounds):
        target = max(current + 1, int(current * (1.0 + step)) + 1)
        for _ in range(3):
            searcher._sweep(pts, target, sigma, width)
        achieved = searcher.min_det(pts)
        if achieved > current and _distinct(pts):
            current, stall = achieved, 0
            step = min(step * 1.5, 3e-3)
            if current > best_min:
                best_min, best_pts = current, pts.copy()
        else:
            step *= 0.55
            sigma = max(1.0, sigma * 0.8)
            stall += 1
            if stall >= 6:
                break
    return [(int(x), int(y)) for x, y in best_pts], best_min


def search_configuration(
    n: int,
    denominator: int = 1 << 22,
    *,
    seconds: float = 60.0,
    seed: int = 0,
    width: int = 96,
    seeds: Iterable[Sequence[tuple[int, int]]] = (),
    progress: bool = False,
    checkpoint: Path | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """Search in the reals, then land on the lattice; returns ``(points, min_double_area)``.

    The returned minimum is re-derived by :func:`exact_min_double_area` inside :func:`certify`
    before anything is claimed; this function's bookkeeping only steers the search.
    """
    scale = float(denominator)
    starts = [np.asarray(item, dtype=float) / scale for item in seeds]
    state: dict[str, Any] = {"points": None, "min": -1}

    def promote(real: np.ndarray, _value: float) -> None:
        try:
            points, minimum = lattice_polish(real, denominator, seed=seed, width=width)
        except ValueError:
            return
        if minimum <= state["min"]:
            return
        state["points"], state["min"] = points, minimum
        if progress:
            area = Fraction(minimum, 2 * denominator * denominator)
            print(f"  n={n} seed={seed} A={decimal_floor(area, 10)}", flush=True)
        if checkpoint is not None:
            certificate = certify(points, denominator)
            payload = certificate.to_payload()
            payload["digest"] = certificate.digest()
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    real, _ = search_real_configuration(
        n, seconds=seconds, seed=seed, starts=starts, on_improve=promote
    )
    if state["points"] is None:
        promote(real, 0.0)
    if state["points"] is None:
        raise RuntimeError("search produced no configuration")
    return state["points"], state["min"]


def _load_seeds(path: Path | None) -> list[list[tuple[int, int]]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [[(int(x), int(y)) for x, y in cfg] for cfg in payload]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--denominator", type=int, default=1 << 20)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--seeds", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    points, _ = search_configuration(
        args.n,
        args.denominator,
        seconds=args.seconds,
        seed=args.seed,
        width=args.width,
        seeds=_load_seeds(args.seeds),
        progress=args.progress,
        checkpoint=args.out,
    )
    certificate = certify(points, args.denominator)
    payload = certificate.to_payload()
    payload["digest"] = certificate.digest()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
