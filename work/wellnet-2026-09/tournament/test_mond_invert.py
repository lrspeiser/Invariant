"""Regression test for the deep-MOND k-scaling of `tw_core.mond_invert`.

Run AQ.  The `rar` branch carried p = 3/2 where its own docstring specified
the AQUAL-matching value; the discrepancy is exactly zero at k = 1 and in
both Newtonian limits, so nothing in the tournament's own gates caught it.
The test below is the one that would have.

    python test_mond_invert.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tw_core import A0, mond_invert, nu_rar          # noqa: E402

KS = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
FAIL = []


def _slope(base, F):
    """d ln g / d ln k for a single flux F, over KS."""
    g = np.array([float(mond_invert(np.array([F]), np.array([k]), A0, base)[0])
                  for k in KS])
    assert np.all(g > 0), f"{base}: non-positive g at F={F:g}"
    return float(np.polyfit(np.log(KS), np.log(g), 1)[0])


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<44s} {got:+.4f}  (want {want:+.2f} +- {tol})")
    if not ok:
        FAIL.append(name)


print("deep-MOND k-scaling, F = 1e-4 a0   [d ln g / d ln k]")
F_deep = 1e-4 * A0
s_aqual = _slope("aqual", F_deep)
check("aqual", s_aqual, -0.75, 0.01)
check("rar", _slope("rar", F_deep), -0.75, 0.01)
check("newton", _slope("newton", F_deep), -1.00, 1e-9)

# The binding requirement: the two MOND bases must agree with EACH OTHER,
# whatever the shared value turns out to be.
d = abs(_slope("rar", F_deep) - s_aqual)
print(f"\n  {'PASS' if d <= 0.01 else 'FAIL'}  |rar - aqual| deep-MOND slope        "
      f"{d:.2e}  (want <= 1e-2)")
if d > 0.01:
    FAIL.append("rar/aqual deep-MOND slopes disagree")

print("\nNewtonian k-scaling, F = 1e4 a0    [all bases -> -1]")
F_newt = 1e4 * A0
for b in ("aqual", "rar", "newton"):
    check(b, _slope(b, F_newt), -1.00, 0.01)

print("\nk = 1 identity: rar must reduce to plain nu_RAR exactly")
one = np.array([1.0])
worst = 0.0
for x in (1e-6, 1e-3, 1e-1, 1.0, 1e1, 1e3, 1e6):
    Fv = np.array([x * A0])
    got = float(mond_invert(Fv, one, A0, "rar")[0])
    want = float((nu_rar(Fv / A0) * Fv)[0])
    worst = max(worst, abs(got - want) / want)
print(f"  {'PASS' if worst <= 1e-14 else 'FAIL'}  worst relative deviation over 7 decades"
      f"      {worst:.2e}  (want <= 1e-14)")
if worst > 1e-14:
    FAIL.append("k=1 rar identity")

print("\nmonotonicity: g must fall as the conductivity k rises")
for b in ("aqual", "rar", "newton"):
    for x in (1e-4, 1.0, 1e4):
        g = np.array([float(mond_invert(np.array([x * A0]), np.array([k]),
                                        A0, b)[0]) for k in KS])
        if not np.all(np.diff(g) < 0):
            FAIL.append(f"{b} non-monotone in k at F={x:g} a0")
print(f"  {'PASS' if not any('non-monotone' in f for f in FAIL) else 'FAIL'}"
      f"  g strictly decreasing in k, all bases, 3 regimes")

print()
if FAIL:
    print(f"{len(FAIL)} FAILURE(S): " + "; ".join(FAIL))
    raise SystemExit(1)
print("all checks passed")
