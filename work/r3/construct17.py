"""Randomised constructive search for small-diameter 17-element weak Sidon sets.

Upper bounds only.  Every hit is re-checked with the naive O(n^2) definition
before being reported, so a reported bound is exactly verified even though the
search itself is heuristic.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, r"C:\Users\henry\Documents\Codex\2026-08-06\for\wt-exh\src")

from sigma_theory_compiler.weak_sidon_exhaustion import (  # noqa: E402
    is_weak_sidon,
    witness_diameter,
)


@njit(cache=True, nogil=True)
def attempt(n, L, seed, rounds):
    """Randomised greedy with restarts; returns a set of size n in [0,L] or empty."""
    np.random.seed(seed)
    chosen = np.zeros(n, np.int64)
    used = np.zeros(2 * L + 4, np.uint8)
    order = np.arange(L + 1)
    for _ in range(rounds):
        for i in range(2 * L + 4):
            used[i] = 0
        for i in range(L + 1):
            order[i] = i
        for i in range(L, 0, -1):
            j = np.random.randint(0, i + 1)
            t = order[i]
            order[i] = order[j]
            order[j] = t
        # 0 and L are forced (a diameter-L set contains both endpoints)
        k = 0
        chosen[0] = 0
        k = 1
        ok = True
        for j in range(k):
            s = chosen[j] + L
            if used[s] != 0:
                ok = False
            used[s] = 1
        if not ok:
            continue
        chosen[1] = L
        k = 2
        for idx in range(L + 1):
            x = order[idx]
            if x == 0 or x == L:
                continue
            if k == n:
                break
            good = True
            j = 0
            while j < k:
                s = chosen[j] + x
                if used[s] != 0:
                    good = False
                    break
                used[s] = 1
                j += 1
            if good:
                chosen[k] = x
                k += 1
            else:
                i = 0
                while i < j:
                    used[chosen[i] + x] = 0
                    i += 1
        if k == n:
            return chosen.copy()
    return np.zeros(0, np.int64)


if __name__ == "__main__":
    n = 17
    best = 172
    t0 = time.time()
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    seed = 1
    while time.time() - t0 < budget:
        L = best - 1
        got = attempt(n, L, seed, 4000)
        seed += 1
        if got.size:
            wit = tuple(int(v) for v in sorted(got))
            assert is_weak_sidon(wit), wit
            d = witness_diameter(wit)
            if d < best:
                best = d
                print("A345731(17) <= %d   %s" % (d, list(wit)))
                sys.stdout.flush()
    print("best upper bound found: %d  (%.0fs)" % (best, time.time() - t0))
