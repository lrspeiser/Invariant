"""Driver: rebuild the A345731 ladder from scratch and check it against OEIS."""
from __future__ import annotations

import sys
import time

sys.path.insert(0, r"C:\Users\henry\Documents\Codex\2026-08-06\for\wt-exh\src")

from sigma_theory_compiler.weak_sidon_exhaustion import (  # noqa: E402
    A345731_PUBLISHED,
    is_weak_sidon,
    min_diameter_lower_bound,
    search_within,
    witness_diameter,
)


def ladder(nmax: int, engine: str = "auto", verbose: bool = True) -> list[int]:
    w = [0, 0]
    for n in range(2, nmax + 1):
        lo = max(w[n - 1], min_diameter_lower_bound(n))
        L = lo
        w.append(L)
        total = 0
        t0 = time.time()
        while True:
            w[n] = L
            found, nodes, wit = search_within(n, L, w, engine=engine)
            total += abs(nodes)
            if found:
                assert is_weak_sidon(wit), wit
                assert witness_diameter(wit) == L, (wit, L)
                w[n] = L
                break
            L += 1
        if verbose:
            pub = A345731_PUBLISHED.get(n)
            tag = "OK" if pub == L else ("*** exp %s" % pub)
            print(
                "A345731(%2d) = %4d  %-10s nodes=%14d  %8.2fs  %s"
                % (n, L, tag, total, time.time() - t0, list(wit))
            )
            sys.stdout.flush()
    return w


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    eng = sys.argv[2] if len(sys.argv) > 2 else "auto"
    ladder(nmax, engine=eng)
