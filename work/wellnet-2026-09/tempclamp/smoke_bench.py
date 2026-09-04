"""
End-to-end smoke test of the PATCHED Bench, with the sealed probes stubbed out.

`Bench.__init__` calls `_kids()` and `_widebin()`.  In this checkout KiDS has no
data file and fails harmlessly, but `_widebin` returns HARD-CODED El-Badry
boosts from the source itself, so a bare `Bench()` loads a sealed probe.  That
is true of the pre-patch bench too -- it is a standing hazard in shared bench
code, not something this patch introduces -- and it is why nothing in this lane
constructs a Bench without stubbing them first.
"""
from __future__ import annotations

import sys
import warnings

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
sys.path.insert(0, ROOT + "work/gravity-wells-2026-09")

import numpy as np                                                # noqa: E402
import invariant_bench as IB                                      # noqa: E402
from invariant_bench import Bench                                 # noqa: E402

# SEAL: neutralise both holdouts before any Bench is constructed
IB.Bench._kids = lambda self: None
IB.Bench._widebin = lambda self: None


def main():
    for mode in ("clamp", "drop", "loglinear"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b = Bench(verbose=False, temp_extrapolation=mode)
        assert "kids" not in b.d and "widebin" not in b.d, "SEAL VIOLATION"
        xc = b.d["xcop"]
        wx = [x for x in w
              if issubclass(x.category, IB.TemperatureExtrapolationWarning)]
        s = b.extrapolation_summary
        print(f"   mode={mode:<10} probes={sorted(b.d)}  n_xcop={len(xc)}  "
              f"affected={s['n_affected']} (kept {s['n_extrapolated']}, dropped {s['n_dropped']}) "
              f"({100*s['frac_extrapolated']:.2f}%)  warnings={len(wx)}")
        assert len(wx) == 1, "the extrapolation warning must fire exactly once"
        # the scoring interface still works, and now reports the exposure
        out = b.score(lambda d: 1.0 / (1 - np.exp(-np.sqrt(d.x))), verbose=False)
        print(f"   {'':<15} baseline median |err| dex: "
              + "  ".join(f"{k}={v:.4f}" for k, v in sorted(out.items())))
    # a caller that refuses to extrapolate now gets a hard stop
    try:
        Bench(verbose=False, temp_extrapolation="forbid")
        print("   forbid: NO ERROR -- wiring broken")
        return 1
    except IB.TemperatureExtrapolationError as e:
        print(f"   forbid: raises as designed -- {str(e)[:110]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
