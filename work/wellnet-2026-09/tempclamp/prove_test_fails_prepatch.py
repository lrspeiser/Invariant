"""
Proof that the regression test FAILS against the pre-patch bench.

Loads the untouched pre-patch source (`invariant_bench.py.orig`, byte-for-byte
the file this lane replaced) as a separate module and runs the same three
demands the regression suite makes.  All three fail, in the way the bug
predicts: there is no way to refuse extrapolation, no mask, and no warning.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "invariant_bench.py.orig")
ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
XR = (ROOT + "runs/gravity/roadmap/"
      "item-59-xcop-forward-observable-gate-v1-source/raw/")


def load_prepatch():
    # .orig is not an importable suffix, so name the loader explicitly
    loader = importlib.machinery.SourceFileLoader("invariant_bench_prepatch", ORIG)
    spec = importlib.util.spec_from_file_location(
        "invariant_bench_prepatch", ORIG, loader=loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules["invariant_bench_prepatch"] = m
    spec.loader.exec_module(m)
    return m


def main():
    old = load_prepatch()
    b = old.Bench.__new__(old.Bench)      # never __init__: KiDS/widebin sealed
    d = os.path.join(XR, sorted(os.listdir(XR))[0])
    out = {"prepatch_sha256":
           hashlib.sha256(open(ORIG, "rb").read()).hexdigest(),
           "checks": []}

    def rec(name, failed, detail):
        out["checks"].append(dict(check=name, fails_prepatch=failed, detail=detail))
        print(f"   {'FAILS' if failed else 'passes':>7}  {name}\n            {detail}")

    # 1. can the caller refuse silent extrapolation?
    try:
        b._cluster_profile(d, temp_extrapolation="forbid")
        rec("forbid raises TemperatureExtrapolationError", True,
            "returned normally: pre-patch code silently extrapolates")
    except TypeError as e:
        rec("forbid raises TemperatureExtrapolationError", True,
            f"TypeError instead: {e}")
    except Exception as e:
        rec("forbid raises TemperatureExtrapolationError",
            type(e).__name__ != "TemperatureExtrapolationError",
            f"{type(e).__name__}: {e}")

    # 2. is the extrapolation visible in the output at all?
    p = b._cluster_profile(d)
    has_mask = hasattr(p, "extrapolated")
    rec("profile carries an .extrapolated mask", not has_mask,
        f"type={type(p).__name__}, len={len(p)}, "
        f"attrs={'present' if has_mask else 'ABSENT -- the output cannot tell you'}")

    # 3. does anything warn?
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b._cluster_profile(d)
    # BUG FOUND BY THIS SCRIPT ON ITS FIRST RUN: counting warnings is not the
    # same as counting the RIGHT warning.  astropy emits 2 FITS warnings here,
    # so `len(w) == 0` reported a pass for a module that says nothing at all
    # about extrapolation.  Match on content instead.
    hits = [str(x.message) for x in w
            if "extrapolat" in str(x.message).lower()]
    rec("a warning carries the extrapolated fraction", not hits,
        f"{len(w)} warnings emitted, {len(hits)} of them about extrapolation "
        f"({[type(x.message).__name__ for x in w]})")

    # 4. do the symbols the test imports even exist?
    for sym in ("TemperatureExtrapolationError", "TemperatureExtrapolationWarning",
                "ClusterProfile", "TEMP_MODES"):
        rec(f"module exports {sym}", not hasattr(old, sym),
            "absent" if not hasattr(old, sym) else "present")

    out["n_failing_checks"] = sum(c["fails_prepatch"] for c in out["checks"])
    out["n_checks"] = len(out["checks"])
    print(f"\n   {out['n_failing_checks']}/{out['n_checks']} regression demands "
          f"fail against the pre-patch code.")
    json.dump(out, open(os.path.join(HERE, "prepatch_proof.json"), "w",
                        encoding="utf-8"), indent=1)
    return out


if __name__ == "__main__":
    main()
