"""run_all.py -- the bridge lane (Run BM), in order.

    1. register the run (reads registry.json: the ONE read outside the lane,
       made BEFORE the provenance guard is armed);
    2. build_cache      -> results/cache_*.npz      (column-product grids)
    3. crosscheck       -> results/crosscheck.json  (direct pair sum vs closed form)
    4. members          -> results/members.json     (member confinement vs rho_*)
    5. run_bridge       -> results/bridge.json      (Job 2)
    6. detect           -> results/detect.json      (Job 4: noise, window)
    7. cdm_attack       -> results/cdm_attack.json  (Job 3)
    8. baseline --compare -> results/baseline_compare.json (Job 1 discipline;
       the --snapshot must have been taken on the UNPATCHED compiler first)
    9. compile_bridge   -> results/compile_bridge.json (Job 5)
   10. certify          -> results/certificate_bridge.json (Job 4)
   11. test_bridge      -> results/tests.json
   12. write_report     -> REPORT.md

Every module arms the guard itself; every result JSON carries its provenance.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "registry"))

STEPS = [
    ["build_cache.py"], ["crosscheck.py"], ["members.py"], ["run_bridge.py"],
    ["detect.py"], ["cdm_attack.py"], ["baseline.py", "--compare"],
    ["compile_bridge.py"], ["certify.py"], ["test_bridge.py"], ["write_report.py"],
]

if __name__ == "__main__":
    import registry as R
    R.register(
        "BM-bridge", "bridge",
        depends_on=["identifiability_gate", "holdout_seal", "confirmation_status"],
        outputs=["work/wellnet-2026-09/bridge/results/*.json",
                 "work/wellnet-2026-09/bridge/REPORT.md",
                 "work/wellnet-2026-09/compiler/compiler.py (additive: two-body probe)"],
        note=("Bridge lane (after Run BL): the two-concentration probe the compiler "
              "bench lacked; the path family's compensated-bridge observables at "
              "detector level against Newton, a QUMOND/RAR scalar, BL's tensor action "
              "and a positive-mass CDM filament; the positivity attack; Stage 4 "
              "certificate; member-safe eps window; P re-compiled. Entirely synthetic; "
              "opens NO observational data (asserted mechanically); KiDS and the wide "
              "binaries sealed; the confirmation reserve untouched."))
    for step in STEPS:
        print(f"\n=== {' '.join(step)} ===", flush=True)
        r = subprocess.run([sys.executable] + step, cwd=HERE)
        if r.returncode != 0 and step[0] != "test_bridge.py":
            raise SystemExit(f"{step[0]} failed with {r.returncode}")
