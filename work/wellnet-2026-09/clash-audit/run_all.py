"""
Regenerate the whole lane in order.  `python run_all.py`

  tests.py            18 self-checks, must pass before anything is believed
  run_provenance.py   Job 2.1 + Job 3   -> provenance_results.json
  run_cancellation.py Job 2.2           -> cancellation_results.json
  run_structure.py    Jobs 2.4/2.5/2.6  -> structure_results.json
  run_diagnostics.py  POWER + robustness-> diagnostics_results.json
  run_null.py         Job 2.3           -> null_results.json      (~4 min)
  run_sensitivity.py  null systematics  -> sensitivity_results.json (~2 min)
  run_truthcheck.py   which nulls the data allows -> truthcheck_results.json
  report.py           REPORT.md + results.json
"""
from __future__ import annotations
import subprocess
import sys
import time

STEPS = ["tests.py", "run_provenance.py", "run_cancellation.py",
         "run_structure.py", "run_diagnostics.py", "run_null.py",
         "run_sensitivity.py", "run_truthcheck.py", "report.py"]

if __name__ == "__main__":
    for s in STEPS:
        t = time.time()
        print(f"\n{'='*72}\n{s}\n{'='*72}")
        r = subprocess.run([sys.executable, s])
        if r.returncode != 0:
            print(f"FAILED at {s} (exit {r.returncode})")
            raise SystemExit(r.returncode)
        print(f"-- {s} ok, {time.time()-t:.0f}s")
    print("\nall steps complete")
