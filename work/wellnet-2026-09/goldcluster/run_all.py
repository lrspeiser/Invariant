"""run_all.py -- the gold-cluster lane, in order.

    1. register the run (reads registry.json: the ONE read outside the lane,
       made BEFORE the guard is armed)
    2. test_lane      -> tests.json          tests before results
    3. acquire        -> raw/erass1_*.       the candidate pool, validated v3
    4. probes         -> probes.json         every route to the six channels
    5. coverage       -> raw/decade_coverage.jsonl   per-cluster, resumable
    6. rank           -> ranking.json        + raw/spec_coverage.jsonl
    7. write_spec     -> SPEC.md, SPEC.json
    8. write_report   -> REPORT.md, spent.json, provenance_ledger.json

Step 5 is the long one: one archive query per candidate, about 15 a minute, and
it is resumable, so an interrupted run continues where it stopped.

    python run_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "registry"))

STEPS = [
    ["register.py"], ["test_lane.py"], ["acquire.py"], ["probes.py"],
    ["coverage.py"], ["rank.py"], ["write_spec.py"], ["write_report.py"],
]


def main():
    for step in STEPS:
        print("\n=== %s ===" % step[0], flush=True)
        r = subprocess.run([sys.executable] + step, cwd=HERE)
        if r.returncode != 0:
            print("FAILED at %s (exit %d)" % (step[0], r.returncode))
            return r.returncode
    print("\nlane complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
