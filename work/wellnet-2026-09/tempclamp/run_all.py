"""
Reproduce the whole lane, in the only order that works.

    python run_all.py

`run_audit.py` writes results.json from scratch; everything after it ADDS to
that file, so the order below is load-bearing.  `report.py` renders REPORT.md
from the finished JSON and types no number of its own.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("test_tempclamp.py", [sys.executable, "-m", "pytest", "test_tempclamp.py", "-q"]),
    ("prove_test_fails_prepatch.py", [sys.executable, "prove_test_fails_prepatch.py"]),
    ("smoke_bench.py", [sys.executable, "smoke_bench.py"]),
    ("run_audit.py  (writes results.json)", [sys.executable, "run_audit.py"]),
    ("at_null_split.py", [sys.executable, "at_null_split.py"]),
    ("verdicts.py", [sys.executable, "verdicts.py"]),
    ("inventory.py", [sys.executable, "inventory.py"]),
    ("report.py  (renders REPORT.md)", [sys.executable, "report.py"]),
]


def main():
    for name, cmd in STEPS:
        print(f"\n{'='*78}\n>>> {name}\n{'='*78}", flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"\n!! {name} FAILED with exit code {r.returncode}")
            return r.returncode
    print("\nall steps green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
