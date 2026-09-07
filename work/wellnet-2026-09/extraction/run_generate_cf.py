"""run_generate_cf.py -- regenerate the counterfactual pool (after the external-axis
rebuild fix in paired.geom_with_scene) and the extra nuisance scans, with a
smaller worker pool so a concurrent generator-2 stage keeps its CPUs."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import guard                     # noqa: E402
import arms as AR                # noqa: E402
import gen1 as W                 # noqa: E402
from run_generate import stage, N_SCAN, N_CF   # noqa: E402

if __name__ == "__main__":
    guard.start()
    t0 = time.time()
    W.get_lib("train")
    W.get_lib("scrambled")
    W.get_pool(nproc=int(os.environ.get("NPROC", 8)), libs=("train",))
    stage("G1_cf", 13000 + np.arange(N_CF), AR.counterfactual_arms(), {"lib": "train"},
          lambda s, a, o: W.run_sets(s, a, o), W.stack)
    stage("G1_scans_extra", 17000 + np.arange(N_SCAN), AR.scan_arms_extra(), {"lib": "train"},
          lambda s, a, o: W.run_sets(s, a, o), W.stack)
    W.close_pool()
    print(f"[done] {time.time() - t0:.0f}s; foreign reads: {guard.stop()['foreign_reads']}", flush=True)
