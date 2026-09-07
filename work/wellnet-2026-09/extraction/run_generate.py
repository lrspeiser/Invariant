"""run_generate.py -- every paired pool this lane analyses, generated once.

    G1_main      BF's shared scene library, 17 arms (the CDM prior, the class at
                 threshold and at fiducial, the scalar null, Newton, systematics)
    G1_heldout   a scene library BF never drew (different seed): the untouched
                 scenes for out-of-sample validation
    G1_scans     CDM with one nuisance moved at a time (f_lss, halo scatter,
                 dark disc, oblateness, systematics, noise), plus the
                 instrument-nuisance arms for C7
    G1_cf        the causal counterfactuals: paired scenes, one thing changed
    G2_main      generator 2 (forward2.py): the class, CDM, Newton, a tensor arm,
                 and the CDM nuisance scans

Each stage is skipped if its file exists, so the run is resumable.  The
provenance ledger is installed in the parent and in every worker.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import guard                     # noqa: E402
import arms as AR                # noqa: E402
import forward2 as F2            # noqa: E402
import gen1 as W                 # noqa: E402

RES = os.environ.get("EXTRACTION_RES", os.path.join(HERE, "results"))   # override for a scratch run
os.makedirs(RES, exist_ok=True)
N_MAIN = int(os.environ.get("N_MAIN", 1000))
N_HELD = int(os.environ.get("N_HELD", 400))
N_SCAN = int(os.environ.get("N_SCAN", 300))
N_CF = int(os.environ.get("N_CF", 300))
N_G2 = int(os.environ.get("N_G2", 600))


def stage(name, seeds, arms, opt, runner, stacker):
    path = os.path.join(RES, f"{name}.npz")
    if os.path.exists(path):
        print(f"[{name}] exists, skipping", flush=True)
        return
    t0 = time.time()
    res = runner(seeds, arms, opt)
    st = stacker(res, arms)
    W.save_stack(path, st)
    print(f"[{name}] {len(seeds)} sets x {len(arms)} arms in {time.time() - t0:.0f}s -> {path}", flush=True)


def main():
    guard.start()
    t00 = time.time()
    # libraries built in the parent so workers only load
    W.get_lib("train")
    W.get_lib("heldout")
    W.get_lib("scrambled")
    print(f"libraries ready ({time.time() - t00:.0f}s)", flush=True)

    stage("G1_main", 1000 + np.arange(N_MAIN), AR.main_arms(), {"lib": "train"},
          lambda s, a, o: W.run_sets(s, a, o), W.stack)
    stage("G1_heldout", 5000 + np.arange(N_HELD), AR.heldout_arms(), {"lib": "heldout"},
          lambda s, a, o: W.run_sets(s, a, o), W.stack)
    stage("G1_scans", 9000 + np.arange(N_SCAN), AR.scan_arms(), {"lib": "train"},
          lambda s, a, o: W.run_sets(s, a, o), W.stack)
    stage("G1_cf", 13000 + np.arange(N_CF), AR.counterfactual_arms(), {"lib": "train"},
          lambda s, a, o: W.run_sets(s, a, o), W.stack)
    stage("G1_scans_extra", 17000 + np.arange(N_SCAN), AR.scan_arms_extra(), {"lib": "train"},
          lambda s, a, o: W.run_sets(s, a, o), W.stack)
    W.close_pool()
    arms2 = F2.arms2()
    stage("G2_main", 20000 + np.arange(N_G2), arms2, {},
          lambda s, a, o: F2.run_sets2(s, a, o), W.stack)
    F2.close_pool2()
    prov = guard.stop()
    prov["elapsed_s"] = time.time() - t00
    prov["n"] = dict(main=N_MAIN, heldout=N_HELD, scans=N_SCAN, cf=N_CF, g2=N_G2)
    with open(os.path.join(RES, "provenance_generate.json"), "w") as f:
        json.dump(prov, f, indent=1, default=str)
    print(f"[done] {time.time() - t00:.0f}s; foreign reads: {prov['foreign_reads']}", flush=True)


if __name__ == "__main__":
    main()
