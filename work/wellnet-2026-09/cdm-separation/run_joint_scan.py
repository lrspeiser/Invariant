"""run_joint_scan.py -- the joint procedure across the alignment grid.

`run_forward.py` reports the CDM discriminator's power and the new-gravity
detector's false-positive rate separately.  The procedure that actually beats
Run BF's 0.648 is the JOINT one -- declare new gravity only if the
external-axis statistic fires AND the baryon-axis statistic does not -- so its
false-positive rate must be measured across the same grid.  That rate is the
answer to "at what point does the answer change".

Both clauses degrade together as the halo takes its alignment from the external
axis instead of from the baryons: the detector starts firing AND the veto stops
vetoing, so the joint rate rises faster than either curve alone.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import fwd_worker as FW                                     # noqa: E402
import guard                                                # noqa: E402
from universes.stats import rate_with_ci                    # noqa: E402

RES = os.path.join(HERE, "results")
N = int(os.environ.get("N_J", 500))
BASE_E, BASE_MIS = 0.45, 22.0


def col(pool, k):
    return np.array([r.get(k, 0.0) for r in pool], float)


def main():
    guard.start()
    t0 = time.time()
    F = json.load(io.open(os.path.join(RES, "F_forward.json"), encoding="utf-8"))
    c_ext = F["F2_sizing"]["new_gravity_null"]["S_ext"]["crit"]
    c_bar = F["F2_sizing"]["cdm_null"]["S_bar"]["crit"]

    def joint(pool):
        se, sb = col(pool, "S_ext"), col(pool, "S_bar")
        fire = np.abs(se) >= c_ext["two"]
        veto = sb >= c_bar["up"]
        return dict(fires=rate_with_ci(int(fire.sum()), len(se)),
                    veto=rate_with_ci(int(veto.sum()), len(se)),
                    joint=rate_with_ci(int(np.sum(fire & ~veto)), len(se)))

    grid = []
    for fl in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0):
        grid.append(dict(kind="halo", e_halo=BASE_E, mis_deg=BASE_MIS, f_lss=fl))
    for mis in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        grid.append(dict(kind="halo", e_halo=BASE_E, mis_deg=mis, f_lss=0.0))
    for e in (0.15, 0.30, 0.60, 0.80):
        grid.append(dict(kind="halo", e_halo=e, mis_deg=BASE_MIS, f_lss=0.0))
    # the reference points: no quadrupole at all, and a real tensor
    ref = {"none": dict(kind="none"),
           "tensor_A0.25": dict(kind="tensor", A_tensor=0.25),
           "tensor_A0.5": dict(kind="tensor", A_tensor=0.5)}

    out = {"n": N, "critical_values": {"S_ext": c_ext, "S_bar": c_bar},
           "halo_grid": {}, "reference": {}}
    for i, cfg in enumerate(grid):
        pool = FW.run(cfg, N, 14_500_000 + 60_013 * i)
        key = (f"mis{cfg['mis_deg']:g}_flss{cfg['f_lss']:g}"
               f"_e{cfg['e_halo']:g}")
        out["halo_grid"][key] = dict(config={k: v for k, v in cfg.items()
                                             if k != "kind"}, **joint(pool))
        print(f"  {key:<26} joint FP = "
              f"{out['halo_grid'][key]['joint']['rate']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    for name, cfg in ref.items():
        pool = FW.run(cfg, N, 15_600_000 + 7 * len(name))
        out["reference"][name] = joint(pool)
        print(f"  {name:<26} joint    = "
              f"{out['reference'][name]['joint']['rate']:.3f}", flush=True)

    # where does the joint false-positive rate cross 0.05?
    fl = [(v["config"]["f_lss"], v["joint"]["rate"])
          for v in out["halo_grid"].values()
          if v["config"]["mis_deg"] == BASE_MIS and v["config"]["e_halo"] == BASE_E]
    fl.sort()
    xs = [a for a, _ in fl]
    ys = [b for _, b in fl]
    cross = None
    for i in range(1, len(xs)):
        if ys[i - 1] < 0.05 <= ys[i]:
            cross = float(np.interp(0.05, [ys[i - 1], ys[i]], [xs[i - 1], xs[i]]))
            break
    out["f_lss_at_which_joint_FP_exceeds_0.05"] = cross
    out["f_lss_curve"] = dict(f_lss=xs, joint_fp=ys)
    FW.close_pool()
    out["provenance"] = guard.stop()
    p = os.path.join(RES, "J_joint_scan.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {p}   crossing at f_lss = {cross}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
