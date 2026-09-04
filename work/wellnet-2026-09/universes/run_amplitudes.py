"""run_amplitudes.py -- translate every knob into predicted OBSERVABLES.

The first version reported only the fractional change in the MATTER
acceleration, which is identically zero for three of the six deformations:
the tensor puts its signal in an l=2 potential rather than the monopole, the
slip universe changes only the LIGHT potential, and the path universe does not
touch gravity at all.  This pass reports, for every knob value:

  d ln g_matter    max fractional change in the radial matter acceleration
  d ln g_light     max fractional change in the light (deflection) potential
  quadrupole       max |chi/Phi|, the l=2 fraction of the lensing potential
  d ln (1+z)       max fractional redshift excess on the path

all against the base U3 at the same scene, plus the responsiveness of z to
log amplitude over the UNSATURATED part of each scan.  Generates no corpora.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from universes import generate as gn          # noqa: E402
from universes import physics as ph           # noqa: E402
from universes import stats as st             # noqa: E402

RES = os.path.join(HERE, "results")
ZSAT = 8.4          # anything at or above this is at the permutation-null cap


def main():
    lib = gn.get_lib()
    with open(os.path.join(RES, "E3_amplitude_scans.json")) as f:
        E3 = json.load(f)

    e7 = {}
    for uid, v in E3.items():
        amps = sorted({r["amp"] for r in v["rows"]})
        rows = []
        for amp in amps:
            dgm, dgl, quad, dz = [], [], [], []
            for gm in lib.geoms[:10]:
                c, rg = gm.clu, gm.rg
                u3 = ph.draw_universe("U03_mond_scalar", np.random.default_rng(5))
                ux = ph.draw_universe(uid, np.random.default_rng(5), knob=amp)
                F3 = ph.cluster_field(u3, c, rg)
                Fx = ph.cluster_field(ux, c, rg)
                gx, gl = Fx["g_m"].copy(), Fx["g_l"].copy()
                if uid == "U06_wellnet":
                    d = amp * np.gradient(gm.Ex_bar, rg)
                    gx, gl = gx + d, gl + d
                sel = (rg > 0.2 * c.R500) & (rg < 2.0 * c.R500)
                dgm.append(float(np.max(np.abs(gx[sel] / F3["g_m"][sel] - 1.0))))
                dgl.append(float(np.max(np.abs(gl[sel] / F3["g_l"][sel] - 1.0))))
                if Fx["chi"] is not None:
                    Phi = ph.cluster_potential_1d(rg, gl)
                    quad.append(float(np.max(np.abs(Fx["chi"][sel] / Phi[sel]))))
            if uid == "U09_path_redshift":
                z = np.array([0.05, 0.2, 0.5, 1.0])
                ux = ph.draw_universe(uid, np.random.default_rng(5), knob=amp)
                op, _ = ph.observed_redshift(ux, z, np.full_like(z, 0.6))
                dz = [float(np.max(op / (1 + z) - 1.0))]
            rows.append({
                "amp": amp,
                "max_dln_g_matter_vs_U03": float(np.median(dgm)),
                "max_dln_g_light_vs_U03": float(np.median(dgl)),
                "max_potential_quadrupole_fraction": float(np.median(quad)) if quad else None,
                "max_fractional_redshift_excess": float(dz[0]) if dz else None,
            })
        # responsiveness on the UNSATURATED part, against log amplitude
        rr = [r for r in v["rows"] if r["amp"] > 0 and r["z"] < ZSAT]
        resp = st.responsiveness([np.log10(r["amp"]) for r in rr],
                                 [r["z"] for r in rr]) if len(rr) >= 3 else \
            {"slope": float("nan"), "se": float("nan"), "responsive": False,
             "n": len(rr), "t": float("nan")}
        e7[uid] = {"knob": ph.KNOB[uid],
                   "fiducial": ph.FIDUCIAL[uid][ph.KNOB[uid]],
                   "rows": rows,
                   "n_unsaturated_points": len(rr),
                   "responsiveness_dz_dlog10amp_unsaturated": resp}
        print(f"  {uid}: dz/dlog10(amp) = {resp['slope']:.2f} +/- {resp['se']:.2f} "
              f"on {len(rr)} unsaturated points", flush=True)
    e7["_note"] = ("median over 10 library clusters of the maximum over 0.2-2.0 R500; "
                   "g_matter moves stars and gas, g_light deflects photons, the "
                   "quadrupole is the l=2 fraction of the lensing potential, and the "
                   "redshift excess is the path term at void fraction 0.6. A null from "
                   "a detector with no power below the predicted amplitude says nothing.")
    e7["_zsat"] = ZSAT
    with open(os.path.join(RES, "E7_observable_amplitudes.json"), "w") as f:
        json.dump(e7, f, indent=1, default=float)
    print("rewrote E7_observable_amplitudes.json")


if __name__ == "__main__":
    main()
