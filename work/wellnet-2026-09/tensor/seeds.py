"""Is the answer a property of the model or of one draw of 300 galaxies?

The member catalogue is a statistical population, not A2029's actual
catalogue, so every number in the map is conditioned on one realisation.  This
repeats the decisive quantities over five independent draws and reports the
scatter.  Anything whose realisation-to-realisation spread is comparable to
the effect itself is shot noise, which is exactly what the earlier
QUMOND-lumpiness calculation turned out to be beyond 300 kpc.
"""
from __future__ import annotations

import json

import numpy as np

import field as F
import mechanism as M
import wellnet as W
from wellnet import KPC, MSUN

XP = M.XP
mu = F.Mu("simple")

CASES = [
    ("lane12 survivor",
     dict(family="plaw", p=0.0, q=1.0, s=2.0, m=1.0, L=300 * KPC,
          M_0=1e11 * MSUN, exclude_nearest=False), "phi_1e12_m4", -24.68),
    ("flat-target survivor",
     dict(family="expo", p=0.0, q=1.0, s=0.5, m=1.0, L=3000 * KPC,
          M_0=1e11 * MSUN, exclude_nearest=True), "phi_1e12_m4", -14.94),
    ("ungated reference",
     dict(family="plaw", p=1.0, q=2.0, s=1.5, m=1.0, L=300 * KPC,
          M_0=1e11 * MSUN, exclude_nearest=False), "none", -4.7),
]
GD = dict(M.GATES)

out = []
for tag, kw, gname, amp in CASES:
    rows = []
    for seed in (20260903, 11, 202, 3033, 77777):
        clu, fld, mem = M.contexts(n=64, seed=seed)
        gkw = GD[gname]
        Sc = W.S_tensor(clu["sub_pts"], clu["wx"], clu["wm"], xp=XP,
                        gN_local=clu["sub_gN"], **kw)
        Sf = W.S_tensor(fld["pts"], fld["wx"], fld["wm"], xp=XP,
                        gN_local=fld["gN"], **kw)
        Sm = W.S_tensor(mem["pts"], mem["wx"], mem["wm"], xp=XP,
                        gN_local=mem["gN"], **kw)
        gc, gf, gm_ = (
            W.gate_field(gkw["kind"], PhiN=z[0], gN=z[1],
                         Phi_0=gkw.get("Phi_0", 1e12), m=gkw.get("m", 1.0),
                         xp=XP)
            for z in ((clu["sub_PhiN"], clu["sub_gN"]),
                      (fld["PhiN"], fld["gN"]), (mem["PhiN"], mem["gN"])))
        bc = gc[:, None] * Sc if not np.isscalar(gc) else gc * Sc
        bf = gf[:, None] * Sf if not np.isscalar(gf) else gf * Sf
        bm = gm_[:, None] * Sm if not np.isscalar(gm_) else gm_ * Sm
        kcl, _ = M.k_means(bc, clu["sub_rhat"], [amp], clu["sub_shells"])
        kfl, _ = M.k_means(bf, fld["rhat"], [amp], fld["masks"])
        kme, _ = M.k_means(bm, mem["rhat"], [amp], mem["masks"])
        B = M.cluster_B(clu, kcl, mu)[0]
        Bf = M.sphere_B(fld, kfl, mu)[0]
        Bm = M.sphere_B(mem, kme, mu)[0]
        rms = float(np.sqrt(np.mean((np.log10(B)
                                     - np.log10(M.BREQ)) ** 2)))
        rows.append(dict(seed=seed, B=list(B),
                         field_dex=float(np.max(np.abs(np.log10(Bf)))),
                         member_dex=float(np.max(np.abs(np.log10(Bm)))),
                         rms_dex=rms))
        print(f"   {tag:<22} seed {seed:>8}  B = "
              + " ".join(f"{v:.3f}" for v in B)
              + f"   rms {rms:.4f}  fld {rows[-1]['field_dex']:.4f}"
              f"  mem {rows[-1]['member_dex']:.4f}")
        del clu, fld, mem, Sc, Sf, Sm
        M._free()
    Ba = np.array([r["B"] for r in rows])
    print(f"   {tag:<22} mean  B = "
          + " ".join(f"{v:.3f}" for v in Ba.mean(0))
          + "   sd = " + " ".join(f"{v:.3f}" for v in Ba.std(0))
          + f"   sd/mean max {np.max(Ba.std(0)/Ba.mean(0)):.4f}")
    out.append(dict(case=tag, params=dict(kw, L_kpc=kw["L"] / KPC),
                    gate=gname, A_T=amp, rows=rows,
                    B_mean=list(Ba.mean(0)), B_sd=list(Ba.std(0)),
                    member_dex_mean=float(np.mean([r["member_dex"]
                                                   for r in rows])),
                    member_dex_sd=float(np.std([r["member_dex"]
                                                for r in rows])),
                    rms_mean=float(np.mean([r["rms_dex"] for r in rows])),
                    rms_sd=float(np.std([r["rms_dex"] for r in rows]))))
    print()

json.dump(out, open("seed_robustness.json", "w"), indent=1)
print("written seed_robustness.json")
