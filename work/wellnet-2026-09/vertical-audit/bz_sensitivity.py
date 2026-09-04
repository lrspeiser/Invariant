"""ITEM 1 -- the exact formula for B_z, measured from the code.

The prose in `adyn/REPORT.md` states

    log B_z = 2 log sigma_z(obs) - log Upsilon_K - log h_z - log k + const

i.e. unit coefficients on Upsilon_K, h_z and k.  That is the CLOSED-FORM
idealisation.  The pipeline does not evaluate a closed form; it runs a forward
chain and fits an exponential to the result.  This script measures the actual
logarithmic sensitivities of the pipeline's B_z to every input, by finite
difference through the real code, and writes them to bz_sensitivity.json.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vaudit_core as V                                       # noqa: E402
import adyn_model as M                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B = V.Bench()
NG = B.NG
FID = dict(zU=np.log10(0.60), sc=0.15, dhz=0.0, kv=1.5, al=0.60,
           lfg=np.log10(0.25), fhg=2.0, fhzg=0.5, lo=0.3, hi=2.0)
D = 0.01                                          # dex


def logBz(mu0=None, hR=None, hz=None, gals=None, sob=None, **C):
    CC = dict(FID); CC.update(C)
    b = V.Bench(gals=gals, mu0K=B.mu0K if mu0 is None else mu0,
                hR_as=B.hR_as_v if hR is None else hR,
                hz_kpc=B.HZ_TAB if hz is None else hz,
                apc=None if (gals is not None or hR is not None) else B.APC)
    Ups = 10 ** (CC["zU"] + CC["sc"] * (b.BK - 3.4))
    hzv = b.HZ_TAB * 10 ** CC["dhz"]
    aN, _, _ = b.amp_newton(Ups, hzv, np.full(NG, 10 ** CC["lfg"]),
                            np.full(NG, CC["al"]), CC)
    s = B.OBS_AMP if sob is None else sob
    return 2 * np.log10(s / aN)


L0 = logBz()
out = {"reference_mean_logBz": float(L0.mean())}


def sens(name, note, **kw):
    L1 = logBz(**kw)
    v = float(np.median((L1 - L0) / D))
    sp = float(np.std((L1 - L0) / D))
    out[name] = dict(exponent=v, galaxy_to_galaxy_sd=sp, note=note)
    print(f"    {name:<28}{v:>9.4f}   +-{sp:>7.4f}   {note}")
    return v


print("  d log10 B_z / d log10 (input), measured through the pipeline")
print(f"    {'input':<28}{'exponent':>9}{'gal-to-gal':>12}   note")
sens("Sigma_L0  (via mu0_K,i)", "SHARED with the x-axis of the headline fit",
     mu0=B.mu0K - 2.5 * D)
sens("Upsilon_K", "common-mode IMF zero point", zU=FID["zU"] + D)
sens("h_z  (h_R held fixed)", "inferred, not measured (Bershady+2010b)",
     dhz=D)
sens("k  (vertical profile)", "prior only; spans a factor 2 (1.0 to 2.0)",
     kv=1.5 * 10 ** D)
sens("h_R  (h_z, Sigma_L0 held)", "thickness T, leakage and the R grid",
     hR=B.hR_as_v * 10 ** D)
sens("alpha = sigma_z/sigma_R", "adopted, not measured", al=0.60 * 10 ** D)
sens("f_gas", "NOT tabulated -- a prior", lfg=FID["lfg"] + D)
sens("sigma_LOS_0 (observed)", "the only measured quantity in the numerator",
     sob=B.OBS_AMP * 10 ** D)

# distance: Sigma_L0 is a SURFACE BRIGHTNESS and so distance-free; D enters
# only through h_R in metres and (via Bershady) through h_z
Bd = V.Bench()
for g in Bd.GAL:
    g.D = g.D * 10 ** D
Bd = V.Bench(gals=Bd.GAL)
Ups = 10 ** (FID["zU"] + FID["sc"] * (Bd.BK - 3.4))
aN, _, _ = Bd.amp_newton(Ups, Bd.HZ_TAB, np.full(NG, 0.25), np.full(NG, 0.60),
                         FID)
v = float(np.median((2 * np.log10(B.OBS_AMP / aN) - L0) / D))
out["distance (h_z from catalogue)"] = dict(
    exponent=v, note="Sigma_L0 is a surface brightness: distance-free")
print(f"    {'distance':<28}{v:>9.4f}   {'':>9}   "
      f"Sigma_L0 is a surface brightness: distance-free")

# inclination
gi = [copy.copy(g) for g in B.GAL]
for g in gi:
    g.incl = min(g.incl * 10 ** D, 85.0)
v = float(np.median((logBz(gals=gi) - L0) / D))
out["inclination"] = dict(exponent=v, note="projection factor only here; the "
                          "mu0_K,i correction is applied upstream by DiskMass")
print(f"    {'inclination':<28}{v:>9.4f}   {'':>9}   projection factor")

# fit window
for lo, hi, tag in ((0.2, 2.0, "window lo 0.3 -> 0.2"),
                    (0.5, 2.0, "window lo 0.3 -> 0.5"),
                    (0.3, 1.5, "window hi 2.0 -> 1.5"),
                    (0.3, 2.5, "window hi 2.0 -> 2.5")):
    v = float(np.median(logBz(lo=lo, hi=hi) - L0))
    out[tag] = dict(shift_dex=v)
    print(f"    {tag:<28}{v:>9.4f} dex")

# ---------- what the abscissa is, exactly
mu = B.mu0K
out["abscissa"] = dict(
    definition="log10 Sigma_L0 = 0.4 * (M_K,sun + 21.572 - mu0_K,i), "
               "Lsun/pc^2, K band, DISK ONLY, inclination-corrected",
    MSUN_K=M.MSUN_K, mu0_min=float(mu.min()), mu0_max=float(mu.max()),
    range_dex=float(0.4 * (mu.max() - mu.min())),
    var_dex2=float(np.var(B.log_sigma0())),
    median_e_mu0_mag=float(np.median([g.emu0K for g in B.GAL])),
    median_e_logSigma_dex=float(np.median([0.4 * g.emu0K for g in B.GAL])))
print(f"\n  abscissa log10 Sigma_L0: range {out['abscissa']['range_dex']:.3f} dex,"
      f" var {out['abscissa']['var_dex2']:.5f},"
      f" median error {out['abscissa']['median_e_logSigma_dex']:.4f} dex")

# ---------- the s2 floor that produces the pathological draws
r = np.random.default_rng(999)
nfl = 0
worst = 0.0
for _ in range(1600):
    C = V.Bench.draw_common(r)
    Ups, hz, fg, al, sob = B.draw_pergal(r, C)
    b = B.newton_chain(Ups, hz, fg, al, C["kv"], C["fhg"], C["fhzg"])
    f = float(np.mean(b["s2"] <= 1e-25))
    nfl += f > 0
    worst = max(worst, f)
out["s2_floor"] = dict(
    draws_with_any_floored_cell=int(nfl), of=1600, worst_fraction=float(worst),
    note="np.maximum(..., 1e-30) in newton_chain: when the leakage term "
         "L_s h_z^2 (1/R) dVc^2/dR exceeds 2 pi G h_z Sigma, sigma_z^2 is "
         "clipped and log B_z diverges for that galaxy")
print(f"\n  sigma_z^2 floor hit in {nfl} of 1600 nuisance draws "
      f"(worst {100*worst:.1f}% of cells) -- this is what produces the "
      f"+10 dex slope outliers")

with open(os.path.join(HERE, "bz_sensitivity.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("\n  wrote bz_sensitivity.json")
