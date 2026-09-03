"""
IS THE h_eff TEST ACTUALLY DISCRIMINATING?

Run A measured s_h = +0.478 +- 0.019 against the cylindrical-confinement
prediction of exactly +0.5 -- 1.2 sigma, which reads as strong support.

Before that is reported as support it has to survive the control this
programme applies to everything: build a SYNTHETIC galaxy that contains no
cylindrical confinement at all, only the acceleration relation, infer h_eff
from it identically, and see whether it gives the same slope.

Algebraically it will, and that can be shown in one line. In the deep-MOND
limit g_obs = sqrt(g_bar a0), so

    h_eff = R g_bar/g_obs = R sqrt(g_bar/a0) = R sqrt(GM/R^2 a0)
          = sqrt(G M / a0)

h_eff ~ sqrt(M) is an IDENTITY of the acceleration relation, not an
independent consequence of flux confinement in a layer. This script confirms
that numerically on the same galaxies.
"""
import json, math
import numpy as np

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
G, KPC, KMS, MSUN, A0 = 6.674e-11, 3.0856775814913673e19, 1e3, 1.98892e30, 1.2e-10
UPS_D, UPS_B = 0.5, 0.7
BAR = "=" * 78

cfg = json.load(open(ROOT + "configs/sparc_rotation_curves_full_v1.json", encoding="utf-8"))
GAL = []
for g in cfg["galaxies"]:
    R, VO, EV, VB = [], [], [], []
    for row in g["rows"]:
        try:
            r, vo, ev, vg, vd, vb = (float(x) for x in row)
        except ValueError:
            continue
        v2 = vg * abs(vg) + UPS_D * vd ** 2 + UPS_B * vb ** 2
        if r <= 0 or vo <= 0 or ev <= 0 or v2 <= 0:
            continue
        R.append(r); VO.append(vo); EV.append(ev); VB.append(math.sqrt(v2))
    if len(R) < 5:
        continue
    R, VO, EV, VB = (np.array(x) for x in (R, VO, EV, VB))
    rm = R * KPC
    GAL.append(dict(name=g["name"], R=R, EV=EV,
                    g_bar=(VB * KMS) ** 2 / rm, g_obs=(VO * KMS) ** 2 / rm,
                    Mb=float((VB[-1] * KMS) ** 2 * rm[-1] / G / MSUN)))


def nu_rar(x):
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-300))))


def slope_of(which):
    HE, LM = [], []
    for x in GAL:
        gobs = (x["g_obs"] if which == "real"
                else x["g_bar"] * nu_rar(x["g_bar"] / A0))
        flat = x["g_bar"] < A0
        if flat.sum() < 3:
            continue
        h = x["R"][flat] * x["g_bar"][flat] / gobs[flat]
        w = 1.0 / np.maximum(x["EV"][flat], 1e-3) ** 2
        HE.append(float(np.sum(w * h) / np.sum(w)))
        LM.append(math.log10(x["Mb"]))
    HE, LM = np.log10(np.array(HE)), np.array(LM)
    A = np.vstack([LM, np.ones_like(LM)]).T
    s, c = np.linalg.lstsq(A, HE, rcond=None)[0]
    r = HE - (s * LM + c)
    n = len(HE)
    se = float(np.sqrt(np.sum(r**2) / (n-2) / np.sum((LM - LM.mean())**2)))
    return s, se, float(np.std(r, ddof=2)), n


print(BAR + "\nDoes a galaxy with NO confinement layer give the same slope?\n" + BAR)
print("""   'real'      = h_eff inferred from the observed rotation curve
   'RAR twin'  = the identical galaxy with g_obs replaced by the
                 acceleration-relation value, exactly, with zero scatter.
                 It contains no layer, no anisotropy, no tensor -- only
                 baryons and the relation.\n""")
print(f"   {'source':<12}{'slope s_h':>12}{'error':>9}{'scatter':>10}{'n':>6}")
print("   " + "-" * 49)
out = {}
for which, lbl in (("real", "real"), ("twin", "RAR twin")):
    s, se, sc, n = slope_of(which)
    out[lbl] = dict(slope=s, err=se, scatter=sc, n=n)
    print(f"   {lbl:<12}{s:>+12.4f}{se:>9.4f}{sc:>10.4f}{n:>6}")
print("   " + "-" * 49)
print(f"   cylindrical-confinement prediction : +0.5000")

d = abs(out["real"]["slope"] - out["RAR twin"]["slope"])
print(f"""
   The synthetic twin reproduces the slope to {d:.4f} dex. It has no
   confinement layer anywhere in its construction.

   VERDICT: h_eff ~ sqrt(M_b) is an identity of the acceleration relation,
   not evidence for flux confinement in a layer. Both hypotheses predict it
   for the same reason, so the test as written CANNOT separate them and the
   1.2-sigma agreement found in Run A is not support for either one over the
   other.

   What WOULD discriminate is the program's own vertical test (section 9):
   confinement makes a specific claim about K_z that a scalar acceleration
   relation does not. The anisotropy diagnostic

       A_dyn = (g_R/g_R,N) / (K_z/K_z,N)

   is the real discriminator -- confinement predicts A_dyn > 1, scalar
   modified gravity predicts A_dyn ~ 1. That test needs vertical data and
   the PDE solve; it cannot be shortcut through h_eff.""")
json.dump(out, open("g02_heff_control.json", "w", encoding="utf-8"), indent=1)
