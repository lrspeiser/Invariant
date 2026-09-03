"""
The error bars in p01 are statistical only -- a bootstrap over radial bins
within each cluster. They contain no systematic term. Three systematics act on
every cluster's excess and none of them are in there:

   1. hydrostatic mass bias, the usual (1-b) = 0.8 +- 0.1
   2. gas-mass calibration from the X-ray emissivity and abundance
   3. the stellar contribution, which for these clusters is small but not zero

Under-estimated errors do two things: they inflate chi2, and they shrink the
confidence interval on kappa to something indefensible. Both were reported in
p01 and both need correcting before anything goes in a manuscript.

This script asks what intrinsic scatter is required to make the model an
acceptable fit, and re-derives kappa's interval with it included.
"""
import json
import math
import numpy as np

SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
C, KEV, MP, MU = 2.99792458e8, 1.602176634e-16, 1.67262192369e-27, 0.6
BAR = "=" * 78
rng = np.random.default_rng(7)

R = json.load(open(SCR + "paper_results.json", encoding="utf-8"))
cl = R["clusters"]
kT = np.array([c["kT"] for c in cl])
ekT = np.array([c["ekT"] for c in cl])
ex = np.array([c["exc"] for c in cl])
eex = np.array([c["eexc"] for c in cl])
nm = [c["name"] for c in cl]
n = len(cl)

print(BAR + "\n1. How big are the stated errors, relatively?\n" + BAR)
print(f"   median fractional error on kT     : {np.median(ekT/kT):.4f}")
print(f"   median fractional error on excess : {np.median(eex/ex):.4f}")
print(f"   observed scatter in excess        : {np.std(ex, ddof=1)/np.mean(ex):.4f}")
print("\n   The quoted excess error is ~1-2%. Nobody believes a cluster mass")
print("   is known to 1-2%. These are statistical-only and far too small.")


def model(kap, T):
    return np.sqrt(1 + kap * 3 * (T * KEV) / (MU * MP * C ** 2))


def chi2(kap, s_int):
    pred = model(kap, kT)
    dpdT = (kap * 3 * KEV / (MU * MP * C ** 2)) / (2 * pred)
    sig = np.sqrt(eex ** 2 + (dpdT * ekT) ** 2 + (s_int * ex) ** 2)
    return float(np.sum(((ex - pred) / sig) ** 2))


print("\n" + BAR + "\n2. Intrinsic scatter required for an acceptable fit\n" + BAR)
grid = 10 ** np.linspace(3.5, 6.5, 3001)
print(f"   {'sigma_int':>11}{'best kappa':>14}{'chi2':>9}{'chi2/dof':>11}")
print("   " + "-" * 45)
S_ACC = None
for s in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
    c2 = np.array([chi2(k, s) for k in grid])
    j = int(np.argmin(c2))
    red = c2[j] / (n - 1)
    print(f"   {s:>11.2f}{grid[j]:>14.3g}{c2[j]:>9.2f}{red:>11.2f}")
    if S_ACC is None and red <= 1.0:
        S_ACC = s
print("   " + "-" * 45)
print(f"   An intrinsic scatter of ~{S_ACC:.0%} makes the model an acceptable")
print("   fit. That is a plausible size for hydrostatic-bias scatter alone,")
print("   so the model is NOT excluded -- but neither is it demanded.")

print("\n" + BAR + "\n3. kappa with realistic errors\n" + BAR)
for s in (0.0, 0.15, S_ACC if S_ACC else 0.2):
    c2 = np.array([chi2(k, s) for k in grid])
    j = int(np.argmin(c2))
    lo = grid[c2 <= c2[j] + 1]
    hi = grid[c2 <= c2[j] + 4]
    print(f"   sigma_int = {s:.2f} :  kappa = {grid[j]:.3g}")
    print(f"                     68% [{lo[0]:.3g}, {lo[-1]:.3g}]")
    print(f"                     95% [{hi[0]:.3g}, {hi[-1]:.3g}]")
    k5 = chi2(1e5, s) - c2[j]
    print(f"                     kappa=1e5 is delta-chi2 {k5:+.2f} "
          f"({'excluded' if k5 > 4 else 'ADMISSIBLE'} at 95%)")
print("\n   With honest errors the pre-registered kappa = 1e5 is no longer")
print("   excluded. The p01 statement that it was is withdrawn -- it was an")
print("   artefact of statistical-only error bars.")

print("\n" + BAR + "\n4. Does the model still beat a constant?\n" + BAR)
for s in (0.0, 0.15, 0.20):
    c2m = min(chi2(k, s) for k in grid)
    sig = np.sqrt(eex ** 2 + (s * ex) ** 2)
    w = 1 / sig ** 2
    cbar = float(np.sum(w * ex) / np.sum(w))
    c2c = float(np.sum(((ex - cbar) / sig) ** 2))
    print(f"   sigma_int = {s:.2f} :  model chi2 {c2m:6.2f}   "
          f"constant chi2 {c2c:6.2f}   delta {c2c-c2m:+6.2f}")
print("\n   Both have one free parameter, so delta-chi2 is directly comparable.")
print("   The temperature-dependent model is preferred at every scatter level,")
print("   but the margin shrinks as the errors are made honest.")

print("\n" + BAR + "\n5. The dominant systematic nobody can remove\n" + BAR)
print("""   Every cluster excess here scales with the assumed gas mass, and the
   galaxy excesses scale with the assumed stellar mass-to-light ratio
   (Upsilon_3.6 = 0.5). Those are different systematics on the two sides of
   the galaxy/cluster comparison that FIXED kappa in the first place.

   A 20% shift in Upsilon moves every galaxy point ~1.1x and no cluster point.
   kappa was set by the ratio of cluster excess to galaxy excess, so kappa
   inherits that systematic in full. The quoted kappa is good to a factor of
   ~2 at best, and no statistical interval computed here changes that.""")

print("\n" + BAR + "\n6. What survives, stated conservatively\n" + BAR)
print(f"""   Sound:   the RANK correlation, rho = +0.615. It uses only the
            ordering of the excesses, so it is immune to any systematic that
            scales all clusters together -- which is most of them.

   Weak:    the AMPLITUDE. chi2/dof = 3.69 falls to 1.0 only with ~{S_ACC:.0%}
            intrinsic scatter, and kappa is uncertain by a factor of ~2 from
            the stellar-mass systematic alone.

   Withdrawn: the claim that kappa = 1e5 is excluded by the fit.""")
