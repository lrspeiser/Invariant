"""The diagnostic figure: does |Phi_b| move nu at fixed g_bar?"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyse

LANE = analyse.LANE
A0 = 1.2e-10
COL = {1: "#1f77b4", 2: "#2ca02c", 3: "#8c564b", 4: "#ff7f0e",
       5: "#d62728", 6: "#9467bd"}
LBL = {1: "1 field galaxies (SPARC)", 2: "2 small groups (SDSS, tier 2)",
       3: "3 poor groups (X-ray)", 4: "4 rich groups (X-ray)",
       5: "5 low-mass clusters", 6: "6 massive clusters (X-COP)"}

d = analyse.load()
res = json.load(open(os.path.join(LANE, "results.json")))
lg, lp, lr = d["lg"], d["lp"], d["lr"]
nu_rar = 1.0 / (1.0 - np.exp(-np.sqrt(d["g_bar"] / A0)))
dev = np.log10(d["nu_obs"] / nu_rar)

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))

# ---- A: coverage in the (g_bar, |Phi_b|) plane ------------------------
a = ax[0]
for k in sorted(COL):
    m = d["rank"] == k
    a.scatter(lg[m], lp[m], s=6 if k in (1, 6) else 26, alpha=.5 if k == 1 else .85,
              c=COL[k], label=LBL[k], lw=0, zorder=10 - k)
lo, hi = -11.50, -11.25
a.axvspan(lo, hi, color="0.85", zorder=0)
sel = (lg >= lo) & (lg < hi)
a.annotate(f"widest matched-$g_{{\\rm bar}}$ bin:\n"
           f"{lp[sel].max()-lp[sel].min():.2f} dex in $|\\Phi_b|$,\n"
           f"all 6 rungs present",
           xy=(-11.30, lp[sel].max()), xytext=(-10.6, 11.55), fontsize=8.5,
           ha="left", arrowprops=dict(arrowstyle="->", lw=1.1))
a.set_ylim(7.7, 12.6)
a.set_xlabel(r"$\log_{10}\ g_{\rm bar}\ \ [{\rm m\,s^{-2}}]$")
a.set_ylabel(r"$\log_{10}\ |\Phi_b|\ \ [{\rm m^2\,s^{-2}}]$")
a.set_title("A.  the leverage: $|\\Phi_b|$ at matched $g_{\\rm bar}$")
a.legend(fontsize=7.5, loc="lower right", framealpha=.95)
a.axvline(math.log10(A0), ls=":", c="k", lw=1)
a.text(math.log10(A0) + .05, 7.9, "$a_0$", fontsize=9)
a.grid(alpha=.25)

# ---- B: the diagnostic itself -----------------------------------------
b = ax[1]
win = (lg >= -11.60) & (lg <= -10.41)
for k in sorted(COL):
    m = win & (d["rank"] == k)
    if m.sum() == 0:
        continue
    b.scatter(lp[m], dev[m], s=6 if k in (1, 6) else 26,
              alpha=.35 if k == 1 else .8, c=COL[k], lw=0, zorder=10 - k)
prof = res["by_rung_in_window"]
xs = np.array([p[4] for p in prof])
ys = np.array([p[6] for p in prof])
b.plot(xs, ys, "o-", c="k", ms=8, lw=2, zorder=30, label="rung medians")
beta = res["beta"]["observed"]
xx = np.linspace(lp[win].min(), lp[win].max(), 50)
c0 = np.median(ys) - beta * np.median(xs)
b.plot(xx, c0 + beta * xx, "--", c="crimson", lw=2, zorder=29,
       label=fr"potential depth, $\beta$={beta:+.3f} ($q$={2*beta:+.2f})")
gal = xs[0]
b.plot([xs[0], xs[0]], [ys[0], ys[0]], " ")
b.hlines([ys[0]], lp[win].min(), 10.0, color="navy", lw=2, zorder=28)
b.hlines([np.median(ys[1:])], 10.0, lp[win].max(), color="navy", lw=2,
         zorder=28, label="step: galaxy / not-a-galaxy (wins by $\\Delta$BIC 17.6)")
b.axhline(0, c="k", lw=.8)
b.set_xlabel(r"$\log_{10}\ |\Phi_b|$")
b.set_ylabel(r"$\log_{10}\ (\nu_{\rm obs}/\nu_{\rm RAR})$")
b.set_title("B.  the diagnostic, inside $0.021 < g_{\\rm bar}/a_0 < 0.32$")
b.legend(fontsize=8, loc="upper left", framealpha=.95)
b.grid(alpha=.25)
b.set_ylim(-1.2, 1.6)

# ---- C: where the leverage comes from ---------------------------------
c = ax[2]
wc = res["leverage"]["within_class"]
ks = sorted(int(k) for k in wc)
vals = [wc[str(k)]["sd_after_gbar"] for k in ks]
c.bar([k - .2 for k in ks], vals, width=.4, color=[COL[k] for k in ks],
      label="within one rung")
tot = res["confound"]["sd_after_gbar"]
lab = res["confound"]["sd_after_gbar_and_class"]
c.axhline(tot, c="k", lw=2, label=f"whole ladder: {tot:.3f} dex")
c.axhline(lab, c="k", ls="--", lw=2,
          label=f"after removing the class label: {lab:.3f} dex")
c.axhline(0.309, c="crimson", ls=":", lw=2, label="SPARC alone: 0.309 dex")
c.set_xticks(ks)
c.set_xticklabels([LBL[k].split(" ", 1)[1].replace(" (", "\n(") for k in ks],
                  fontsize=7.5, rotation=20, ha="right")
c.set_ylabel(r"sd of $\log_{10}|\Phi_b|$ at fixed $g_{\rm bar}$  [dex]")
c.set_title("C.  86% of the leverage IS the class label")
c.legend(fontsize=8)
c.grid(alpha=.25, axis="y")

fig.tight_layout()
p = os.path.join(LANE, "potential_depth_diagnostic.png")
fig.savefig(p, dpi=145)
print("wrote", p)
