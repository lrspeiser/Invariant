"""
Refining Test 2. The rank correlation within a cluster came out +0.643 with
12/12 positive -- but a shuffled-temperature null reached +0.625, so 36% of
random pairings did as well. That is not the model failing; it is the TEST
lacking power, because every cluster's temperature profile declines outward
with much the same shape, and a rank statistic sees only shape.

The amplitude is the part that could still discriminate. If the pressure model
is tracking THIS cluster's temperature rather than a generic profile, then

    ratio(r) = excess_obs(r) / sqrt(1 + kappa*3kT(r)/(mu m_p c^2))

should be flatter and tighter for the true pairing than for a shuffled one.
Two statistics, both amplitude-sensitive and neither rank-based:

    (a) scatter of ratio(r) WITHIN each cluster   -- does T(r) track excess(r)?
    (b) scatter of the per-cluster median ratio   -- does kappa transfer?
"""
import math, os, re, json
import numpy as np
from astropy.io import fits
from invariant_bench import Bench, KPC

SCR = os.path.dirname(os.path.abspath(__file__)) + "/"
XR = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
      "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/")
C, KEV, MP, MU = 2.99792458e8, 1.602176634e-16, 1.67262192369e-27, 0.6
OM, OL, KAPPA = 0.3, 0.7, 1.36e5
rng = np.random.default_rng(99)
BAR = "=" * 78

tex = open(SCR + "xcop_T/XCOP_thermo.tex", encoding="utf-8").read()
i = tex.index("Basic properties of the X-COP sample")
TAB = {}
for line in tex[i:i+4000].split("\n"):
    cs = [c.strip() for c in line.split("&")]
    if len(cs) < 6 or line.strip().startswith("%"): continue
    nm = cs[0].strip()
    if not re.match(r"^[A-Za-z]+\d", nm): continue
    v = lambda s: float(re.search(r"([\d.]+)", s.replace("$","")).group(1))
    z, M5, R5 = v(cs[1]), v(cs[3]), v(cs[4])
    Ez = math.sqrt(OM*(1+z)**3 + OL)
    TAB[nm] = dict(R500=R5, T500=8.85*(M5*1e14/1e15)**(2/3)*Ez**(2/3))

b = Bench(verbose=False); xc = b.d["xcop"]
ext = np.asarray(xc.extent, float)*np.ones(len(xc))/KPC
NAME = {}
for v_ in sorted(np.unique(ext)):
    h = [k for k,t in TAB.items() if abs(t["R500"]-v_) < 3.0]
    if len(h) == 1: NAME[v_] = h[0]
lv = [v_ for v_ in sorted(np.unique(ext)) if v_ not in NAME]
ln = [k for k in TAB if k not in NAME.values()]
for v_ in lv:
    k = min(ln, key=lambda q: abs(TAB[q]["R500"]-v_)); NAME[v_] = k; ln.remove(k)

CL = []
for v_ in sorted(np.unique(ext)):
    nm = NAME[v_]; t = TAB[nm]
    m = np.abs(ext-v_) < 1e-9; o = np.argsort(xc.r[m])
    rk = (xc.r[m]/KPC)[o]
    exc = (xc.nu[m]/(1.0/(1.0-np.exp(-np.sqrt(np.maximum(xc.x[m],1e-300))))))[o]
    with fits.open(os.path.join(XR, nm, f"{nm}_temperature.fits")) as hh:
        d = hh[1].data
        rw, tx = np.asarray(d["RW_X"],float), np.asarray(d["T_X"],float)
    ok = np.isfinite(rw)&np.isfinite(tx)&(tx>0)
    kT = np.interp(rk/t["R500"], rw[ok], tx[ok])*t["T500"]
    CL.append(dict(name=nm, r=rk, kT=kT, exc=exc, frac=rk/t["R500"]))

def pred(kT):
    return np.sqrt(1.0 + KAPPA*3*(kT*KEV)/(MU*MP*C**2))

def stats(pairing):
    """pairing[i] = index of the cluster whose temperature is used for i."""
    within, med = [], []
    for i2, c in enumerate(CL):
        other = CL[pairing[i2]]
        kt = np.interp(c["frac"], other["frac"], other["kT"])
        ratio = c["exc"]/pred(kt)
        good = np.isfinite(ratio)&(ratio>0)
        if good.sum() < 8: continue
        within.append(float(np.std(np.log10(ratio[good]))))
        med.append(float(np.median(ratio[good])))
    return float(np.median(within)), float(np.std(np.log10(med)))

print(BAR + "\nAmplitude test: does the model track THIS cluster's temperature?\n" + BAR)
true_w, true_m = stats(list(range(len(CL))))
print(f"\n   TRUE pairing")
print(f"      median within-cluster scatter of log10(ratio) : {true_w:.4f} dex")
print(f"      scatter of the per-cluster median ratio       : {true_m:.4f} dex")

W, M = [], []
for _ in range(3000):
    p = rng.permutation(len(CL))
    while np.any(p == np.arange(len(CL))):
        p = rng.permutation(len(CL))
    w, m = stats(p)
    W.append(w); M.append(m)
W, M = np.array(W), np.array(M)
print(f"\n   SHUFFLED pairing, 3000 derangements")
print(f"      within-cluster scatter : {np.median(W):.4f} "
      f"[{np.percentile(W,2.5):.4f}, {np.percentile(W,97.5):.4f}]")
print(f"      median-ratio scatter   : {np.median(M):.4f} "
      f"[{np.percentile(M,2.5):.4f}, {np.percentile(M,97.5):.4f}]")
pw = float(np.mean(W <= true_w)); pm = float(np.mean(M <= true_m))
print(f"\n   fraction of shuffles at least as TIGHT as the true pairing")
print(f"      on within-cluster scatter : {pw:.4f}")
print(f"      on median-ratio scatter   : {pm:.4f}")
print("\n" + BAR + "\nVERDICT\n" + BAR)
if pw > 0.05 and pm > 0.05:
    print("""   The true temperature pairing is NOT tighter than random pairings on
   either amplitude statistic. Within the precision of these twelve clusters
   the model is not tracking each cluster's own temperature -- it is matching
   a generic declining profile that every cluster shares.

   The zero-free-parameter radial prediction therefore does NOT support the
   pressure model. It does not refute it either: the test has little power,
   because cluster temperature profiles are too similar to each other for a
   twelve-object sample to tell them apart.""")
else:
    print("   The true pairing IS tighter than random. That is genuine")
    print("   temperature-specific information and supports the model.")
json.dump(dict(true_within=true_w, true_median=true_m,
               p_within=pw, p_median=pm), open(SCR+"c71.json","w"), indent=1)
