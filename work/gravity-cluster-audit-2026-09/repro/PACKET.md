# Reproducibility packet — `c70_endogeneity.py` and `c71_amplitude.py`

Prepared 2026-09-03. Scope: **documentation and verification only.** No
scientific interpretation is offered, no bug found here has been fixed, and no
register, artifact or summary document has been modified. The two pre-existing
result files (`c70_endogeneity.json`, `c71.json`) were snapshotted before any
re-execution and are byte-identical afterwards
(`repro/_preexisting/` holds the snapshots).

**Headline for a reviewer in a hurry:** every number reported in the write-up
for these two tests reproduces exactly, to the last printed digit. The problems
this packet documents are not arithmetic errors. They are (a) the provenance of
`kappa = 1.36e5` — item 12 — (b) statistics computed and never reported —
item 10 — and (c) two mechanical properties of the code a reader would not
guess from the write-up — items 4 and 8.

---

## 1. Repository state, and the fact that the scripts are not in it

### 1a. The git repository

Path: `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration`

```
$ git rev-parse HEAD
b2b54146c1498b62170ac362ae83195b496e6117

$ git log -1 --format="%H%n%an <%ae>%n%ad%n%s"
b2b54146c1498b62170ac362ae83195b496e6117
Leonard Speiser <leonard@horizon3.net>
Thu Sep 3 15:30:28 2026 -0700
gravitylab: axisymmetric tensor solver, validated, and A_dyn confirmed as a discriminator

$ git status
On branch feat/gravitylab-runA
nothing to commit, working tree clean

$ git status --porcelain
(no output)

$ git rev-parse --abbrev-ref HEAD
feat/gravitylab-runA
```

The working tree is clean. Note that HEAD is on `feat/gravitylab-runA`, not on
`main`, and that the HEAD commit (15:30 local) post-dates both test runs
(15:13 and 15:14 local) — it concerns unrelated work.

### 1b. The test scripts are NOT under version control

**Stated plainly: `c70_endogeneity.py`, `c71_amplitude.py` and every supporting
script named in this packet live in the session scratchpad
`C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\`,
which is not a git repository and is not inside one.** There is no commit, no
tag and no branch that contains them. `git log` cannot be used to audit their
history. A search of the repository for the strings `c70_endogeneity` and
`c71_amplitude` returns nothing.

In place of commit SHAs, the SHA-256 of each file as it stands:

| file | bytes | SHA-256 |
|---|---:|---|
| `c70_endogeneity.py` | 9,015 | `875ab24bb4a4effd1c5e3389bfc4056a0bba5de33e88ff3feb52e7dc23af03de` |
| `c71_amplitude.py` | 5,633 | `024b08966adf7af6d0fc0dc8f5aed793cd83253b85f0de735fc5ba22a75e36e8` |
| `invariant_bench.py` | 20,139 | `fe817b228d6d495c15d4bdaa84144ede30c638bb9a39ea1124700e7a309dde0f` |
| `p01_rigorous.py` | 13,316 | `ccd752ae610a6da28ef2b54b457c9b38e7e26dc2482a3a7d7f778169cb046add` |
| `p02_systematics.py` | 5,617 | `2c0f15b5efd1d5a53f9aa5a14d5eb96cc2f209152b0b8f834c44408bc689d6bc` |
| `m08_pressure_source.py` | 7,132 | `ab6c7822ada3bf511d95e2c67eaeb9f28e94e6a40c57e0af1d70cad4c15c4a36` |
| `m10_xcop_temperature.py` | 8,786 | `1910c31160732d574c1fe918e8ad2d61fb9a8ebeb76eca9a21ba73ac12ef262f` |
| `m11_verify_identity.py` | 4,337 | `8b16e8adda803669bc588aa323a217bada68d662c894e6ab505062bddb87e94d` |
| `m12_redo_correct.py` | 8,312 | `aae35ba2fea07519787ab6117e53195eb4d79be48731927403db4818e792a3d1` |
| `c70_endogeneity.json` (result) | 211 | `f77a0f685dd3b3b27f6da86d1fae3b49d9f10307e618b8ee22e648373821820a` |
| `c71.json` (result) | 148 | `2bf5b0fe242ea05c1e1d8f63984008a31008175e3b5e7ce8ec1a22c6532a2f0e` |

A scratchpad is by design ephemeral. **If this directory is cleared, the two
tests become unreproducible**, because the scripts exist nowhere else. That is
a reproducibility defect independent of anything in the statistics.

---

## 2. Exact command lines, from a clean shell

Working directory must be the scratchpad: `invariant_bench.py` is imported by
name (so it must be on `sys.path`), `c71_amplitude.py` resolves its data
directory from `__file__`, and both read `xcop_T/XCOP_thermo.tex` relative to
that directory.

```bash
cd "C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad"
export PYTHONIOENCODING=utf-8
python c70_endogeneity.py
python c71_amplitude.py
```

PowerShell equivalent:

```powershell
Set-Location "C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad"
$env:PYTHONIOENCODING = "utf-8"
python c70_endogeneity.py
python c71_amplitude.py
```

The interpreter used is `C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe`
(see item 5). Wall time on this machine: c70 1.1 s, c71 1.4 s. Neither script
takes arguments, reads stdin, or uses the network. Each writes exactly one file
(`c70_endogeneity.json`, `c71.json` respectively, both into the scratchpad
root); **re-running them overwrites those two files.**

The original invocations, recovered from the session transcript, were:

```bash
PYTHONIOENCODING=utf-8 timeout 2000 python c70_endogeneity.py 2>&1 | tail -55
PYTHONIOENCODING=utf-8 timeout 2000 python c71_amplitude.py 2>&1 | tail -30
```

`c70_endogeneity.py` emits **56** lines; `tail -55` therefore discarded line 1.
Line 1 is blank (the first `head()` call prints a leading newline), so no
content was lost — but the operator did not see the full output. `c71` emits 28
lines, so `tail -30` discarded nothing.

Auxiliary commands used to build this packet (all in `repro/`):

```bash
python repro/trace_inputs.py c70_endogeneity.py repro/inputs_c70.json   # item 4
python repro/trace_inputs.py c71_amplitude.py  repro/inputs_c71.json    # item 4
python repro/build_packet_data.py                                       # items 6,7,9
python repro/kappa_sensitivity.py                                       # item 12
python repro/tie_and_clamp_audit.py                                     # items 4,8
python repro/scan_history.py                                            # item 10
```

---

## 3. Full source of both test scripts, verbatim

### 3a. `c70_endogeneity.py` (9,015 bytes, SHA-256 `875ab24b…03de`)

```python
"""
TWO TESTS THE CRITIQUE DEMANDS, both runnable on data in hand.

TEST 1 -- ENDOGENEITY. The objection: temperature sits on BOTH sides of the
correlation. The hydrostatic acceleration is

    g_HSE(r) = -(kT/(mu m_p r)) [ dln n_e/dln r + dln T/dln r ]

so g_obs is proportional to T at fixed logarithmic slopes, while g_bar depends
on n_e and not on T. Taking logs of the excess,

    ln excess = ln T + ln S - ln( nu_RAR * g_bar * r ) + const

the +ln T term is there by construction. A correlation with kT is therefore
partly guaranteed, and the question is HOW MUCH.

This is decidable because the two hypotheses predict DIFFERENT SLOPES:

    pure algebraic channel        d ln excess / d ln T  ->  1
    pressure model at kappa       d ln excess / d ln T  =  (1/2) x/(1+x)
                                  with x = kappa*3kT/(mu m_p c^2) ~ 3.6
                                  giving ~ 0.39

So measure the slope. Near 1 means the signal is bookkeeping; near 0.4 means it
is not, and the difference is large enough to see with twelve points.

TEST 2 -- THE RADIAL PROFILE, zero free parameters. Every cluster test so far
used ONE number per cluster, the median excess. But the pressure model makes a
prediction at every radius, because T varies with radius inside each cluster:

    excess(r) = sqrt( 1 + kappa * 3kT(r) / (mu m_p c^2) )

With kappa already fixed there is nothing left to tune. If the model is right
it must track the radial run of the excess WITHIN each cluster, not merely its
cluster-to-cluster average. That is a far stronger test and it has never been
run.
"""
import json
import math
import os
import re

import numpy as np
from astropy.io import fits

from invariant_bench import Bench, KPC

SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
XR = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
      "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/")
C, KEV, MP, MU = 2.99792458e8, 1.602176634e-16, 1.67262192369e-27, 0.6
A0, OM, OL = 1.2e-10, 0.3, 0.7
KAPPA = 1.36e5
BAR = "=" * 78
rng = np.random.default_rng(4242)


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def nu_rar(x):
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-300))))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / d) if d > 0 else float("nan")


# ------------------------------------------------------------------- ingest
tex = open(SCR + "xcop_T/XCOP_thermo.tex", encoding="utf-8").read()
i = tex.index("Basic properties of the X-COP sample")
TAB = {}
for line in tex[i:i + 4000].split("\n"):
    cells = [c.strip() for c in line.split("&")]
    if len(cells) < 6 or line.strip().startswith("%"):
        continue
    nm = cells[0].strip()
    if not re.match(r"^[A-Za-z]+\d", nm):
        continue

    def val(s):
        m = re.search(r"([\d.]+)", s.replace("$", ""))
        return float(m.group(1)) if m else float("nan")
    z, M500, R500 = val(cells[1]), val(cells[3]), val(cells[4])
    Ez = math.sqrt(OM * (1 + z) ** 3 + OL)
    TAB[nm] = dict(z=z, M500=M500, R500=R500,
                   T500=8.85 * (M500 * 1e14 / 1e15) ** (2 / 3) * Ez ** (2 / 3))

b = Bench(verbose=False)
xc = b.d["xcop"]
ext = np.asarray(xc.extent, float) * np.ones(len(xc)) / KPC
NAME = {}
for v in sorted(np.unique(ext)):
    hit = [k for k, t in TAB.items() if abs(t["R500"] - v) < 3.0]
    if len(hit) == 1:
        NAME[v] = hit[0]
left_v = [v for v in sorted(np.unique(ext)) if v not in NAME]
left_n = [k for k in TAB if k not in NAME.values()]
for v in left_v:
    k = min(left_n, key=lambda q: abs(TAB[q]["R500"] - v))
    NAME[v] = k
    left_n.remove(k)

CL = []
for v in sorted(np.unique(ext)):
    nm = NAME[v]
    t = TAB[nm]
    m = np.abs(ext - v) < 1e-9
    o = np.argsort(xc.r[m])
    r_kpc = (xc.r[m] / KPC)[o]
    gobs = xc.go[m][o]
    gbar = xc.gb[m][o]
    exc_r = (xc.nu[m] / nu_rar(xc.x[m]))[o]
    with fits.open(os.path.join(XR, nm, f"{nm}_temperature.fits")) as h:
        d = h[1].data
        rw = np.asarray(d["RW_X"], float)
        tx = np.asarray(d["T_X"], float)
    ok = np.isfinite(rw) & np.isfinite(tx) & (tx > 0)
    kT_r = np.interp(r_kpc / t["R500"], rw[ok], tx[ok]) * t["T500"]
    CL.append(dict(name=nm, r=r_kpc, kT=kT_r, exc=exc_r, gobs=gobs, gbar=gbar,
                   kT_med=float(np.median(kT_r)),
                   exc_med=float(np.median(exc_r)),
                   gbar_med=float(np.median(gbar)),
                   gobs_med=float(np.median(gobs)),
                   elim=nm in ("A644", "A2319")))

head("TEST 1  --  is the temperature correlation algebraic?")
kT = np.array([c["kT_med"] for c in CL])
ex = np.array([c["exc_med"] for c in CL])
gb = np.array([c["gbar_med"] for c in CL])
go = np.array([c["gobs_med"] for c in CL])

print("   First, where the shared dependence would enter:\n")
print(f"   rho(kT, g_obs)   = {spearman(kT, go):+.3f}   "
      f"<- g_HSE carries T directly")
print(f"   rho(kT, g_bar)   = {spearman(kT, gb):+.3f}   "
      f"<- g_bar carries n_e, not T")
print(f"   rho(kT, excess)  = {spearman(kT, ex):+.3f}")

lt, le = np.log(kT), np.log(ex)
A = np.vstack([lt, np.ones_like(lt)]).T
slope, icept = np.linalg.lstsq(A, le, rcond=None)[0]
res = le - (slope * lt + icept)
se = float(np.sqrt(np.sum(res ** 2) / (len(lt) - 2)
                   / np.sum((lt - lt.mean()) ** 2)))
x_typ = KAPPA * 3 * (5.0 * KEV) / (MU * MP * C ** 2)
pred_pressure = 0.5 * x_typ / (1.0 + x_typ)
print(f"""
   Now the slope, which the two explanations disagree about:

      d ln(excess) / d ln(kT)  measured   = {slope:+.3f} +- {se:.3f}

      pure algebraic channel   predicts   = +1.000
      pressure model, kappa = {KAPPA:.2e}    = {pred_pressure:+.3f}
        (x = kappa*3kT/(mu m_p c^2) = {x_typ:.2f} at 5 keV)""")
d_alg = abs(slope - 1.0) / se
d_prs = abs(slope - pred_pressure) / se
print(f"""
      distance from algebraic  : {d_alg:.1f} sigma
      distance from pressure   : {d_prs:.1f} sigma""")

head("TEST 2  --  the radial profile, with kappa already fixed")
print("""   excess(r) = sqrt( 1 + kappa * 3kT(r)/(mu m_p c^2) ),  no free parameters.
   Within each cluster T varies with radius, so the model predicts how the
   excess RUNS, not just its average. Tested per cluster below.\n""")
print(f"   {'cluster':<10}{'n':>4}{'kT range keV':>16}{'exc obs range':>16}"
      f"{'exc pred range':>17}{'rho(r)':>9}{'med ratio':>11}")
print("   " + "-" * 83)
rows = []
for c in CL:
    pred = np.sqrt(1.0 + KAPPA * 3 * (c["kT"] * KEV) / (MU * MP * C ** 2))
    rho_r = spearman(c["exc"], pred)
    ratio = float(np.median(c["exc"] / pred))
    rows.append((c["name"], rho_r, ratio, c["elim"]))
    print(f"   {c['name']:<10}{len(c['r']):>4}"
          f"{f'{c[chr(107)+chr(84)].min():.1f}-{c[chr(107)+chr(84)].max():.1f}':>16}"
          f"{f'{c[chr(101)+chr(120)+chr(99)].min():.2f}-{c[chr(101)+chr(120)+chr(99)].max():.2f}':>16}"
          f"{f'{pred.min():.2f}-{pred.max():.2f}':>17}"
          f"{rho_r:>+9.3f}{ratio:>11.3f}")
print("   " + "-" * 83)
rr = np.array([r[1] for r in rows])
ra = np.array([r[2] for r in rows])
print(f"   median within-cluster rho(observed, predicted) : {np.median(rr):+.3f}")
print(f"   clusters with rho > 0                          : "
      f"{int((rr > 0).sum())} of {len(rr)}")
print(f"   normalisation ratio, median across clusters    : {np.median(ra):.3f}")
print(f"   scatter of that ratio                          : {np.std(ra):.3f}")

head("A cheaper null: does the same test pass on a shuffled temperature map?")
null = []
for _ in range(2000):
    perm = rng.permutation(len(CL))
    vals = []
    for i2, c in enumerate(CL):
        other = CL[perm[i2]]
        kt = np.interp(np.linspace(0, 1, len(c["r"])),
                       np.linspace(0, 1, len(other["kT"])), other["kT"])
        p = np.sqrt(1.0 + KAPPA * 3 * (kt * KEV) / (MU * MP * C ** 2))
        vals.append(spearman(c["exc"], p))
    null.append(np.median(vals))
null = np.array(null)
obs = float(np.median(rr))
print(f"   observed median within-cluster rho : {obs:+.3f}")
print(f"   shuffled-temperature null          : {np.median(null):+.3f} "
      f"[{np.percentile(null,2.5):+.3f}, {np.percentile(null,97.5):+.3f}]")
print(f"   fraction of null >= observed       : {float(np.mean(null >= obs)):.4f}")

json.dump(dict(slope=float(slope), slope_err=se,
               pred_pressure=float(pred_pressure),
               within_rho_median=float(np.median(rr)),
               ratio_median=float(np.median(ra)),
               null_frac=float(np.mean(null >= obs))),
          open(SCR + "c70_endogeneity.json", "w", encoding="utf-8"), indent=1)
print("\n   wrote c70_endogeneity.json")
```

### 3b. `c71_amplitude.py` (5,633 bytes, SHA-256 `024b0896…e36e`)

```python
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
```

---

## 4. Every input file actually opened at runtime

Determined by instrumentation, not by reading the source: `repro/trace_inputs.py`
installs a `sys.addaudithook` on CPython's `open` audit event (which fires for
`builtins.open`, `os.open` and `io.open`, and therefore for everything astropy
does underneath), then executes the target script with `runpy` so `__file__` is
set exactly as under `python <script>`. Imported project modules are recovered
separately by sweeping `sys.modules` after the run, because the frozen importlib
does not always raise the `open` event. Raw traces:
`repro/inputs_c70.json`, `repro/inputs_c71.json`.

Excluded from the table below: the CPython interpreter's own stdlib and DLLs,
`site-packages`, the tracer itself, and two incidental
`*.dist-info/entry_points.txt` reads made by astropy's entry-point scan
(`cffi-2.0.0`, `markdown-3.9`).

| # | absolute path | bytes | SHA-256 | opens c70 | opens c71 |
|---:|---|---:|---|---:|---:|
| 1 | `C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\c70_endogeneity.py` | 9,015 | `875ab24bb4a4effd1c5e3389bfc4056a0bba5de33e88ff3feb52e7dc23af03de` | 4 | 0 |
| 2 | `C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\c71_amplitude.py` | 5,633 | `024b08966adf7af6d0fc0dc8f5aed793cd83253b85f0de735fc5ba22a75e36e8` | 0 | 4 |
| 3 | `C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\g2\Fig-3_Lensing-rotation-curves_Massbin-1.txt` | 1,760 | `cd8171d248a5c660701c2fcfb5f39eea01ae57b5b9ec2bae233e5aef77e7d78e` | 1 | 1 |
| 4 | `C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\g2\Fig-3_Lensing-rotation-curves_Massbin-2.txt` | 1,762 | `279d82e4faee34041221b617f0ce9cfc97966c431616c94608b0983e60421ae7` | 1 | 1 |
| 5 | `C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\g2\Fig-3_Lensing-rotation-curves_Massbin-3.txt` | 1,760 | `88eca49e85504c1eb6ce11e09edcdda37c0903dd678f2207ed4bca4fa31f7a22` | 1 | 1 |
| 6 | `C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\g2\Fig-3_Lensing-rotation-curves_Massbin-4.txt` | 1,765 | `05853565ae193347adff22f8aec58c50f80b2e22866fb9c532137c6f198a79e1` | 1 | 1 |
| 7 | `C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\invariant_bench.py` | 20,139 | `fe817b228d6d495c15d4bdaa84144ede30c638bb9a39ea1124700e7a309dde0f` | 1 | 1 |
| 8 | `C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\wl\wicker.tsv` | 23,715 | `e6c64de603a20bed635076c8c0bcb862357936848a2c262de731ed061a78730c` | 1 | 1 |
| 9 | `C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\xcop_T\XCOP_thermo.tex` | 127,657 | `ef37d3b5708c5e648da5e1c402ccd66441f979a42993dd2b4f9cc1eba1ca8838` | 1 | 1 |
| 10 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\configs\sparc_rotation_curves_full_v1.json` | 247,315 | `dde80c7fc72974358b1370e1978726b87fe1a4048f0880ae79cf513e260a7cf1` | 1 | 1 |
| 11 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\g4\cluster-lensing-exploration-v7-source\fig2.tsv` | 6,976 | `6ae2cc0a75e2113f2af73c054d5283099b8764b534d17c89b65d8562d799d58a` | 1 | 1 |
| 12 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A1644\A1644_density_L1.fits` | 14,400 | `281a71858531733c974fea521791ef81a0f26bba918dabc50439f288fd494fcc` | 2 | 2 |
| 13 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A1644\A1644_temperature.fits` | 14,400 | `a752bffdc36d3e9ee34c29a93e1316ce1fdd3d4f9fac1eecf69bce70f5594cb9` | 4 | 4 |
| 14 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A1795\A1795_density_L1.fits` | 14,400 | `e45a8f7bb1c28b3aaa0645684c8890d2471c717ed3454fed57c04fbfd5534e15` | 2 | 2 |
| 15 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A1795\A1795_mstar.fits` | 40,320 | `18afc10ce2ba22630c3d2ad9a2c721e2734148e45da59532511109670e3acf27` | 4 | 4 |
| 16 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A1795\A1795_temperature.fits` | 14,400 | `508c379ddeb3eccae030c6c04143bd6d1b129d2b75bba80039f5e7ab1f8d0dc7` | 4 | 4 |
| 17 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2029\A2029_density_L1.fits` | 14,400 | `473594d299944c0f818a37f47f9810f50090de49cbc0e163cab26120f21fb84b` | 2 | 2 |
| 18 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2029\A2029_mstar.fits` | 40,320 | `f7bdabd23395254ad30615d6302cc24e2ea957982a14df03d1b271718645038d` | 4 | 4 |
| 19 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2029\A2029_temperature.fits` | 14,400 | `4264ab2c3f0ebe8484a1096c285968ef3c24867dd5b0d4f360839d14d69c0a9c` | 4 | 4 |
| 20 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2142\A2142_density_L1.fits` | 14,400 | `f87274934034842c268f4da5e10ae7fd89310b8ea99a234b69dcb6b5b603f370` | 2 | 2 |
| 21 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2142\A2142_mstar.fits` | 46,080 | `c46bb6a972707a1af52d0c5bfcee01efda409766dbb2a1536f6a68820af71bc1` | 4 | 4 |
| 22 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2142\A2142_temperature.fits` | 14,400 | `5968d8661a9805f4ab17a17193b06092a8d0bc4222dded0c59241208b0e7ae43` | 4 | 4 |
| 23 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2255\A2255_density_L1.fits` | 14,400 | `996b814c12e7591e6ca7c1a1fb13627c66e1a2e8e41faafdfc77571be504b20b` | 2 | 2 |
| 24 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2255\A2255_temperature.fits` | 14,400 | `fa8305d24587eaaa55a3c9e35f98eaf033a74b88a5aace55bdbaefd56d37507b` | 4 | 4 |
| 25 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2319\A2319_density_L1.fits` | 14,400 | `2d65480271b22801392c1fbf528bfc523e6f4ee1880f884db5846b54a43858b7` | 2 | 2 |
| 26 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2319\A2319_mstar.fits` | 31,680 | `840f67549fb3806f423be363ff62c11549954ab93f363fc011e642630184c8b8` | 4 | 4 |
| 27 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A2319\A2319_temperature.fits` | 14,400 | `7bae4e64c49d387a4da78f7114714bfb04c1969bf2b3cc1c87fab9be0cb90a46` | 4 | 4 |
| 28 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A3158\A3158_density_L1.fits` | 14,400 | `6af54c4e6d357cf1261435d97ae5ad88ef23eaa5ddd2498549298b8a7ccc2f66` | 2 | 2 |
| 29 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A3158\A3158_temperature.fits` | 14,400 | `1f1d6b6a6d3276bc8168e16305b99a3233c686766171b6b0fbbafaf6380d7111` | 4 | 4 |
| 30 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A3266\A3266_density_L1.fits` | 14,400 | `49ae59dee4c2615dec83a81ad0a5897acbbdcd939deb4f17efd3fb6ecd4b431c` | 2 | 2 |
| 31 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A3266\A3266_temperature.fits` | 14,400 | `2d3e422f5d44a77d9da8a873286a03da79d672a2dd9abf1fdf803723acd03dc0` | 4 | 4 |
| 32 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A644\A644_density_L1.fits` | 14,400 | `767e51177201b19a7d0b90404bd480dad3769867ba13c6b01c437225aa8580d1` | 2 | 2 |
| 33 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A644\A644_mstar.fits` | 40,320 | `68b7fb85bde0d73a970541ae87f556791a1d8fe0ddb4ac69ae87ff86c8c30920` | 4 | 4 |
| 34 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A644\A644_temperature.fits` | 14,400 | `f69cd4496f0d8019cacc78623c72b26c44b55dc5f57575188cf5c27235ec7d11` | 4 | 4 |
| 35 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A85\A85_density_L1.fits` | 14,400 | `ee0ec85ca23bb06cf0c54304004523e84002c3ba26530a4f76b29bf26bece085` | 2 | 2 |
| 36 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A85\A85_mstar.fits` | 34,560 | `e6992926deb67b4b8bc863f0585d31a5df74030e149abdff39f7398fe7e7e2be` | 4 | 4 |
| 37 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\A85\A85_temperature.fits` | 14,400 | `06e54855f293ce77fad850526f97213795736c2c543c9ce9d64c05129bbe3f1a` | 4 | 4 |
| 38 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\RXC1825\RXC1825_density_L1.fits` | 14,400 | `44689c8630529c872ac45bd80e62f660f62284e485040ae9520b7b7c995c453f` | 2 | 2 |
| 39 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\RXC1825\RXC1825_temperature.fits` | 14,400 | `3e660ed3c3bd7db5d6bd39e1bd218694da33e42e15af18d3daad610b2aa0feb1` | 4 | 4 |
| 40 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\ZW1215\ZW1215_density_L1.fits` | 14,400 | `6f5bfffe166c4bfc547ec25a442c4113a66dd76d28faa136f1bc37ffa7b74458` | 2 | 2 |
| 41 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\ZW1215\ZW1215_mstar.fits` | 40,320 | `977bb510c54beb244b0a282d64fa711da930f30d03ff1d0dbb8b3b5cd28df605` | 4 | 4 |
| 42 | `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\runs\gravity\roadmap\item-59-xcop-forward-observable-gate-v1-source\raw\ZW1215\ZW1215_temperature.fits` | 14,400 | `23227e14464f6c7ac53ad3bbaacdb0211c9e344b402d384f6d6b264339f13cbf` | 4 | 4 |

**Notes a reviewer needs:**

1. **Rows 3–6, 8, 10, 11 are read but do not enter either statistic.**
   `invariant_bench.Bench.__init__` eagerly loads every probe it can find —
   SPARC, X-COP, CLASH, KiDS, Wicker, Solar, wide binaries. Both scripts use
   only `b.d["xcop"]`. The SPARC JSON, the KiDS mass-bin text files, the Wicker
   TSV and the CLASH `fig2.tsv` are opened at import and then ignored.

2. **Each `*_temperature.fits` is read twice, for two different purposes, with
   two different temperature normalisations.**
   - Inside `invariant_bench._cluster_profile`, `T_X` is scaled by
     `kT500 = G·M500·mu·m_p/(2·R500)` using the **FITS header** `M500`/`R500`,
     and used to build `g_obs` through the hydrostatic equation.
   - Inside `c70`/`c71`, `T_X` is scaled by
     `T500 = 8.85·(M500/1e15)^(2/3)·E(z)^(2/3)` using **Ghirardini Table 1**
     `M500` and `z` parsed out of `xcop_T/XCOP_thermo.tex`, and used to build
     the model prediction.
   These are not the same number. The temperature therefore enters the observed
   quantity and the predicted quantity through two separate scalings.

3. **Five of the twelve clusters have no `*_mstar.fits`.** For those,
   `invariant_bench._cluster_profile` falls back to `M_star = 0.10 · M_gas`.
   Present: A1795, A85, A644, ZW1215, A2319, A2029, A2142. Absent (fallback
   used): A1644, RXC1825, A3158, A2255, A3266. This affects `g_bar`, hence the
   observed excess, for those five clusters only. Full listing:
   `repro/tie_and_clamp_audit.txt`.

4. `XCOP_thermo.tex` is the source of the cluster names, redshifts, `M500` and
   `R500`. Only the first 4,000 characters after the string
   `"Basic properties of the X-COP sample"` are parsed.

---

## 5. Environment, seeds, permutation counts, sidedness

### Environment

| item | value |
|---|---|
| Python | 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] |
| executable | `C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe` |
| numpy | 2.2.6 |
| scipy | 1.16.1 (installed; **neither script imports it**) |
| astropy | 7.1.1 |
| platform | Windows-11-10.0.26200-SP0 |

Only `numpy` and `astropy.io.fits` are imported by the two scripts (plus stdlib
`json`, `math`, `os`, `re`). scipy is not used anywhere in either test — the
Spearman coefficient is hand-rolled, not `scipy.stats.spearmanr`. This matters:
see item 8.

### Random seeds

| script | line | statement | consumed by |
|---|---:|---|---|
| `c70_endogeneity.py` | 56 | `rng = np.random.default_rng(4242)` | the shuffled-temperature null only |
| `c71_amplitude.py` | 29 | `rng = np.random.default_rng(99)` | the derangement null only |

Both are `numpy.random.Generator` (PCG64). No other stochastic element exists in
either script: the point estimates are deterministic functions of the input
files. There is no global `np.random.seed`, no `random` module use, and no
environment-dependent hashing in either path.

For reference, the seeds of the upstream scripts that produced `kappa`:
`p01_rigorous.py` line 41 `default_rng(20260902)`; `p02_systematics.py` line 25
`default_rng(7)` (drawn but never used — p02 is deterministic);
`m12_redo_correct.py` line 32 `default_rng(101)`.

### Permutation counts and sidedness

| statistic | script/lines | draws | null construction | p-value | sided |
|---|---|---:|---|---|---|
| median within-cluster &rho; | c70 199–215 | **2000** | `rng.permutation(12)` — a **uniform random permutation, fixed points allowed** (it is *not* a derangement); on average ~1 cluster per draw keeps its own temperature | `np.mean(null >= obs)` = 0.3565 | **one-sided, upper tail** |
| within-cluster scatter *W* | c71 92–104 | **3000** | rejection-sampled **derangement** (`while np.any(p == arange(12))`) | `np.mean(W <= true_w)` = 0.7083 | **one-sided, lower tail** |
| per-cluster median-ratio scatter *S* | c71 92–104 | **3000** | same derangements as *W* (the same `p` produces both) | `np.mean(M <= true_m)` = 0.0643 | **one-sided, lower tail** |
| d ln(excess)/d ln(kT) slope | c70 149–169 | — | none | **no p-value is computed**; only the &sigma;-distances 2.4 and 0.6 | n/a (a two-sided *z*-distance in spirit) |

Additional facts about these p-values:

* **No `+1` correction.** All three use `mean(indicator)`, not
  `(1 + count)/(1 + n)`. A p-value of exactly 0 is attainable and would be
  reported as `0.0000`.
* **The two nulls are not the same null.** c70 re-maps the donor temperature
  profile onto an *index* grid (`np.linspace(0,1,n)` on both sides), so the
  donor profile is stretched to the recipient's bin count regardless of radius.
  c71 re-maps it onto the *radius* grid (`np.interp(c["frac"], other["frac"],
  other["kT"])`), i.e. matched in r/R500. The two nulls answer different
  questions and are not comparable draw-for-draw.
* **c70's null admits fixed points.** With 12 clusters, the probability a given
  draw is a derangement is ~0.368, so ~63% of the 2000 draws leave at least one
  cluster paired with its own temperature. The null is therefore slightly
  contaminated toward the observed value. c71's is a strict derangement.
* Resolution floor: 1/2000 = 5.0e-4 for c70, 1/3000 = 3.3e-4 for c71.

---

## 6. Per-(cluster, radius) table — `repro/points.csv`

**778 rows** (every radial bin in the twelve X-COP density profiles),
**588 of which** pass the mask and are the points the two tests actually use.
One row per (cluster, radius). Columns:

| column | meaning |
|---|---|
| `cluster` | cluster name as assigned by the scripts' extent→name map |
| `identity_how` | `exact` (R500 matched Table 1 to <3 kpc) or `elimination` (A644, A2319) |
| `radius_kpc` | bin centre, `0.5·(R_IN+R_OUT)` from `*_density_L1.fits`, in kpc |
| `r_over_R500_tab1` | `radius_kpc / R500`, with R500 from **Ghirardini Table 1** |
| `excess_obs` | `(g_obs/g_bar) / nu_RAR(g_bar/a0)` — the observed excess |
| `kT_keV` | `interp(r/R500, RW_X, T_X) · T500`, the temperature driving the prediction |
| `pred_true_pairing` | `sqrt(1 + kappa·3·kT/(mu·m_p·c^2))` with kappa = 1.36e5 |
| `pred_shuffled_c70` | same, with kT taken from the donor of the **first** c70 permutation |
| `pred_shuffled_c71` | same, with kT taken from the donor of the **first** c71 derangement |
| `sigma_excess_used_by_script` | **empty in every row — see the warning below** |
| `eT_X_scaled_keV` | `interp(r/R500, RW_X, eT_X) · T500`, the spectroscopic 1&sigma; on kT |
| `sigma_pred_from_eT` | that error propagated onto the prediction, `(kappa·3·k/(mu m_p c^2))/(2·pred) · eT` |
| `g_obs`, `g_bar` | the two accelerations, m/s^2 |
| `mask_bench_radius_and_sign` | 1 if `(r>120 kpc) & (r<1650 kpc) & (g_obs>0) & (g_bar>0)` |
| `mask_c71_ratio_finite_positive` | 1 if `isfinite(excess/pred) & (excess/pred > 0)` — c71's inner mask |
| `used_in_c70_c71` | 1 if the point reaches either statistic (identical to the bench mask) |

**Warning, and it is the answer to the reviewer's question about per-point
uncertainty: neither `c70_endogeneity.py` nor `c71_amplitude.py` defines,
reads, or propagates any uncertainty on the observed excess.** There is no
error bar on `excess_obs` anywhere in either test. `eT_X` is present in the
FITS files and is read by `p01_rigorous.py`, but c70 and c71 never open that
column. The two columns `eT_X_scaled_keV` and `sigma_pred_from_eT` were
computed *for this packet* from data the scripts had in hand but discarded;
they are labelled as such and are not part of either test. Consequently:

* the slope's `±0.205` is a **residual-based** OLS standard error, not an
  error-propagated one, and assumes homoscedastic Gaussian residuals in
  ln(excess) at fixed ln(kT);
* the two amplitude statistics in c71 are unweighted;
* the permutation p-values do not depend on any uncertainty model — which is a
  genuine strength of them, and worth saying.

**The mask.** The `(120, 1650) kpc` radius cut and the positivity cuts are
applied inside `invariant_bench._xcop`, i.e. *before* the data reach either
test script. Masked points never appear in the tests' arrays at all. They are
included in `points.csv` with `mask_bench_radius_and_sign = 0` so a reviewer can
see what was dropped: **190 of 778 points (24.4%)** — 136 at r &le; 120 kpc and
54 at r &ge; 1650 kpc. Four of those 190 also carry g_obs &le; 0; **no point is
excluded by the positivity cut alone**, so the mask is purely a radius cut in
practice.

**Second, undocumented mask — `np.interp` clamping.** `np.interp` returns the
endpoint value outside the tabulated range; it does not extrapolate and does not
return NaN. The X-COP spectroscopic temperature tables stop between 0.79 and
1.10 R500, while the density profiles reach 1.09–1.52 R500. **94 of the 588
used points (16.0%) lie beyond the last temperature bin and are assigned a
constant, clamped temperature** — for those points the "prediction" carries no
radial information at all. Per cluster this ranges from 3.7% (A2319) to 35.7%
(A1644). Full table: `repro/tie_and_clamp_audit.txt`. This is documented, not
fixed.

---

## 7. Complete permutation distributions — every draw

Not summaries. Every draw of every reported permutation statistic, together
with the permutation vector that produced it, so a reviewer can re-derive the
p-values or re-run any individual draw.

| file | rows | contents |
|---|---:|---|
| `repro/perm_c70_null_median_rho.csv` | 2000 | `draw`, `median_within_cluster_rho`, then `perm_0 … perm_11` |
| `repro/perm_c70_null_median_rho.npy` | 2000 | the statistic column as float64 |
| `repro/perm_c71_within.csv` | 3000 | `draw`, `within_cluster_scatter_dex`, then `perm_0 … perm_11` |
| `repro/perm_c71_within.npy` | 3000 | the statistic column as float64 |
| `repro/perm_c71_median.csv` | 3000 | `draw`, `median_ratio_scatter_dex`, then `perm_0 … perm_11` |
| `repro/perm_c71_median.npy` | 3000 | the statistic column as float64 |

`perm_k` is the index of the cluster whose temperature profile was donated to
cluster *k*; the cluster order is `['A1644', 'RXC1825', 'A3158', 'A1795', 'A2255', 'A85', 'A644', 'ZW1215', 'A2319', 'A2029', 'A2142', 'A3266']`.
The `perm_*` columns in `perm_c71_within.csv` and `perm_c71_median.csv` are the
same draws (c71 computes both statistics from one derangement).

These were regenerated by `repro/build_packet_data.py`, which re-executes the
null loops verbatim with the scripts' own seeds. They reproduce the published
p-values exactly:

```
c70 null: n=2000  observed=0.642557  frac>=obs=0.356500      (script printed 0.3565)
c71: true_w=0.168939 true_m=0.068251
     p_w=0.708333 p_m=0.064333                               (script printed 0.7083 / 0.0643)
```

---

## 8. Exact mathematical definitions, read off the code

Notation: clusters *c* = 1…12; within cluster *c*, radial points *i* = 1…n_c
(n_c between 38 and 63, total 588). Constants exactly as in the source:
c = 2.99792458e8 m/s, 1 keV = 1.602176634e-16 J, m_p = 1.67262192369e-27 kg,
mu = 0.6, a0 = 1.2e-10 m/s², kappa = 1.36e5.

**Observed excess** (c70 line 122, c71 line 61)

&nbsp;&nbsp;&nbsp; `nu_RAR(x) = 1 / (1 - exp(-sqrt(max(x, 1e-300))))`, `x = g_bar/a0`

&nbsp;&nbsp;&nbsp; **E_{c,i} = (g_obs,{c,i} / g_bar,{c,i}) / nu_RAR(g_bar,{c,i} / a0)**

**Model prediction** (c70 line 180, c71 lines 69–70)

&nbsp;&nbsp;&nbsp; **P_{c,i}(kT) = sqrt( 1 + kappa · 3·kT_{c,i}·[J/keV] / (mu · m_p · c²) )**

with **kT_{c,i} = interp(r_{c,i}/R500_c ; RW_X_c, T_X_c) · T500_c** and
**T500_c = 8.85 · (M500_c/10^15 M_sun)^(2/3) · E(z_c)^(2/3)**,
**E(z) = sqrt(0.3(1+z)³ + 0.7)**. `interp` is `np.interp`, which **clamps**
outside `[min RW_X, max RW_X]`.

### 8.1 The rank statistic

`spearman(a,b)` (c70 lines 67–75) is **not** `scipy.stats.spearmanr`. It is:

1. drop indices where either value is non-finite;
2. **r_a = argsort(argsort(a))** — the *ordinal* rank 0…n−1. **Ties are broken
   by array position, not by the average rank.** `invariant_bench._rank`
   documents this exact construction as a defect ("a global constant scored
   corr = +0.948 with the dataset label, an entirely manufactured number") and
   ships a tie-correct replacement — which `c70` does not use;
3. centre both rank vectors;
4. **rho(a,b) = (r_a · r_b) / sqrt((r_a·r_a)(r_b·r_b))**, or NaN if the
   denominator is 0.

Per cluster (c70 line 181): **rho_c = spearman(E_c, P_c)**.
Reported statistic (c70 line 192): **R_obs = median_c rho_c = +0.643**.

Because the arrays are pre-sorted by radius and 94 points carry a clamped
(hence tied) prediction, the tied predictions receive monotonically increasing
ranks *in radius order*. Measured effect of the tie handling
(`repro/tie_and_clamp_audit.txt`): per-cluster |Δrho| ≤ 0.034 (worst RXC1825,
+0.803 → +0.837 with average ranks), and the median across clusters is **+0.643
either way**. The defect is real and immaterial to the headline number.

### 8.2 "Per-cluster median ratio"

The phrase denotes the same quantity in both scripts, but each script then does
something different with it.

&nbsp;&nbsp;&nbsp; **ratio_{c,i} = E_{c,i} / P_{c,i}**
&nbsp;&nbsp;&nbsp; **m_c = median_i ratio_{c,i}**   (c70 line 182; c71 line 82, restricted to `good`)

* c70 line 195 reports **median_c m_c = 1.032**, called the "normalisation ratio".
* c70 line 196 reports **std_c(m_c) = 0.165** — a **linear**, population
  (`ddof=0`) standard deviation of the ratios themselves. *This number was
  computed and printed but does not appear in the write-up* (see item 10).
* c71 line 83 reports **S = std_c( log10 m_c ) = 0.0683 dex** — a **base-10
  logarithmic**, population (`ddof=0`) standard deviation. This is the
  "scatter of the per-cluster median ratio".

`std_c(m_c) = 0.165` and `std_c(log10 m_c) = 0.068 dex` are two different
statistics of the same twelve numbers and must not be confused.

c71's `good` mask (line 79) is `isfinite(ratio) & (ratio > 0)`, and a cluster is
dropped entirely if `good.sum() < 8` (line 80). Under the true pairing all 588
points are `good` and all twelve clusters survive (min n_c = 38), so for the
true pairing the two definitions coincide exactly.

### 8.3 "Within-cluster scatter"

c71 line 81:

&nbsp;&nbsp;&nbsp; **W_c = std_i( log10 ratio_{c,i} )**, `np.std`, population,
**ddof = 0**, over `good` points only

&nbsp;&nbsp;&nbsp; **W = median_c W_c = 0.1689 dex**

Note it is the *median* over clusters of a *standard deviation* within each
cluster — a robust summary of a non-robust dispersion.

### 8.4 The slope

c70 lines 149–154. OLS with intercept of ln(median excess) on ln(median kT),
one point per cluster (n = 12), via `np.linalg.lstsq`:

&nbsp;&nbsp;&nbsp; **ln E_c^med = beta · ln kT_c^med + alpha**, where
E_c^med = median_i E_{c,i} and kT_c^med = median_i kT_{c,i}

&nbsp;&nbsp;&nbsp; **se(beta) = sqrt( ( Sum_c res_c² / (n−2) ) / Sum_c (ln kT_c^med − mean)² )**

Comparison targets (c70 lines 155–169), with
**x = kappa·3·(5 keV)/(mu m_p c²) = 3.6161**:

&nbsp;&nbsp;&nbsp; algebraic null: **1.000**;
pressure model: **(1/2)·x/(1+x) = 0.3919**

&nbsp;&nbsp;&nbsp; **d_alg = |beta − 1|/se = 2.4**, **d_prs = |beta − 0.3919|/se = 0.6**

These are *distances in units of the OLS standard error*, not p-values. No
p-value is computed for the slope. The comparison value `0.3919` is evaluated
at a fixed 5 keV, not at each cluster's own temperature.

### 8.5 The three permutation p-values

Let Pi be the set of maps {1…12}→{1…12} drawn by each script (c70: uniform
permutations, fixed points allowed; c71: uniform derangements by rejection).

**c70, one-sided upper tail, B = 2000** (lines 199–215):

&nbsp;&nbsp;&nbsp; donor temperature, **index**-matched:
**kt^{(b)}_{c,i} = interp( linspace(0,1,n_c)_i ; linspace(0,1,n_{pi_b(c)}), kT_{pi_b(c)} )**

&nbsp;&nbsp;&nbsp; **R_b = median_c spearman( E_c, P(kt^{(b)}_c) )**

&nbsp;&nbsp;&nbsp; **p = (1/B) · #{ b : R_b >= R_obs } = 0.3565**

**c71, one-sided lower tail, B = 3000** (lines 92–107):

&nbsp;&nbsp;&nbsp; donor temperature, **radius**-matched:
**kt^{(b)}_{c,i} = interp( r_{c,i}/R500_c ; r_{pi_b(c)}/R500_{pi_b(c)}, kT_{pi_b(c)} )**

&nbsp;&nbsp;&nbsp; **W_b = median_c std_i( log10( E_{c,i} / P(kt^{(b)}_{c,i}) ) )**,
**S_b = std_c( log10 median_i( E_{c,i} / P(kt^{(b)}_{c,i}) ) )**

&nbsp;&nbsp;&nbsp; **p_W = (1/B)·#{ b : W_b <= W } = 0.7083**,
**p_S = (1/B)·#{ b : S_b <= S } = 0.0643**

No continuity correction in any of the three.

---

## 9. Leave-one-cluster-out — every statistic, twelve drops

Recomputed from scratch with each cluster removed in turn. Row 1 is the full
sample for reference; rows 2–13 are the twelve drops. Machine-readable:
`repro/loco.csv` (same columns, full precision).

Because the permutation loops are re-seeded from the scripts' own seeds
(`4242`, `99`) at each drop, and the RNG stream depends on the sample size, the
LOCO p-values are *fresh* Monte-Carlo estimates, not sub-samples of the
full-sample draws. Monte-Carlo error on a p near 0.35 with B = 2000 is ~0.011;
near 0.06 with B = 3000 it is ~0.004.

Column key: `slope`, `se` = the c70 log-log slope and its standard error;
`σ_alg`, `σ_prs` = distances to 1.000 and 0.3919; `ρ(kT,exc)` = cluster-level
Spearman; `med ρ_c` = median within-cluster rank correlation; `#ρ>0` = count
positive; `med ratio` = median per-cluster median ratio; `p_null` = c70's
one-sided permutation p; `W`, `p_W` = c71 within-cluster scatter and its p;
`S`, `p_S` = c71 median-ratio scatter and its p.

| dropped | n | slope | se | &sigma;<sub>alg</sub> | &sigma;<sub>prs</sub> | &rho;(kT,exc) | med &rho;<sub>c</sub> | #&rho;>0 | med ratio | p<sub>null</sub> | W | p<sub>W</sub> | S | p<sub>S</sub> |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (none: full sample) | 12 | 0.517 | 0.205 | 2.358 | 0.608 | 0.615 | 0.643 | 12 | 1.032 | 0.3565 | 0.1689 | 0.7083 | 0.0683 | 0.0643 |
| A1644 | 11 | 0.418 | 0.284 | 2.046 | 0.093 | 0.545 | 0.692 | 11 | 1.036 | 0.2900 | 0.1682 | 0.8910 | 0.0688 | 0.1510 |
| RXC1825 | 11 | 0.568 | 0.223 | 1.943 | 0.789 | 0.609 | 0.593 | 11 | 1.028 | 0.7680 | 0.1682 | 0.7347 | 0.0711 | 0.0763 |
| A3158 | 11 | 0.522 | 0.223 | 2.138 | 0.583 | 0.609 | 0.593 | 11 | 1.036 | 0.8275 | 0.1682 | 0.5107 | 0.0712 | 0.0947 |
| A1795 | 11 | 0.540 | 0.209 | 2.203 | 0.708 | 0.700 | 0.593 | 11 | 1.028 | 0.7955 | 0.1697 | 0.3420 | 0.0684 | 0.0410 |
| A2255 | 11 | 0.499 | 0.161 | 3.111 | 0.662 | 0.691 | 0.692 | 11 | 1.036 | 0.1275 | 0.1697 | 0.6073 | 0.0555 | 0.0207 |
| A85 | 11 | 0.521 | 0.214 | 2.239 | 0.605 | 0.627 | 0.692 | 11 | 1.028 | 0.1955 | 0.1697 | 0.5073 | 0.0708 | 0.0730 |
| A644 | 11 | 0.423 | 0.179 | 3.219 | 0.174 | 0.564 | 0.593 | 11 | 1.028 | 0.7195 | 0.1682 | 0.7820 | 0.0540 | 0.0337 |
| ZW1215 | 11 | 0.513 | 0.215 | 2.263 | 0.560 | 0.582 | 0.593 | 11 | 1.028 | 0.7985 | 0.1697 | 0.5550 | 0.0709 | 0.0813 |
| A2319 | 11 | 0.579 | 0.239 | 1.761 | 0.784 | 0.545 | 0.593 | 11 | 1.036 | 0.7595 | 0.1682 | 0.7833 | 0.0713 | 0.0890 |
| A2029 | 11 | 0.516 | 0.226 | 2.143 | 0.547 | 0.564 | 0.692 | 11 | 1.028 | 0.1990 | 0.1697 | 0.4497 | 0.0713 | 0.1080 |
| A2142 | 11 | 0.535 | 0.226 | 2.057 | 0.631 | 0.609 | 0.692 | 11 | 1.036 | 0.1940 | 0.1697 | 0.7357 | 0.0712 | 0.0793 |
| A3266 | 11 | 0.539 | 0.206 | 2.245 | 0.713 | 0.682 | 0.692 | 11 | 1.036 | 0.1450 | 0.1682 | 0.8157 | 0.0682 | 0.0560 |

**What the table shows, factually:**

* **The slope never approaches the algebraic value.** Across all twelve drops
  it runs 0.418–0.579, always ≥1.76σ from 1.000 and always ≤0.79σ from 0.3919.
  The narrowest margin against the algebraic null is on dropping A2319 (1.76σ).
* **`med ρ_c` takes only three values** (0.593, 0.643, 0.692). With eleven
  clusters the median is a single order statistic, so it hops between adjacent
  per-cluster values rather than moving smoothly. Do not read it as a
  continuous sensitivity.
* **c70's permutation p-value is wildly unstable: 0.128 to 0.828.** Dropping
  A2255 gives 0.128; dropping A3158 gives 0.828. It never crosses 0.05, so the
  qualitative conclusion ("the null reaches the observed value often") is
  robust, but the *number* 0.36 carries no stability at n = 12.
* **c71's `p_S` crosses 0.05 in three of twelve drops** — A2255 (0.0207),
  A644 (0.0337), A1795 (0.0410) — against 0.0643 for the full sample. Dropping
  A1644 pushes it the other way, to 0.1510. `p_W` never crosses (range
  0.342–0.891), so the script's printed verdict branch
  (`if pw > 0.05 and pm > 0.05`) would not have flipped in any drop, because it
  requires both. But the median-ratio statistic alone is one cluster away from
  "significant" in either direction.
* A644 and A2319, the two identities assigned by elimination, are individually
  droppable without changing any qualitative conclusion.

---

## 10. Inventory of every summary statistic computed

### 10a. Search of the edit history — method and result

The scratchpad is not in git, so the only edit history that exists is the Claude
Code session transcript. `repro/scan_history.py` walked **161 transcript files**
under `C:\Users\henry\.claude\projects\C--Users-henry-dev\` (main sessions
and all subagent transcripts) and extracted every tool call touching either
script. `repro/window_commands.py` then listed every tool call in the working
window 2026-09-03T22:10Z–22:30Z. Findings:

* **`c70_endogeneity.py`: exactly one `Write`**, 2026-09-03T22:13:09Z, 223
  lines. Recovered content is byte-identical to the file on disk (modulo CRLF).
  No subsequent edit. Saved at `repro/history/00_Write_c70_endogeneity.py.txt`.
* **`c71_amplitude.py`: exactly one creation**, 2026-09-03T22:14:18Z, via a
  quoted bash heredoc (`cat > c71_amplitude.py <<'PYEOF'`), 123 lines.
  Recovered content is byte-identical to the file on disk. No subsequent edit.
  Saved at `repro/history/c71_amplitude.py.asWritten.txt`.
* **Each script was executed exactly once** before the write-up was edited
  (22:15:25 and 22:16:00), after which the session moved to unrelated work.
* Consequently **there is no earlier revision of either script**, and no
  statistic was computed and then deleted from the code. The three `Read` hits
  in the scan are from this packet's own session.

**Stated plainly: the search was done, across 161 transcript files, and found
no statistic that was calculated and then removed from either script.** What it
*did* find is a set of statistics that the scripts compute and print but that
never reached the write-up. Those are listed below.

### 10b. Everything `c70_endogeneity.py` computes

| # | quantity | value | printed | in write-up |
|---:|---|---|:---:|:---:|
| 1 | rho(kT, g_obs) across clusters | +0.769 | yes | **NO** |
| 2 | rho(kT, g_bar) across clusters | +0.811 | yes | yes |
| 3 | rho(kT, excess) across clusters | +0.615 | yes | yes |
| 4 | log-log slope beta | +0.517 | yes | yes |
| 5 | se(beta) | 0.205 | yes | yes |
| 6 | intercept alpha | (not printed) | **no** | no |
| 7 | residual vector res | (not printed) | **no** | no |
| 8 | x at 5 keV | 3.62 | yes | yes |
| 9 | pressure-model slope | +0.392 | yes | yes |
| 10 | d_alg | 2.4σ | yes | yes |
| 11 | d_prs | 0.6σ | yes | yes |
| 12 | per-cluster n | 38–63 | yes (12 rows) | **NO** |
| 13 | per-cluster kT min–max | 2.5–9.9 keV | yes (12 rows) | **NO** |
| 14 | per-cluster observed-excess min–max | 0.15–9.22 | yes (12 rows) | **NO** |
| 15 | per-cluster predicted-excess min–max | 1.68–2.86 | yes (12 rows) | **NO** |
| 16 | per-cluster rho_c | +0.252 … +0.827 | yes (12 rows) | **NO** |
| 17 | per-cluster median ratio m_c | 0.737 … 1.437 | yes (12 rows) | **NO** |
| 18 | median_c rho_c | +0.643 | yes | yes |
| 19 | count rho_c > 0 | 12 of 12 | yes | yes |
| 20 | median_c m_c | 1.032 | yes | yes |
| 21 | **std_c(m_c), linear, ddof=0** | **0.165** | yes | **NO** |
| 22 | median of the 2000-draw null | +0.625 | yes | yes |
| 23 | null 2.5/97.5 percentiles | [+0.525, +0.738] | yes | yes |
| 24 | fraction of null ≥ observed | 0.3565 | yes | yes (as 0.36) |
| 25 | `elim` flag per cluster (A644, A2319) | True/False | **no** | no |

### 10c. Everything `c71_amplitude.py` computes

| # | quantity | value | printed | in write-up |
|---:|---|---|:---:|:---:|
| 26 | true within-cluster scatter W | 0.1689 dex | yes | yes |
| 27 | true median-ratio scatter S | 0.0683 dex | yes | yes |
| 28 | median of the 3000 null W_b | 0.1670 dex | yes | yes |
| 29 | **null W 2.5/97.5 percentiles** | **[0.1592, 0.1728]** | yes | **NO** |
| 30 | median of the 3000 null S_b | 0.0846 dex | yes | yes |
| 31 | **null S 2.5/97.5 percentiles** | **[0.0635, 0.1013]** | yes | **NO** |
| 32 | p_W | 0.7083 | yes | yes (as 0.71) |
| 33 | p_S | 0.0643 | yes | yes (as 0.064) |
| 34 | per-cluster W_c (12 values) | — | **no** | no |
| 35 | per-cluster m_c under each pairing | — | **no** | no |

### 10d. The four items worth a reviewer's attention

1. **#21, `std_c(m_c) = 0.165`.** Computed, printed on the console, dropped
   before the write-up. It is the cluster-to-cluster scatter of the
   normalisation, i.e. the direct measure of how well a single kappa transfers.
   The write-up reports the *centre* of that distribution (1.032) without its
   *width* (±0.165, a 16% spread on a quantity whose expected value is 1).
2. **#1, `rho(kT, g_obs) = +0.769`.** Computed and printed as one of the two
   halves of the endogeneity diagnostic. The write-up quotes only the g_bar
   half (+0.811). Both were needed to make the endogeneity argument as the
   script's own docstring frames it.
3. **#29 and #31, the two null confidence intervals.** The write-up gives the
   null medians but not their spread. With the intervals in hand a reader can
   locate the observed values inside their nulls: the observed W (0.1689) sits
   between the null's median (0.1670) and its 97.5th percentile (0.1728), and
   the observed S (0.0683) sits between the null's 2.5th percentile (0.0635)
   and its median (0.0846). Without them the reader cannot.
4. **#25, the `elim` flag.** `c70` computes `elim = nm in ("A644","A2319")`,
   stores it in the `rows` tuples, and **never reads it again**. No
   exact-identity-only subsample statistic is computed anywhere in c70 or c71,
   although the earlier `p01_rigorous.py` does compute one for its own test
   (n = 10, rho = +0.442, p = 0.205). The scaffolding for that check exists in
   c70 and was not used. Item 9 above supplies the closest available
   substitute.

---

## 11. Injection-recovery power analysis — DEFERRED

**Not in this packet, by instruction.** An injection-recovery power analysis for
these two tests is being run by a separate agent elsewhere and its results are
not incorporated here. Nothing in items 1–10, 12 or 13 depends on it, and no
number in this packet should be read as a power estimate.

The nearest thing this packet contains is the leave-one-cluster-out table
(item 9), which measures *stability*, not power. `p01_rigorous.py` section 7
contains an unrelated Fisher-z power calculation for the cluster-level Spearman
test (power 0.51 at rho = 0.6, n = 12); that is a different statistic from
either of the two tested here and is quoted only to note it exists.

---

## 12. Provenance of `kappa = 1.36e5`

### 12a. The trace

Both test scripts hard-code the value and neither derives it:

* `c70_endogeneity.py` line 54: `KAPPA = 1.36e5`
* `c71_amplitude.py` line 28: `OM, OL, KAPPA = 0.3, 0.7, 1.36e5`

Neither script reads any file that contains it. The value was typed in.

**It first appears as a computed number in `p02_systematics.py`.** That script
is deterministic (its `default_rng(7)` is never consumed) and reproduces on
demand. Its section 3 output:

```
   sigma_int = 0.15 :  kappa = 1.36e+05
                     68% [1.19e+05, 1.54e+05]
                     95% [1.03e+05, 1.73e+05]
                     kappa=1e5 is delta-chi2 +4.83 (excluded at 95%)
```

That is the exact source of both `1.36e5` and `[1.19, 1.54]e5`. Mechanically:

1. `p01_rigorous.py` builds the twelve X-COP clusters, computes for each a
   median excess `exc`, a bootstrap error `eexc` (4000 resamples over that
   cluster's radial bins), a median temperature `kT` and its error `ekT`, and
   writes them to `paper_results.json`.
2. `p01_rigorous.py` section 4 fits the **one-parameter** model
   `excess = sqrt(1 + kappa·3kT/(mu m_p c²))` to those twelve points by
   chi-squared minimisation over a log grid `10**linspace(3,7,4001)`, obtaining
   **kappa = 1.563e5, 68% [1.479e5, 1.648e5]**, chi²/dof = 3.69. These values
   are in `paper_results.json` as `kappa`, `kappa_lo`, `kappa_hi`.
3. `p02_systematics.py` reads `paper_results.json`, adds an intrinsic-scatter
   term `(sigma_int · exc)²` in quadrature, and re-minimises the **same**
   chi-squared over a log grid `10**linspace(3.5,6.5,3001)`. At the
   sigma_int = 0.15 that makes chi²/dof = 0.93, the minimum lands at
   **kappa = 1.36e5**, and the Δchi² ≤ 1 range on that grid is
   **[1.19e5, 1.54e5]**. That is the interval.

### 12b. Was it fitted to the same twelve X-COP excesses used in the radial test?

**Yes. Verified numerically, not inferred.** The twelve excess values in
`paper_results.json` that the kappa fit minimised against are identical, to the
last digit resolvable in the CSV, to the twelve cluster-median excesses that
`c70`/`c71` compute from the bench:

```
cluster      p01 excess       c70/c71 median excess      |difference|
A1644       1.633446240            1.633446240             1.9e-10
RXC1825     2.275317206            2.275317206             3.9e-10
A3158       2.122892224            2.122892224             9.9e-11
A1795       2.570611143            2.570611143             1.3e-11
A2255       1.595421090            1.595421090             3.8e-10
A85         2.479274923            2.479274923             1.6e-10
A644        3.522540827            3.522540827             2.9e-11
ZW1215      2.527173830            2.527173830             2.2e-10
A2319       2.594705943            2.594705943             1.7e-10
A2029       2.649326290            2.649326291             1.0e-10
A2142       2.543846861            2.543846860             1.4e-10
A3266       2.046216098            2.046216098             2.7e-10
```

(residuals are CSV round-trip only). Same clusters, same definition of excess,
same pipeline. The kappa used in the "zero free parameters" radial test was
fitted to these twelve numbers.

### 12c. Where the earlier `kappa ~ 1e5` came from, and how it differs

`m08_pressure_source.py`, run before any per-cluster temperature existed. Two
distinct numbers appear there:

* **Section 2, an analytic one-liner** (lines 96–102). The cluster excess was
  taken as a single scalar, `need = 2.29` (the mean over the xcop/wicker/clash
  probes of the median RAR miss). In the deep-MOND regime the boost goes as the
  square root of the source, so the source must rise by `need² = 5.244`. With
  `3P/rho c² = 2.664e-5` at an assumed 5 keV,
  **kappa = (need² − 1)/2.664e-5 = 1.593e5**.
* **Section 3, a coarse grid sweep** (line 132) over
  `(1, 1e3, 1e4, 3e4, 1e5, 1.6e5, 3e5, 1e6, 3e6)`, scoring each on median
  |log10| RAR error for the three cluster probes. The winner was
  **kappa = 1e5** (cluster error 0.1556 dex vs 0.3605 baseline). That grid
  point is the "kappa = 1e5" quoted downstream as pre-specified — see
  `m10_xcop_temperature.py` line 184 ("kappa = 1e5 was fixed by the
  galaxy-cluster comparison. No freedom.") and `p01_rigorous.py` line 19
  ("Amplitude pre-specified: kappa = 1e5").

**How they differ.** `1e5` is a single scalar calibration from aggregate
galaxy-vs-cluster medians, chosen from a nine-point grid, computed before any
X-COP temperature was in hand — genuinely prior to these data. `1.36e5` is a
maximum-likelihood point estimate of the same parameter refitted to the twelve
X-COP cluster medians, with an assumed 15% intrinsic scatter chosen so that
chi²/dof ≈ 1. They differ by a factor of 1.36, and 1e5 lies just inside the
95% interval of the refit ([1.03, 1.73]e5) and outside its 68% interval
([1.19, 1.54]e5). `p02_systematics.py` itself withdraws p01's stronger claim:
"the pre-registered kappa = 1e5 is no longer excluded."

### 12d. Is "zero free parameters" misleading? — the flat answer

**Partly yes, and the packet can be precise about which part.**

**Yes, as a description of the value used.** `kappa = 1.36e5` is not a prior
value. It is the fitted maximum of a one-parameter likelihood over the *same
twelve cluster excesses* the radial test then uses, with an intrinsic-scatter
nuisance chosen to make the fit acceptable. Describing a test that uses it as
having "zero free parameters" without saying so is misleading. The write-up
compounds this by saying in the same document that "kappa ≈ 10⁵ came from the
galaxy-to-cluster comparison, not from these data" while the number actually
plugged into c70/c71 is 1.36e5, which did come from these data.

**No, for the specific statistic that carries the headline result — and this is
the sharper point.** `repro/kappa_sensitivity.py` recomputes every reported
statistic at kappa = 1e4, 5e4, 1e5, 1.19e5, 1.36e5, 1.54e5, 1.56e5, 3e5, 1e6:

```
       kappa   c70 med rho  n rho>0  c70 med ratio  c70 p_null  c71 true_w   c71 p_w  c71 true_m   c71 p_m
----------------------------------------------------------------------------------------------------------
       1e+04     +0.642557       12       2.159528      0.3565    0.178661    0.6653    0.083023    0.0877
       5e+04     +0.642557       12       1.496492      0.3565    0.172350    0.6830    0.072832    0.0760
       1e+05     +0.642557       12       1.170732      0.3565    0.169847    0.6970    0.069843    0.0733
    1.19e+05     +0.642557       12       1.091177      0.3565    0.169316    0.7077    0.068929    0.0697
    1.36e+05     +0.642557       12       1.032064      0.3565    0.168939    0.7083    0.068251    0.0643
    1.54e+05     +0.642557       12       0.979037      0.3565    0.168612    0.7200    0.067676    0.0600
    1.56e+05     +0.642557       12       0.973646      0.3565    0.168580    0.7200    0.067632    0.0600
       3e+05     +0.642557       12       0.727595      0.3565    0.166140    0.6150    0.065954    0.0510
       1e+06     +0.642557       12       0.410461      0.3565    0.165021    0.6393    0.064635    0.0473
----------------------------------------------------------------------------------------------------------

Note: Spearman rho is invariant under any strictly monotone transform of
one variable, and pred(kT) is strictly increasing in kT for every kappa>0,
so the c70 Test-2 rank column and its null are identically kappa-free.
```

The c70 Test-2 rank correlation (+0.642557), the 12-of-12 positive count, and
its permutation p (0.3565) are **bit-identical at every kappa across two orders
of magnitude.** The reason is exact, not numerical: `P(kT)` is a strictly
increasing function of kT for any kappa > 0, and Spearman's rho is invariant
under strictly monotone transformations. **The radial rank test contains no
information about kappa whatsoever.** Calling it "zero free parameters" is
correct for that statistic in the strongest possible sense — but by the same
token it is not a test of the pressure model's amplitude at all. It tests only
whether the excess increases monotonically with the temperature profile.

**What kappa does control** is exactly the numbers the write-up presents as
supporting evidence:

* the normalisation ratio, 1.032 at kappa = 1.36e5 — but **1.171 at 1e5, 0.974
  at 1.56e5, 2.160 at 1e4, 0.410 at 1e6**. Its closeness to 1 is a restatement
  of the fit, not an independent check;
* c71's W (0.1689) and S (0.0683) and their p-values, all of which drift with
  kappa (p_S: 0.088 at 1e4 → 0.047 at 1e6).

**Summary for the reviewer's specific suspicion:** the suspicion is justified
about the *label*. The radial test's headline rank statistic is genuinely
parameter-free, but it is parameter-free because it is blind to the parameter,
not because the parameter was fixed in advance. The one number in that test
that does depend on kappa — the normalisation ratio of 1.032 — is close to
unity because kappa was fitted to these twelve clusters to make it so.

---

## 13. Clean rerun from a fresh process, line by line

Both scripts were re-executed in fresh processes from a clean shell, with
`PYTHONIOENCODING=utf-8`, with no other change to the machine. Verbatim stdout
is preserved at `repro/rerun_c70_stdout.txt` (56 lines) and
`repro/rerun_c71_stdout.txt` (28 lines); stderr was empty for both; exit status
0 for both.

### 13a. Verbatim stdout, `python c70_endogeneity.py`

```

==============================================================================
TEST 1  --  is the temperature correlation algebraic?
==============================================================================
   First, where the shared dependence would enter:

   rho(kT, g_obs)   = +0.769   <- g_HSE carries T directly
   rho(kT, g_bar)   = +0.811   <- g_bar carries n_e, not T
   rho(kT, excess)  = +0.615

   Now the slope, which the two explanations disagree about:

      d ln(excess) / d ln(kT)  measured   = +0.517 +- 0.205

      pure algebraic channel   predicts   = +1.000
      pressure model, kappa = 1.36e+05    = +0.392
        (x = kappa*3kT/(mu m_p c^2) = 3.62 at 5 keV)

      distance from algebraic  : 2.4 sigma
      distance from pressure   : 0.6 sigma

==============================================================================
TEST 2  --  the radial profile, with kappa already fixed
==============================================================================
   excess(r) = sqrt( 1 + kappa * 3kT(r)/(mu m_p c^2) ),  no free parameters.
   Within each cluster T varies with radius, so the model predicts how the
   excess RUNS, not just its average. Tested per cluster below.

   cluster      n    kT range keV   exc obs range   exc pred range   rho(r)  med ratio
   -----------------------------------------------------------------------------------
   A1644       56         2.9-4.7       0.22-5.44        1.76-2.10   +0.518      0.892
   RXC1825     49         4.0-6.0       0.84-4.89        1.98-2.31   +0.803      1.065
   A3158       53         3.8-5.1       1.05-4.67        1.94-2.17   +0.763      1.000
   A1795       38         2.7-5.5       1.06-4.33        1.73-2.23   +0.692      1.185
   A2255       49         4.2-6.1       0.90-4.23        2.01-2.32   +0.568      0.737
   A85         46         4.3-6.1       1.00-3.56        2.04-2.33   +0.593      1.085
   A644        47         2.5-7.3       0.61-4.61        1.68-2.51   +0.753      1.437
   ZW1215      49         5.0-6.3       1.61-3.85        2.15-2.35   +0.827      1.079
   A2319       54         4.8-9.9       0.41-6.27        2.12-2.86   +0.775      1.028
   A2029       38         5.0-8.0       1.18-4.48        2.16-2.60   +0.252      1.036
   A2142       46         4.7-8.0       0.59-3.67        2.09-2.61   +0.261      0.999
   A3266       63         4.3-7.5       0.15-9.22        2.03-2.54   +0.592      0.878
   -----------------------------------------------------------------------------------
   median within-cluster rho(observed, predicted) : +0.643
   clusters with rho > 0                          : 12 of 12
   normalisation ratio, median across clusters    : 1.032
   scatter of that ratio                          : 0.165

==============================================================================
A cheaper null: does the same test pass on a shuffled temperature map?
==============================================================================
   observed median within-cluster rho : +0.643
   shuffled-temperature null          : +0.625 [+0.525, +0.738]
   fraction of null >= observed       : 0.3565

   wrote c70_endogeneity.json
```

### 13b. Verbatim stdout, `python c71_amplitude.py`

```
==============================================================================
Amplitude test: does the model track THIS cluster's temperature?
==============================================================================

   TRUE pairing
      median within-cluster scatter of log10(ratio) : 0.1689 dex
      scatter of the per-cluster median ratio       : 0.0683 dex

   SHUFFLED pairing, 3000 derangements
      within-cluster scatter : 0.1670 [0.1592, 0.1728]
      median-ratio scatter   : 0.0846 [0.0635, 0.1013]

   fraction of shuffles at least as TIGHT as the true pairing
      on within-cluster scatter : 0.7083
      on median-ratio scatter   : 0.0643

==============================================================================
VERDICT
==============================================================================
   The true temperature pairing is NOT tighter than random pairings on
   either amplitude statistic. Within the precision of these twelve clusters
   the model is not tracking each cluster's own temperature -- it is matching
   a generic declining profile that every cluster shares.

   The zero-free-parameter radial prediction therefore does NOT support the
   pressure model. It does not refute it either: the test has little power,
   because cluster temperature profiles are too similar to each other for a
   twelve-object sample to tell them apart.
```

### 13c. Line-by-line comparison against the reported numbers

Reported values are as they stand in the write-up section "6b. Two tests the
critique demanded" (`scratchpad/lessons.html`).

| # | quantity | reported | rerun | discrepancy |
|---:|---|---|---|---|
| 1 | rho(kT, g_bar) | +0.811 | +0.811 | none |
| 2 | d ln(excess)/d ln(kT) | +0.517 ± 0.205 | +0.517 ± 0.205 | none |
| 3 | algebraic-channel prediction | +1.000 | +1.000 | none |
| 4 | distance from algebraic | 2.4σ | 2.4σ | none |
| 5 | pressure-model prediction at kappa = 1.36e5 | +0.392 | +0.392 | none |
| 6 | distance from pressure | 0.6σ | 0.6σ | none |
| 7 | 3kT/(mu m_p c²) at 5 keV | 2.66e−5 | 2.664e−5 (m08) | none (rounding) |
| 8 | kappa · that | 3.62 | 3.62 | none |
| 9 | median within-cluster rho | +0.643 | +0.643 | none |
| 10 | clusters with rho > 0 | 12 of 12 | 12 of 12 | none |
| 11 | normalisation ratio | 1.032 | 1.032 | none |
| 12 | shuffled-temperature null, median | +0.625 | +0.625 | none |
| 13 | shuffled null, 95% range | [+0.525, +0.738] | [+0.525, +0.738] | none |
| 14 | fraction of shuffles ≥ observed | 0.36 | 0.3565 | none (rounding) |
| 15 | within-cluster scatter, true | 0.1689 | 0.1689 | none |
| 16 | within-cluster scatter, shuffled | 0.1670 | 0.1670 | none |
| 17 | p on within-cluster scatter | 0.71 | 0.7083 | none (rounding) |
| 18 | per-cluster median ratio, true | 0.0683 | 0.0683 | none |
| 19 | per-cluster median ratio, shuffled | 0.0846 | 0.0846 | none |
| 20 | p on median-ratio scatter | 0.064 | 0.0643 | none (rounding) |
| 21 | Spearman(kT, excess), n = 12 | +0.615 | +0.615 | none |
| 22 | fitted kappa | 1.36e5 [1.19, 1.54] | 1.36e5 [1.19e5, 1.54e5] (p02 rerun) | none |
| 23 | chi²/dof, statistical errors only | 3.69 | 3.69 (p02 rerun) | none |
| 24 | chi²/dof with 15% intrinsic scatter | 0.93 | 0.93 (p02 rerun) | none |
| 25 | Δchi² vs constant excess | +6.9 | +6.85 (p02 rerun) | none (rounding) |
| 26 | scatter of the normalisation ratio | **not reported** | **0.165** | **omission — see item 10** |
| 27 | rho(kT, g_obs) | **not reported** | **+0.769** | **omission — see item 10** |
| 28 | null W 95% range | **not reported** | **[0.1592, 0.1728]** | **omission — see item 10** |
| 29 | null S 95% range | **not reported** | **[0.0635, 0.1013]** | **omission — see item 10** |

Both result JSON files were regenerated by the rerun and are **byte-identical**
to the snapshots taken before any re-execution:

```
c70_endogeneity.json  f77a0f685dd3b3b27f6da86d1fae3b49d9f10307e618b8ee22e648373821820a   (unchanged)
c71.json              2bf5b0fe242ea05c1e1d8f63984008a31008175e3b5e7ce8ec1a22c6532a2f0e   (unchanged)
```

**No reported number failed to reproduce. Not one, at any printed digit.** Both
scripts are fully deterministic given their seeds and inputs. The four rows
marked "omission" are not failures to reproduce; they are quantities the code
computes that the write-up does not carry.

### 13d. Additional verifications performed

* **Cluster identity map — verified correct, 12 of 12.** The scripts assign
  names by matching each bench `extent` value to a Table 1 `R500` within 3 kpc,
  falling back to nearest-neighbour elimination for the two that fail. The
  bench's `extent` is in fact the FITS header `R500` of the source directory, so
  the true name is knowable without inference. Cross-checked in
  `repro/identity_audit.txt`: **all twelve assignments, including the two by
  elimination (A644 at extent 1250, A2319 at extent 1368), are correct.** Note
  that for those two the header R500 (1250, 1368 kpc) differs from the Table 1
  R500 (1230, 1346 kpc) by 20–22 kpc, which is why they missed the 3 kpc window.
* **Tie-handling in the hand-rolled Spearman — real but immaterial here.**
  Recomputed with average ranks: per-cluster |Δrho| ≤ 0.034, median across
  clusters unchanged at +0.643. Table in `repro/tie_and_clamp_audit.txt`.
* **`np.interp` clamping — 94 of 588 used points (16.0%)** lie beyond the last
  tabulated temperature bin and receive a constant temperature. Documented, not
  fixed.

---

## Files in this packet

All in `C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad\repro\`.

| file | SHA-256 |
|---|---|
| `PACKET.md` | `(this file)` |
| `points.csv` | `5c025f2c68e25588e9ca57639efe0a7abbbea2e55552a6a6adf1927a2285c3ba` |
| `perm_c70_null_median_rho.csv` | `4a986a255c1c0b2961a9a35e69bc6d98a4ea87d24277818ce1ff054b8690b2bb` |
| `perm_c70_null_median_rho.npy` | `e4bbe170ca0a723503d0fff8475b5aa6e95faae56521a47c2952e5e06e003a60` |
| `perm_c71_within.csv` | `a80397fb3092644c7efad9ba7043d683243b2597c508caeb4d88a401a24fbf79` |
| `perm_c71_within.npy` | `5372c077085024516e3de387b24abf9a5091ae3d6e623686576e0ce2fbd10988` |
| `perm_c71_median.csv` | `1c8a6e3e03e06a9cba3be41777f590cb22445b3b5c661ce5ac4f18a5d0bffd12` |
| `perm_c71_median.npy` | `29b8edfc5a43613d02742d291529abe32c810f7ad47b8e036a06a1b6e47b2ab6` |
| `loco.csv` | `14cdc1f8d770267e583a551162055f69b047dcb784641467ed91c95ba2e36590` |
| `packet_meta.json` | `70d1c3953f6ef57f448d2d4f1e5ea08286af34aaf51ab0cd8865cf08aa6c757c` |
| `identity_audit.txt` | `a32e6f42921ac90f79ce71c6b81b546539b95197dabc06a1e0a99cea4543ef31` |
| `kappa_sensitivity.txt` | `4520b8aaba1fdc5782282d1f50d3c893f476fb1c7ef4fdf13f6f4c685263d563` |
| `tie_and_clamp_audit.txt` | `3957f0fd0d5150ca97474fca64db671f6e330e0378da5e1ca3d5adab5dbf55d4` |
| `inputs_c70.json` | `6ce7287dcaa2d239cd4ff459e2bf73c50ea80bafd0d0a6041e5111db3a22aa46` |
| `inputs_c71.json` | `4309513fcaa34794fb9246249a7580c7cf85539b3c2ed64888e91d29955bf12f` |
| `rerun_c70_stdout.txt` | `4ca04d68771118093a59b74cd63a21661882ac9751d14ab5f8cd40d649657080` |
| `rerun_c71_stdout.txt` | `3bf891ebb9a5dede6ab0a65b3d560cee6462cf12941bb05cad82acc9557f4d55` |
| `build_packet_data.py` | `01ed1084148111258699d920482c2b3173e351d2c95a079199134b6b16c95f41` |
| `trace_inputs.py` | `e4316062bb3b93f01a81de12a888a7c65ac9bbbec2c3bb348f6d0d7b8b6dd257` |
| `kappa_sensitivity.py` | `5bd1f65d8cf59d2d7154d37b61aa387d6b95c6ca8f0a4bbe1a935f43ab7bc2de` |
| `tie_and_clamp_audit.py` | `5006c9e28dee46be03e5c33847efe39b0d22ad41b29f9282b83ddd8cc77e6770` |
| `scan_history.py` | `bd076e5405781ca104270d1efd3c796877effbc589382902e1d0f88a7aa0c3af` |
| `make_packet.py` | `93868ca9b69b1ee1acff005016d0ca44cc882d1e6090248372adfb2e44ec0e17` |

Also present: `repro/_preexisting/` (pre-rerun snapshots of the two result
JSONs), `repro/history/` (recovered script revisions and the transcript command
window), `repro/_item4_table.md`, `repro/find_c71.py`,
`repro/extract_c71_heredoc.py`, `repro/window_commands.py`,
`repro/trace_c70.log`, `repro/trace_c71.log`,
`repro/rerun_c70_stderr.txt`, `repro/rerun_c71_stderr.txt` (both empty).
