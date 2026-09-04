"""
TEMPCLAMP -- shared ingest for the temperature-extrapolation audit.

WHAT THE BUG WAS
    invariant_bench._cluster_profile interpolated the COARSE X-COP temperature
    profile onto the FINE density grid with a bare np.interp -- no left=, no
    right= -- so kT was CLAMPED to the endpoint value outside the measured
    range.  The hydrostatic acceleration is

        g_obs = -(kT/(mu m_p)) * (dln n_e/dln r + dln kT/dln r) / r

    so past the last measured temperature bin  dln kT/dln r == 0 EXACTLY and
    the temperature-gradient term is silently deleted -- in the cluster
    outskirts, which is where this programme reads its radial trend and where
    the true temperature is falling.

WHAT THIS MODULE DOES
    Loads X-COP through the PATCHED bench under each extrapolation mode, keeps
    the mask of extrapolated points, and reproduces each recorded estimator so
    the impact of the bug can be measured rather than asserted.

SEALED DATA
    KiDS and the wide binaries are NEVER loaded here.  `Bench` is deliberately
    NOT instantiated -- Bench.__init__ calls _kids() and _widebin() -- we build
    an uninitialised instance with Bench.__new__ and call _cluster_profile /
    _xcop directly, so no sealed probe is ever touched.
"""
from __future__ import annotations

import math
import os
import re
import sys

import numpy as np
from astropy.io import fits

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
BENCH_DIR = ROOT + "work/gravity-wells-2026-09"
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

import invariant_bench as IB                                     # noqa: E402
from invariant_bench import (Bench, KPC, MSUN, G, MU, MP, A0,    # noqa: E402
                             TemperatureExtrapolationError)

XR = (ROOT + "runs/gravity/roadmap/"
      "item-59-xcop-forward-observable-gate-v1-source/raw/")
SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
THERMO_TEX = SCR + "xcop_T/XCOP_thermo.tex"

KEV = 1.602176634e-16
C_LIGHT = 2.99792458e8
OM, OL = 0.3, 0.7
N_XCOP = 12
N_POINTS_RECORDED = 588          # asserted: the recorded X-COP point count

MODES = ("clamp", "drop", "loglinear")


# --------------------------------------------------------------------------
#  sealed-probe guard
# --------------------------------------------------------------------------
def bench_shell():
    """An uninitialised Bench: has the methods, has loaded nothing.

    Bench() would call _kids() and _widebin(); both are sealed for this
    programme, so we never run __init__.
    """
    b = Bench.__new__(Bench)
    b.temp_extrapolation = "clamp"
    b.warn_extrapolation = False
    b.extrapolation_report = []
    b.d = {}
    return b


def assert_no_sealed_probe(b):
    for k in ("kids", "widebin"):
        assert k not in getattr(b, "d", {}), f"SEAL VIOLATION: {k} was loaded"


# --------------------------------------------------------------------------
#  the published X-COP table (the route p01 / c70 / c71 / power all take)
# --------------------------------------------------------------------------
def load_thermo_table():
    """Basic properties of the X-COP sample, from the published source .tex.

    Row/column assertions on every ingest, per the acquisition-traps memory.
    """
    tex = open(THERMO_TEX, encoding="utf-8").read()
    i = tex.index("Basic properties of the X-COP sample")
    TAB = {}
    for line in tex[i:i + 4000].split("\n"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 6 or line.strip().startswith("%"):
            continue
        nm = cells[0].strip()
        if not re.match(r"^[A-Za-z]+\d", nm):
            continue

        def pm(s):
            s = s.replace("$", "")
            m = re.search(r"([\d.]+)\s*\\pm\s*([\d.]+)", s)
            if m:
                return float(m.group(1)), float(m.group(2))
            m = re.search(r"([\d.]+)", s)
            return (float(m.group(1)), 0.0) if m else (float("nan"), 0.0)

        z = pm(cells[1])[0]
        M500, eM500 = pm(cells[3])
        R500, eR500 = pm(cells[4])
        Ez = math.sqrt(OM * (1 + z) ** 3 + OL)
        T500 = 8.85 * (M500 * 1e14 / 1e15) ** (2 / 3) * Ez ** (2 / 3)
        TAB[nm] = dict(z=z, M500=M500, eM500=eM500, R500=R500, eR500=eR500,
                       T500=T500, eT500=T500 * (2 / 3) * eM500 / M500)
    assert len(TAB) == N_XCOP, f"XCOP_thermo table: {len(TAB)} rows, expected {N_XCOP}"
    for k, v in TAB.items():
        assert np.isfinite(v["R500"]) and v["R500"] > 0, f"{k}: bad R500"
        assert np.isfinite(v["T500"]) and v["T500"] > 0, f"{k}: bad T500"
    return TAB


def name_map(ext_kpc, TAB):
    """The bench identifies clusters by their R500 'extent'.  Reproduce the
    exact exact-then-elimination match p01 uses."""
    uq = sorted(np.unique(ext_kpc))
    NAME = {}
    for v in uq:
        hit = [k for k, t in TAB.items() if abs(t["R500"] - v) < 3.0]
        if len(hit) == 1:
            NAME[v] = (hit[0], "exact")
    left_v = [v for v in uq if v not in NAME]
    left_n = [k for k in TAB if k not in [x[0] for x in NAME.values()]]
    for v in left_v:
        k = min(left_n, key=lambda q: abs(TAB[q]["R500"] - v))
        NAME[v] = (k, "elimination")
        left_n.remove(k)
    assert len(NAME) == len(uq)
    return NAME, uq


# --------------------------------------------------------------------------
#  X-COP under one extrapolation mode
# --------------------------------------------------------------------------
def nu_rar(x):
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-300))))


def load_xcop(mode="clamp"):
    """Returns (D-object, per-cluster list). Never touches a sealed probe."""
    b = bench_shell()
    b.temp_extrapolation = mode
    xc = b._xcop(temp_extrapolation=mode)
    assert_no_sealed_probe(b)
    TAB = load_thermo_table()
    ext = np.asarray(xc.extent, float) * np.ones(len(xc)) / KPC
    NAME, uq = name_map(ext, TAB)
    assert len(uq) == N_XCOP, f"{len(uq)} distinct R500 values, expected {N_XCOP}"

    CL = []
    for v in uq:
        nm, how = NAME[v]
        t = TAB[nm]
        m = np.abs(ext - v) < 1e-9
        o = np.argsort(xc.r[m])
        r_kpc = (xc.r[m] / KPC)[o]
        gbar = xc.gb[m][o]
        gobs = xc.go[m][o]
        exc = (xc.nu[m] / nu_rar(xc.x[m]))[o]
        exq = xc.extrapolated[m][o]
        stq = xc.stencil[m][o]
        with fits.open(os.path.join(XR, nm, f"{nm}_temperature.fits")) as h:
            dd = h[1].data
            rw = np.asarray(dd["RW_X"], float)
            tx = np.asarray(dd["T_X"], float)
            etx = np.asarray(dd["eT_X"], float)
        assert len(rw) == len(tx) == len(etx) == len(dd), "T table row mismatch"
        ok = np.isfinite(rw) & np.isfinite(tx) & (tx > 0)
        fr = r_kpc / t["R500"]
        # the per-cluster scaled-temperature route used by p01/c70/power --
        # ITSELF a bare np.interp, so it clamps too.  Reported separately.
        ts = np.interp(fr, rw[ok], tx[ok])
        ets = np.interp(fr, rw[ok], etx[ok])
        kT = ts * t["T500"]
        CL.append(dict(
            name=nm, how=how, R500=t["R500"], T500=t["T500"], M500=t["M500"],
            eM500=t["eM500"], eT500=t["eT500"], z=t["z"],
            r=r_kpc, frac=fr, gbar=gbar, gobs=gobs, exc=exc,
            kT=kT, tsc=ts, etsc=ets,
            extrap=exq, stencil=stq,
            rw_last=float(rw[ok].max()), rw_first=float(rw[ok].min()),
            n=int(m.sum())))
    tot = sum(c["n"] for c in CL)
    return xc, CL, tot


# --------------------------------------------------------------------------
#  estimators, each reproducing one recorded number
# --------------------------------------------------------------------------
def spearman(a, b_):
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b_)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb / math.sqrt((ra @ ra) * (rb @ rb)))


def perm_p(a, b_, n=20000, seed=0):
    """Two-sided permutation p for a Spearman rho. Its own FPR is calibrated
    in run_audit.py before any verdict is read off it."""
    rng = np.random.default_rng(seed)
    obs = abs(spearman(a, b_))
    a = np.asarray(a, float); b_ = np.asarray(b_, float)
    cnt = 0
    for _ in range(n):
        if abs(spearman(a, rng.permutation(b_))) >= obs - 1e-15:
            cnt += 1
    return (cnt + 1) / (n + 1)


def ols_slope(x, y):
    """slope, standard error -- log-log OLS, the c70 estimator."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    sl, ic = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - (sl * x + ic)
    se = float(np.sqrt(np.sum(res ** 2) / (len(x) - 2)
                       / np.sum((x - x.mean()) ** 2)))
    return float(sl), se


def radial_slope(CL, mask_key=None, fixed_effects=True):
    """d(log10 excess)/d(log10 r) -- the number Run AT quotes as -0.4803.

    fixed_effects: give every cluster its own level, i.e. the WITHIN-cluster
    slope, which is the estimand the programme actually reads.
    """
    xs, ys, gs = [], [], []
    for i, c in enumerate(CL):
        k = np.ones(len(c["r"]), bool) if mask_key is None else ~c[mask_key]
        if k.sum() < 3:
            continue
        xs.append(np.log10(c["r"][k]))
        ys.append(np.log10(c["exc"][k]))
        gs.append(np.full(int(k.sum()), i))
    x = np.concatenate(xs); y = np.concatenate(ys); g = np.concatenate(gs)
    if not fixed_effects:
        return ols_slope(x, y) + (len(x),)
    idx = np.unique(g)
    Adm = np.zeros((len(x), len(idx) + 1))
    Adm[:, 0] = x
    for j, gi in enumerate(idx):
        Adm[g == gi, j + 1] = 1.0
    beta, *_ = np.linalg.lstsq(Adm, y, rcond=None)
    res = y - Adm @ beta
    dof = len(x) - Adm.shape[1]
    xc_ = x.copy()
    for gi in idx:                       # within-transform for the SE
        xc_[g == gi] -= x[g == gi].mean()
    se = float(np.sqrt(np.sum(res ** 2) / dof / np.sum(xc_ ** 2)))
    return float(beta[0]), se, len(x)


def rar_slope_vs_frac(CL, mask_key=None):
    """same, but against r/R500 -- identical by the AT.3 identity, kept as a
    check that the identity still holds after the patch."""
    xs, ys, gs = [], [], []
    for i, c in enumerate(CL):
        k = np.ones(len(c["r"]), bool) if mask_key is None else ~c[mask_key]
        xs.append(np.log10(c["frac"][k])); ys.append(np.log10(c["exc"][k]))
        gs.append(np.full(int(k.sum()), i))
    x = np.concatenate(xs); y = np.concatenate(ys); g = np.concatenate(gs)
    idx = np.unique(g)
    Adm = np.zeros((len(x), len(idx) + 1)); Adm[:, 0] = x
    for j, gi in enumerate(idx):
        Adm[g == gi, j + 1] = 1.0
    beta, *_ = np.linalg.lstsq(Adm, y, rcond=None)
    return float(beta[0])
