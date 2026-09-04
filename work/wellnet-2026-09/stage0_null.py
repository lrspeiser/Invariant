"""STAGE 0: the least-favourable scalar null, with the whole search calibrated.

WHY THE PREVIOUS NULL WAS STILL FRIENDLY

Run AC repaired a degenerate null — one where the injected scalar truth lay
EXACTLY in the scalar basis, so admitting tensor atoms improved nothing, the
statistic was identically zero, and power was 1.00 by construction. The repair
was to inject between the bank's grid points (bank scales 0.3/1.0/3.0, injection
0.55/1.8).

That is still friendly. It asks only whether the tensor family can beat a
SLIGHTLY MISSPECIFIED MEMBER OF THE SAME GRAMMAR. The hypothesis that actually
has to be excluded is

    H0 : g = F(invariants) grad Phi_N     for ANY sufficiently smooth scalar F
    H1 : genuinely directional tensor terms

so the null must contain scalar responses that the bank cannot express at all,
not merely ones it expresses badly.

FIVE QUALITATIVELY DIFFERENT SCALAR FAMILIES

  1. a different interpolating-function family (simple/standard MOND nu forms,
     which are not in the bank's shape list)
  2. a rational Pade response with free poles
  3. a piecewise-linear spline with knots at random positions -- not smooth in
     the bank's sense at all
  4. a response built from a DIFFERENT invariant combination (rho and r rather
     than g), so the argument itself is outside the bank's span
  5. a GAUSSIAN-PROCESS draw in log x -- literally "any sufficiently smooth
     scalar response", drawn without reference to the grammar

Family 5 is the one that matters. If the detector survives a GP draw it has some
claim to being a test of directionality rather than of interpolation error.

REALISTIC FALSE-ANISOTROPY GENERATORS

A flexible tensor model can absorb any of these and call it anisotropy, so each
is applied to the SOURCE or the OBSERVABLE while the truth stays strictly scalar:
triaxial baryons at varying projection angle, deprojection error, a radial
mass-to-light gradient, miscentring, unresolved substructure, and a multiplicative
shear-calibration error.

CALIBRATING THE WHOLE SEARCH, NOT A CHOSEN STATISTIC

The test statistic is the improvement of the best tensor law found ANYWHERE in
the search, so every null realisation passes through the same pipeline the real
analysis would: atom generation, coefficient fitting, scale selection, subset
selection, AND the choice of preferred axis. Taking the maximum over the axis
grid is what carries the look-elsewhere effect of the search itself.

    D_max = max over axes, over subsets  [ RMS(scalar only) - RMS(all atoms) ]

THREE DISJOINT SIMULATION SETS

    calibration   sets the critical value D*
    audit         UNTOUCHED, verifies the realised false-positive rate at D*
    injection     measures power

A rate that is 5% "by construction" because the 95th percentile was taken on the
same simulations is not a measurement. The audit set is what makes it one.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_grammar import (GPU, KPC, MSUN, A0, FieldBank, div, nu_rar,
                           sphericity, tensor_basis, tidal_tensor, xp)  # noqa
from field_search import Search, observables, is_tensor                 # noqa
from qumond_degeneracy import independent_atoms                         # noqa

HERE = os.path.dirname(os.path.abspath(__file__))

#: candidate preferred axes the search is allowed to try. Taking the max over
#: these is what puts the look-elsewhere effect of axis choice into the null.
AXES = [(0, 0, 1), (1, 0, 0), (0, 1, 0),
        (1, 1, 0), (1, 0, 1), (1, 1, 1)]


# ------------------------------------------------------- the scalar H0 families
def scalar_truth(kind, inv, gmag, rng):
    """A strictly scalar response field W(x). No direction anywhere in it."""
    x = gmag / A0
    lx = xp.log10(xp.maximum(x, 1e-30))
    if kind == "nu_simple":
        # the simple MOND interpolating function -- a shape the bank lacks
        s = 10 ** rng.uniform(-0.7, 0.7)
        u = xp.maximum(x / s, 1e-12)
        w = xp.log10(0.5 + xp.sqrt(0.25 + 1.0 / u))
    elif kind == "pade":
        # a rational response with free poles
        a, b, c = rng.uniform(0.2, 3.0, 3)
        w = (1.0 + a * x) / (1.0 + b * x + c * x ** 2)
    elif kind == "spline":
        # piecewise linear in log x with random knots -- not smooth at all
        k = np.sort(rng.uniform(float(lx.min()), float(lx.max()), 5))
        v = rng.normal(0, 1, 5)
        w = xp.asarray(np.interp((lx.get() if GPU else lx), k, v))
    elif kind == "other_inv":
        # built from a DIFFERENT invariant combination than the bank's argument
        w = xp.log10(1.0 + inv["x_rho"] / 10 ** rng.uniform(-1, 1)) \
            * (1.0 + 0.5 * xp.tanh(inv["q_L"]))
    elif kind == "gp":
        # a Gaussian-process draw in log x: literally an arbitrary smooth scalar
        lo, hi = float(lx.min()), float(lx.max())
        g = np.linspace(lo, hi, 24)
        C = np.exp(-0.5 * (g[:, None] - g[None, :]) ** 2 / 0.8 ** 2) \
            + 1e-8 * np.eye(24)
        y = rng.multivariate_normal(np.zeros(24), C)
        w = xp.asarray(np.interp((lx.get() if GPU else lx), g, y))
    else:
        raise ValueError(kind)
    w = w - xp.mean(w)
    sd = float(xp.std(w))
    return w / sd if sd > 1e-12 else w


SCALAR_FAMILIES = ("nu_simple", "pade", "spline", "other_inv", "gp")


# ------------------------------------------------- false-anisotropy generators
def make_source(n, rng, generator=None, axis=(1.0, 0.78, 0.55)):
    """A source that may carry a realistic anisotropy-mimicking defect."""
    L = 400.0 * KPC
    h = L / n
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None, None] * xp.ones((1, n, n))
    Y = xp.ones((n, 1, n)) * ax[None, :, None]
    Z = xp.ones((n, n, 1)) * ax[None, None, :]
    a = list(axis)
    off = [0.0, 0.0, 0.0]
    tilt = 0.0
    extra = None
    mlgrad = 0.0
    if generator == "triaxial_tilt":
        tilt = rng.uniform(0.2, 1.2)
    elif generator == "deprojection":
        a[1] *= rng.uniform(0.75, 1.3)          # wrong assumed axis ratio
    elif generator == "miscentre":
        off = list(rng.normal(0, 25.0 * KPC, 3))
    elif generator == "ml_gradient":
        mlgrad = rng.uniform(-0.5, 0.5)
    elif generator == "substructure":
        extra = rng.normal(0, 90.0 * KPC, (6, 3))
    if tilt:
        c, s = math.cos(tilt), math.sin(tilt)
        X, Z = c * X - s * Z, s * X + c * Z
    Xs, Ys, Zs = X - off[0], Y - off[1], Z - off[2]
    q = xp.sqrt((Xs / a[0]) ** 2 + (Ys / a[1]) ** 2 + (Zs / a[2]) ** 2)
    rs = 80.0 * KPC
    rho = xp.exp(-(q / rs) ** 2)
    if mlgrad:
        rho = rho * (1.0 + mlgrad * xp.tanh(q / rs))     # radial M/L gradient
    if extra is not None:
        for p in extra:
            d2 = (X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2
            rho = rho + 0.10 * xp.exp(-d2 / (2 * (18.0 * KPC) ** 2))
    rho *= 1e14 * MSUN / float(rho.sum() * h ** 3)
    return rho, h


GENERATORS = (None, "triaxial_tilt", "deprojection", "miscentre",
              "ml_gradient", "substructure")


# --------------------------------------------------------------- the machinery
class Cfg:
    """One source configuration: bank, dedup, and per-axis designs."""

    def __init__(self, rho, h, n):
        self.bank = FieldBank(rho, h, dhat=(0, 0, 1), verbose=False)
        groups, _ = independent_atoms(self.bank)
        keep = [g[0] for g in groups]
        keep = [i for i in keep
                if self.bank.meta[i].split(" x ")[-1] != "gg"]
        self.meta = [self.bank.meta[i] for i in keep]
        y0, A, npt = observables(self.bank)
        self.y0, self.npt = y0, npt
        self.A = A[xp.asarray(np.array(keep, dtype=np.int64))]
        self.sd0 = float(xp.std(y0))
        sc = [i for i, m in enumerate(self.meta) if not is_tensor(m)]
        self.scal = sc
        self.S_all = Search(self.A)
        self.S_sca = Search(self.A[xp.asarray(np.array(sc, dtype=np.int64))])
        # the per-axis part: only the dd atoms depend on dhat, so rebuilding
        # the whole bank per axis is unnecessary
        self.axis_designs = []
        T, _, _ = tidal_tensor(self.bank.Phi_N, h)
        gx, gy, gz = self.bank.gvec
        nu = nu_rar(self.bank.gmag / A0)
        for dh in AXES:
            B = tensor_basis(self.bank.inv, self.bank.gvec, self.bank.gmag,
                             T, dh)
            rows = []
            for iname, ival in self.bank.inv.items():
                for s in (0.3, 1.0, 3.0):
                    w = xp.tanh(ival / s)
                    w = w - xp.mean(w)
                    sd = float(xp.std(w))
                    if sd < 1e-12:
                        continue
                    w = w / sd
                    c = tuple(w * ci for ci in B["dd"])
                    k11, k22, k33, k12, k13, k23 = c
                    vx = nu * (k11 * gx + k12 * gy + k13 * gz)
                    vy = nu * (k12 * gx + k22 * gy + k23 * gz)
                    vz = nu * (k13 * gx + k23 * gy + k33 * gz)
                    R = self.bank.P.solve(div(vx, vy, vz, h))
                    rows.append(_obs(self.bank, R))
            self.axis_designs.append(
                Search(xp.concatenate([self.A, xp.stack(rows)], axis=0))
                if rows else self.S_all)


def _obs(bank, Psi):
    n, h = bank.n, bank.h
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None] * xp.ones((1, n))
    Y = xp.ones((n, 1)) * ax[None, :]
    R = xp.sqrt(X ** 2 + Y ** 2)
    m = (R > 20 * KPC) & (R < 140 * KPC)
    a = xp.gradient(Psi, h, axis=0)[:, :, n // 2]
    b = xp.gradient(Psi, h, axis=1)[:, :, n // 2]
    v = ((X * a + Y * b) / (R + 1e-30))[m]
    S = Psi.sum(axis=2) * h
    d = xp.sqrt(xp.gradient(S, h, axis=0) ** 2
                + xp.gradient(S, h, axis=1) ** 2)[m]
    return xp.concatenate([v / (float(xp.std(v)) + 1e-300),
                           d / (float(xp.std(d)) + 1e-300)])


def response_of(cfg, W):
    """Turn a scalar response field W into its observable, through the solver."""
    b = cfg.bank
    gx, gy, gz = b.gvec
    nu = nu_rar(b.gmag / A0)
    R = b.P.solve(div(nu * W * gx, nu * W * gy, nu * W * gz, b.h))
    return _obs(b, R)


def D_max(cfg, y):
    """The statistic: best tensor improvement ANYWHERE in the search.

    Maximising over the axis grid as well as over subsets is what puts the
    look-elsewhere effect of the search into the null. Scoring only the axis a
    real analysis would have picked after seeing the data is exactly the
    mistake this is designed to prevent.
    """
    t = y - cfg.y0
    cfg.S_sca.set_target(t)
    r_sca, _ = cfg.S_sca.exhaustive_k(2)
    best = -np.inf
    for S in cfg.axis_designs:
        S.set_target(t)
        r, _ = S.exhaustive_k(2)
        best = max(best, r_sca - r)
    return best


def draw_null(cfg, rng, fam, noise, cal_err=0.0):
    """A strictly scalar truth, plus noise and an optional calibration error."""
    W = scalar_truth(fam, cfg.bank.inv, cfg.bank.gmag, rng)
    amp = rng.uniform(0.10, 0.40)
    o = response_of(cfg, W)
    y = cfg.y0 + amp * cfg.sd0 * o / (float(xp.std(o)) + 1e-30)
    if cal_err:
        y = y * (1.0 + rng.normal(0, cal_err))     # multiplicative shear cal
    return y + xp.asarray(rng.normal(0, noise * cfg.sd0, y.size))


def draw_signal(cfg, rng, noise, amp):
    """A genuinely directional truth, built on an axis NOT in the search grid."""
    b = cfg.bank
    T, _, _ = tidal_tensor(b.Phi_N, b.h)
    d = rng.normal(0, 1, 3)
    d = d / np.linalg.norm(d)
    B = tensor_basis(b.inv, b.gvec, b.gmag, T, tuple(d))
    gx, gy, gz = b.gvec
    nu = nu_rar(b.gmag / A0)
    y = cfg.y0.copy()
    for sgn, sc in ((1.0, 0.55), (-0.6, 1.8)):
        iname = list(b.inv)[rng.integers(len(b.inv))]
        w = xp.tanh(b.inv[iname] / sc)
        w = w - xp.mean(w)
        sd = float(xp.std(w))
        if sd < 1e-12:
            continue
        w = w / sd
        comp = B[["That", "dd"][rng.integers(2)]]
        c = tuple(w * ci for ci in comp)
        k11, k22, k33, k12, k13, k23 = c
        R = b.P.solve(div(nu * (k11 * gx + k12 * gy + k13 * gz),
                          nu * (k12 * gx + k22 * gy + k23 * gz),
                          nu * (k13 * gx + k23 * gy + k33 * gz), b.h))
        o = _obs(b, R)
        y = y + sgn * amp * cfg.sd0 * o / (float(xp.std(o)) + 1e-30)
    return y + xp.asarray(rng.normal(0, noise * cfg.sd0, y.size))


def main():
    print("=" * 78)
    print("STAGE 0 -- least-favourable scalar null, whole-search calibration")
    print("=" * 78)
    n = int(os.environ.get("NGRID", 40))
    NCAL = int(os.environ.get("NCAL", 60))
    NAUD = int(os.environ.get("NAUD", 60))
    NINJ = int(os.environ.get("NINJ", 30))
    NOISE = float(os.environ.get("NOISE", 0.10))
    rng = np.random.default_rng(20260904)
    t0 = time.time()

    print(f"\n   grid {n}^3, noise {NOISE:.0%}, axis grid {len(AXES)}")
    print(f"   calibration {NCAL} / audit {NAUD} / injection {NINJ}, DISJOINT")
    print(f"   scalar families: {', '.join(SCALAR_FAMILIES)}")
    print(f"   false-anisotropy generators: "
          f"{', '.join(str(g) for g in GENERATORS)}")

    cfgs = {}
    def cfg_for(gen, seed):
        key = (gen, seed)
        if key not in cfgs:
            rho, h = make_source(n, np.random.default_rng(seed), gen)
            cfgs[key] = Cfg(rho, h, n)
        return cfgs[key]

    def run(ntr, tag, inject=None):
        out = []
        for i in range(ntr):
            gen = GENERATORS[rng.integers(len(GENERATORS))]
            # a SMALL seed pool so banks are reused across realisations;
            # a wide pool rebuilt a bank per draw and the bank build is
            # the whole cost. Six generators x four seeds = 24 banks.
            cfg = cfg_for(gen, 1000 + int(rng.integers(4)))
            if inject is None:
                fam = SCALAR_FAMILIES[rng.integers(len(SCALAR_FAMILIES))]
                y = draw_null(cfg, rng, fam, NOISE,
                              cal_err=0.02 if rng.random() < 0.5 else 0.0)
                lab = fam
            else:
                y = draw_signal(cfg, rng, NOISE, inject)
                lab = f"amp{inject}"
            out.append((D_max(cfg, y), lab, str(gen)))
        return out

    print("\n   --- calibration set (scalar truth only) ---")
    cal = run(NCAL, "cal")
    Dv = np.array([d for d, _, _ in cal])
    Dstar = float(np.percentile(Dv, 95))
    print(f"      D median {np.median(Dv):.4e},  D* (95th) {Dstar:.4e}")
    byfam = {}
    for d, f, g in cal:
        byfam.setdefault(f, []).append(d)
    for f in SCALAR_FAMILIES:
        if f in byfam:
            print(f"      {f:10s} median D {np.median(byfam[f]):.3e}  "
                  f"max {np.max(byfam[f]):.3e}   n={len(byfam[f])}")

    print("\n   --- AUDIT set, untouched, verifies the realised rate ---")
    aud = run(NAUD, "audit")
    Da = np.array([d for d, _, _ in aud])
    fpr = float((Da > Dstar).mean())
    se = math.sqrt(max(fpr * (1 - fpr), 1e-9) / NAUD)
    print(f"      realised false-positive rate at D* : {fpr:.3f} +- {se:.3f}"
          f"   (nominal 0.05)")
    bygen = {}
    for d, f, g in aud:
        bygen.setdefault(g, []).append(d > Dstar)
    print("      by false-anisotropy generator:")
    for g, v in sorted(bygen.items()):
        print(f"         {g:16s} {np.mean(v):.2f}   n={len(v)}")

    print("\n   --- injection set, power ---")
    power = {}
    for amp in (0.15, 0.35, 0.60):
        inj = run(NINJ, "inj", inject=amp)
        p = float((np.array([d for d, _, _ in inj]) > Dstar).mean())
        power[str(amp)] = p
        print(f"      amplitude {amp:4.2f}   power = {p:.2f}")

    print(f"\n   elapsed {time.time()-t0:.0f}s, {len(cfgs)} source banks built")
    print("\n" + "=" * 78)
    print("   HOW TO READ THIS")
    print("   D* is set on the CALIBRATION set and verified on the UNTOUCHED")
    print("   AUDIT set. If the audit rate is near 0.05 the detector is")
    print("   correctly sized against arbitrary smooth scalar responses AND")
    print("   against realistic anisotropy-mimicking defects -- which is a much")
    print("   stronger claim than Run AC's, where the null was an off-grid")
    print("   member of the detector's own grammar.")
    print("   The per-generator column says WHICH defect the detector confuses")
    print("   for anisotropy. A rate well above 0.05 in one row names a")
    print("   systematic that must be controlled before any tensor claim.")

    out = {"grid": n, "noise": NOISE, "n_cal": NCAL, "n_audit": NAUD,
           "n_inj": NINJ, "axes": len(AXES),
           "scalar_families": list(SCALAR_FAMILIES),
           "generators": [str(g) for g in GENERATORS],
           "D_star_95": Dstar, "D_cal_median": float(np.median(Dv)),
           "audit_fpr": fpr, "audit_fpr_se": se,
           "fpr_by_generator": {g: float(np.mean(v)) for g, v in bygen.items()},
           "D_by_family": {f: [float(np.median(v)), float(np.max(v))]
                           for f, v in byfam.items()},
           "power": power}
    with open(os.path.join(HERE, "stage0_null.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n   written: stage0_null.json")


if __name__ == "__main__":
    main()
