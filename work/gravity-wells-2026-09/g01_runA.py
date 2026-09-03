"""
RUN A of the anisotropic-void test program: one-dimensional galaxy screening.

Implements the program's own equations on SPARC, verbatim, and reports the
outcome of each. Nothing here is a plan; every number below is computed.

Pipeline, exactly as specified in the program document:

    V_b^2 = (D/D0)[ V_gas|V_gas| + Ups_d V_disk^2 + Ups_b V_bulge^2 ]
    g_bar = V_b^2 / R,   g_obs = V_obs^2 / R

D0 = D so the distance factor is unity; SPARC's public mass-model table carries
no inclination or 3.6um luminosity column, so the D and i nuisances of the full
program cannot be sampled here and Ups is held at the programme's global value.
That is stated rather than hidden: this is the screening layer, not the fit.

The signed gas term is retained as the program requires -- tabulated V_gas can
be negative where the gas distribution pulls outward.
"""
import json
import math
import numpy as np

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
G = 6.674e-11
KPC = 3.0856775814913673e19
KMS = 1e3
MSUN = 1.98892e30
A0 = 1.2e-10
UPS_D, UPS_B = 0.5, 0.7
BAR = "=" * 78


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


# --------------------------------------------------------------- ingest
cfg = json.load(open(ROOT + "configs/sparc_rotation_curves_full_v1.json",
                     encoding="utf-8"))
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
    GAL.append(dict(name=g["name"], R=R, VO=VO, EV=EV, VB=VB,
                    g_bar=(VB * KMS) ** 2 / rm, g_obs=(VO * KMS) ** 2 / rm,
                    Mb=float((VB[-1] * KMS) ** 2 * rm[-1] / G / MSUN),
                    Vf=float(np.median(VO[-3:]))))
head("1. Ingest and transform")
print(f"   galaxies with >= 5 usable points : {len(GAL)}")
print(f"   total radial points              : {sum(len(x['R']) for x in GAL)}")
print(f"   Upsilon_disk = {UPS_D}, Upsilon_bulge = {UPS_B}  (global, not per galaxy)")
gb = np.concatenate([x["g_bar"] for x in GAL])
go = np.concatenate([x["g_obs"] for x in GAL])
Mb = np.array([x["Mb"] for x in GAL])
print(f"   g_bar range : {gb.min():.2e} - {gb.max():.2e} m/s^2")
print(f"   M_b range   : {Mb.min():.2e} - {Mb.max():.2e} Msun")

# --------------------------------------------------- candidate accelerations
def d1(gn):
    return gn


def d3(gn):
    """Piecewise cylindrical-confinement target."""
    return np.where(gn >= A0, gn, np.sqrt(A0 * gn))


def m1(gn):
    """mu(X)=X/(1+X) solved in spherical symmetry: g^2 -g_N g -g_N a0 = 0."""
    return 0.5 * (gn + np.sqrt(gn ** 2 + 4 * gn * A0))


def m2(gn):
    """mu(X)=X/sqrt(1+X^2): u^2 - g_N^2 u - g_N^2 a0^2 = 0 with u = g^2."""
    u = 0.5 * (gn ** 2 + np.sqrt(gn ** 4 + 4 * gn ** 2 * A0 ** 2))
    return np.sqrt(u)


def rar(gn):
    return gn / (1.0 - np.exp(-np.sqrt(gn / A0)))


def scalar_void(gn, alpha, n):
    """K1 = exp(-alpha q) I with q = q_g. Spherical: g = g_N exp(alpha q)."""
    q = 1.0 / (1.0 + (gn / A0) ** n)
    return gn * np.exp(alpha * q)


head("2. Screening the algebraic candidates")
print("   Score is RMS of log10(g_obs/g_pred) over all points, all galaxies.\n")


def score(pred):
    return float(np.sqrt(np.mean((np.log10(go) - np.log10(pred)) ** 2)))


ROWS = [
    ("D1  baryons only, Newton", d1(gb)),
    ("D3  piecewise sqrt(a0 g_N)", d3(gb)),
    ("M1  mu = X/(1+X)", m1(gb)),
    ("M2  mu = X/sqrt(1+X^2)", m2(gb)),
    ("RAR McGaugh 2016 (reference)", rar(gb)),
]
for a in (1.0, 2.0, 3.0):
    for n in (1.0, 2.0):
        ROWS.append((f"K1xQ2 scalar void  alpha={a:.0f} n={n:.0f}",
                     scalar_void(gb, a, n)))
print(f"   {'model':<38}{'RMS dex':>10}{'vs Newton':>11}")
print("   " + "-" * 59)
base = score(d1(gb))
for nm, p in ROWS:
    s = score(p)
    print(f"   {nm:<38}{s:>10.4f}{base/s:>10.2f}x")
print("   " + "-" * 59)

head("3. Why every scalar K is structurally dead")
print("""   The program's K1 is K = exp(-alpha q) I. In spherical symmetry the
   field equation integrates exactly:

       exp(-alpha q) g r^2 = G M      ->      g = g_N exp(alpha q)

   Every candidate q is BOUNDED in (0, 1] and tends to 1 as the source is
   left behind, because both rho_L and |g_N| tend to zero. So""")
for a in (1.0, 3.0, 5.0):
    far = np.array([1e-13])
    q = 1.0 / (1.0 + (far / A0) ** 2)
    print(f"      alpha={a:.0f}:  q(g_N=1e-13) = {float(q[0]):.6f},  "
          f"boost -> exp(alpha q) = {float(np.exp(a*q)[0]):.3f} (a constant)")
print("""
   g -> exp(alpha) g_N asymptotically, which is still an inverse-square law
   with a rescaled G. It can never produce a flat rotation curve, for ANY
   alpha and ANY bounded q.

   CONSEQUENCE: K1 x Q1..Q4 -- eight of the program's 24 models -- are
   eliminated analytically, before any fitting. Anisotropy or nonlinearity is
   REQUIRED, not optional. This matches the program's own statement that a
   constant K keeps an anisotropic inverse-square field.""")

head("4. The h_eff test -- the program's sharpest discriminator")
print("""   h_eff(R) = G M_b(<R) / (R g_R). With M_b(<R) = V_b^2 R / G and
   g_R = V_obs^2 / R this reduces to h_eff = R (V_b/V_obs)^2 = R g_bar/g_obs.

   The cylindrical-confinement argument REQUIRES h_eff ~ sqrt(G M_b / a0),
   i.e. a slope of exactly 1/2 against log M_b. If instead each galaxy needs
   its own h_eff, the model has renamed the halo scale.\n""")
HE, LM, NAMES = [], [], []
for x in GAL:
    flat = x["g_bar"] < A0                       # the low-acceleration part
    if flat.sum() < 3:
        continue
    h = x["R"][flat] * x["g_bar"][flat] / x["g_obs"][flat]
    w = 1.0 / np.maximum(x["EV"][flat], 1e-3) ** 2
    HE.append(float(np.sum(w * h) / np.sum(w)))
    LM.append(math.log10(x["Mb"])); NAMES.append(x["name"])
HE, LM = np.array(HE), np.array(LM)
A = np.vstack([LM, np.ones_like(LM)]).T
sh, c = np.linalg.lstsq(A, np.log10(HE), rcond=None)[0]
res = np.log10(HE) - (sh * LM + c)
n = len(HE)
se = float(np.sqrt(np.sum(res ** 2) / (n - 2) / np.sum((LM - LM.mean()) ** 2)))
print(f"   galaxies with >= 3 sub-a0 points : {n}")
print(f"   fitted slope s_h                 : {sh:+.4f} +- {se:.4f}")
print(f"   required by cylindrical argument : +0.5000")
print(f"   deviation                        : {abs(sh-0.5)/se:.1f} sigma")
print(f"   scatter about the fit            : {float(np.std(res, ddof=2)):.4f} dex")
print(f"   h_eff range                      : {HE.min():.2f} - {HE.max():.2f} kpc")

head("5. Baryonic Tully-Fisher, measured")
Vf = np.array([x["Vf"] for x in GAL])
lv = np.log10(Vf / 100.0)
lm = np.log10(np.array([x["Mb"] for x in GAL]))
A2 = np.vstack([lv, np.ones_like(lv)]).T
s_btf, b_btf = np.linalg.lstsq(A2, lm, rcond=None)[0]
r2 = lm - (s_btf * lv + b_btf)
se2 = float(np.sqrt(np.sum(r2 ** 2) / (len(lm) - 2) / np.sum((lv - lv.mean()) ** 2)))
print(f"   log10 M_b = b + s log10(V_f/100 km/s)")
print(f"   fitted slope s   : {s_btf:+.3f} +- {se2:.3f}   (MOND/RAR predicts 4)")
print(f"   intercept b      : {b_btf:+.3f}")
print(f"   intrinsic scatter: {float(np.std(r2, ddof=2)):.3f} dex")
print(f"   n galaxies       : {len(lm)}")
print(f"\n   NOTE: this uses the OBSERVED V_f. The program requires the fit to")
print(f"   use each model's PREDICTED V_f, which is only available after the")
print(f"   PDE tournament. This row is the target, not a model result.")

head("6. What the rank-2 theorem already forbids")
print("""   An earlier result on this bench: the eight-variable set
   (a_N, Sigma_b, rho_b, r, M_b, theta, Phi_b, environment) has RANK 2.
   SVD singular values 1.5e2, 9.6e1, 2.7e-12; Sigma_b = a_N/(pi G) exactly.

   Consequence for this program's void definitions:

     Q2 = [1 + (|g_N|/a0)^n]^-1     POINT-LOCAL in g_N alone. It lives
                                    entirely inside the rank-2 span, so it
                                    cannot carry information beyond f(a_N, r).
     Q1, Q3                         use rho_L, a SMOOTHED density. The kernel
                                    scale L_rho is genuinely new information
                                    and escapes the span.
     Q4                             nonlocal by construction; escapes.

   So the four void definitions are not four independent options. Q2 is
   redundant with the acceleration axis the RAR already uses, and the
   program's discriminating power sits in L_rho and L_q -- the scales, not
   the functional forms.""")

json.dump(dict(n_galaxies=len(GAL),
               scores={nm: score(p) for nm, p in ROWS},
               s_h=float(sh), s_h_err=float(se), n_heff=int(n),
               btfr_slope=float(s_btf), btfr_err=float(se2),
               btfr_scatter=float(np.std(r2, ddof=2))),
          open("g01_runA.json", "w", encoding="utf-8"), indent=1)
print("\n   wrote g01_runA.json")
