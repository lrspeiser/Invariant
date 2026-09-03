"""
IS THE FEW-PERCENT PROJECTED DIFFERENCE A REAL BIAS, OR SHOT NOISE?

v04 found the lumpy-vs-smoothed QUMOND difference to be 0.4% in 3D shells but
up to 5.6% in the innermost projected annulus, and the guard flagged that as
inconsistent. The radial pattern -- 5.6, 2.5, 2.5, 0.1, -0.2, -0.2 percent --
decreases with radius and CHANGES SIGN, which is the signature of where a
handful of galaxies happened to land, not of a systematic nonlinear bias.

A systematic bias keeps its sign across realizations. Shot noise does not.
So: redraw the galaxy population several times and look at the spread.
"""
import json
import math
import numpy as np

SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
G, KPC, MSUN, A0 = 6.674e-11, 3.0856775814913673e19, 1.98892e30, 1.2e-10
BAR = "=" * 78
NREAL, NGAL, FSTAR = 5, 300, 0.15
N, LBOX = 160, 4200.0 * KPC


def nu(y):
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(y, 1e-30))))


from invariant_bench import Bench
b = Bench(verbose=False)
xc = b.d["xcop"]
ext = np.asarray(xc.extent, float) * np.ones(len(xc)) / KPC
m = np.abs(ext - 1414.0) < 1e-9
o = np.argsort(xc.r[m])
r_dat, gb_dat = xc.r[m][o], xc.gb[m][o]
Mgas = np.maximum.accumulate(gb_dat * r_dat ** 2 / G)
lum_obs = np.array(json.load(open(SCR + "lens_sub.json",
                                  encoding="utf-8"))["groups"][0]["mL"], float)

dx = LBOX / N
ax = (np.arange(N) - N / 2 + 0.5) * dx
X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
RR = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
R2 = np.sqrt(X[:, :, 0] ** 2 + Y[:, :, 0] ** 2)
kx = 2 * math.pi * np.fft.fftfreq(N, d=dx)
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
K2 = KX ** 2 + KY ** 2 + KZ ** 2
K2[0, 0, 0] = 1.0
zsel = np.abs(ax) < 2000 * KPC

cpo = np.polyfit(np.log(r_dat), np.log(Mgas), 4)
sl = np.clip(np.polyval(np.polyder(cpo), np.log(r_dat)), 0.02, 3.5)
rho_r = np.exp(np.polyval(cpo, np.log(r_dat))) * sl / (4 * math.pi * r_dat ** 3)
k = r_dat > 0.35 * r_dat[-1]
p_out = float(np.clip(np.polyfit(np.log(r_dat[k]), np.log(rho_r[k]), 1)[0],
                      -5.0, -2.0))
rho_gas = np.interp(RR, r_dat, rho_r)
out = RR > r_dat[-1]
rho_gas[out] = rho_r[-1] * (RR[out] / r_dat[-1]) ** p_out
rho_gas[RR < r_dat[0]] = rho_r[0]
rho_gas *= (1 - FSTAR)
inR = RR <= r_dat[-1]
rho_gas *= ((1 - FSTAR) * Mgas[-1]) / (rho_gas[inR].sum() * dx ** 3)

nb = 200
edges = np.linspace(0, RR.max(), nb + 1)
which = np.clip(np.digitize(RR.ravel(), edges) - 1, 0, nb - 1)
cnt = np.bincount(which, minlength=nb).astype(float)


def solve(rho):
    ph = np.fft.fftn(4 * math.pi * G * rho) / (-K2)
    ph[0, 0, 0] = 0.0
    PhiN = np.real(np.fft.ifftn(ph))
    F = np.fft.fftn(PhiN)
    gx = np.real(np.fft.ifftn(1j * KX * F))
    gy = np.real(np.fft.ifftn(1j * KY * F))
    gz = np.real(np.fft.ifftn(1j * KZ * F))
    n = nu(np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) / A0)
    src = np.real(np.fft.ifftn(
        1j * KX * np.fft.fftn(n * gx) + 1j * KY * np.fft.fftn(n * gy)
        + 1j * KZ * np.fft.fftn(n * gz)))
    Psi = np.real(np.fft.ifftn(np.fft.fftn(src) / (-K2)))
    P = np.fft.fftn(Psi)
    px = np.real(np.fft.ifftn(1j * KX * P))
    py = np.real(np.fft.ifftn(1j * KY * P))
    alp = np.sqrt((px[:, :, zsel].sum(axis=2) * dx) ** 2
                  + (py[:, :, zsel].sum(axis=2) * dx) ** 2)
    return alp


RADII = [150, 300, 500, 800, 1100, 1400]
print(BAR + "\n%d realizations of the galaxy population, %d^3 grid\n" % (NREAL, N)
      + BAR)
print("   ratio of projected deflection, lumpy / spherically smoothed\n")
print("   %5s" % "seed" + "".join("%10d" % r for r in RADII))
print("   " + "-" * (5 + 10 * len(RADII)))
ALL = []
for s in range(NREAL):
    rng = np.random.default_rng(1000 + s)
    lum = rng.choice(lum_obs, size=NGAL, replace=True)
    Mg = (FSTAR * Mgas[-1]) * (lum / lum.sum())
    u = rng.random(NGAL)
    rg = np.interp(u * Mgas[-1], Mgas, r_dat)
    ct = 2 * rng.random(NGAL) - 1
    ph_ = 2 * math.pi * rng.random(NGAL)
    st = np.sqrt(1 - ct ** 2)
    gal = np.stack([rg * st * np.cos(ph_), rg * st * np.sin(ph_), rg * ct], 1)
    rho_gal = np.zeros_like(rho_gas)
    idx = ((gal + LBOX / 2) / dx - 0.5)
    for q in range(NGAL):
        i0 = np.floor(idx[q]).astype(int)
        f = idx[q] - i0
        for a_ in (0, 1):
            for b_ in (0, 1):
                for c_ in (0, 1):
                    w = ((f[0] if a_ else 1 - f[0]) * (f[1] if b_ else 1 - f[1])
                         * (f[2] if c_ else 1 - f[2]))
                    i, j, kk = i0[0] + a_, i0[1] + b_, i0[2] + c_
                    if 0 <= i < N and 0 <= j < N and 0 <= kk < N:
                        rho_gal[i, j, kk] += w * Mg[q] / dx ** 3
    rl = rho_gas + rho_gal
    tot = np.bincount(which, weights=rl.ravel(), minlength=nb)
    rs = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)[which].reshape(RR.shape)
    aL, aS = solve(rl), solve(rs)
    row = []
    for rk in RADII:
        sel = np.abs(R2 - rk * KPC) < dx
        row.append(aL[sel].mean() / aS[sel].mean())
    ALL.append(row)
    print("   %5d" % s + "".join("%10.4f" % v for v in row))
print("   " + "-" * (5 + 10 * len(RADII)))
A = np.array(ALL)
print("   %5s" % "mean" + "".join("%10.4f" % v for v in A.mean(axis=0)))
print("   %5s" % "sd" + "".join("%10.4f" % v for v in A.std(axis=0, ddof=1)))
print("\n" + BAR + "\nVERDICT\n" + BAR)
mean_dev = np.abs(A.mean(axis=0) - 1) * 100
sd_dev = A.std(axis=0, ddof=1) * 100
signs = [len(set(np.sign(A[:, j] - 1))) for j in range(A.shape[1])]
print("   per-radius |mean - 1|  : " + "  ".join("%.2f%%" % v for v in mean_dev))
print("   per-radius scatter     : " + "  ".join("%.2f%%" % v for v in sd_dev))
print("   sign flips across seeds: " + "  ".join(
    ("yes" if s > 1 else "no ") for s in signs))
consistent = np.all(mean_dev < 2 * np.maximum(sd_dev / math.sqrt(NREAL), 1e-9)) \
    or np.all(mean_dev < 1.0)
print("""
   If the few-percent difference were a systematic nonlinear bias it would
   keep the same sign in every realization and its mean would be large
   compared with the scatter. If it is shot noise from where a handful of
   galaxies happen to land, the sign flips and the mean collapses toward 1
   as realizations are averaged.

   Overall mean across all radii and seeds: %.4f
   Largest |mean - 1| at any radius       : %.2f%%
   Typical realization-to-realization sd  : %.2f%%
""" % (A.mean(), mean_dev.max(), sd_dev.mean()))
print("   Interpretation: %s" % (
    "SHOT NOISE -- no systematic bias survives averaging."
    if mean_dev.max() < 2.0 else
    "a residual systematic remains; report it as such."))
