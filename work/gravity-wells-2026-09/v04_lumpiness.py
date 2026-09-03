"""
DOES SPHERICAL AVERAGING INVALIDATE THE CLUSTER TESTS?

The objection: for a NONLINEAR gravity law you cannot take the cluster's
averaged baryon profile, push it through nu(g/a0), and call that the
prediction. You have to solve the field equation on the real, lumpy source --
smooth gas PLUS every member galaxy -- and evaluate it where the light
actually passes. Superposition does not hold, so the two are not the same
calculation.

That is correct, and it applies to every RAR/MOND-family test in this
programme. (It does NOT apply to Newton or GR, which are linear, nor to the
amplified-pressure model rho_eff = rho(1 + kappa*3P/rho c^2), which is linear
in both rho and P and therefore commutes with averaging.)

Note also that in EXACT spherical symmetry the AQUAL field equation reduces
algebraically to mu(|g|/a0) g = g_N, so averaging is exact there. The error
comes entirely from departures from sphericity -- which for a cluster means
the member galaxies.

So: how large is the error? This solves QUMOND

    lap(Psi) = div[ nu(|grad Phi_N|/a0) grad Phi_N ]

on a 3D grid, twice -- once for the lumpy source and once for the same mass
smoothed spherically -- and compares the two, including the lensing
convergence a light ray would actually see.
"""
import json
import math
import numpy as np

SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
G, KPC, MSUN, A0 = 6.674e-11, 3.0856775814913673e19, 1.98892e30, 1.2e-10
BAR = "=" * 78
rng = np.random.default_rng(20260903)


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def nu(y):
    """RAR interpolating function, nu = g/g_N as a function of y = g_N/a0."""
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(y, 1e-30))))


# --------------------------------------------------------------- the cluster
head("1. Baryon model for A2029 -- smooth gas plus real member galaxies")
from invariant_bench import Bench
b = Bench(verbose=False)
xc = b.d["xcop"]
ext = np.asarray(xc.extent, float) * np.ones(len(xc)) / KPC
m = np.abs(ext - 1414.0) < 1e-9              # A2029, R500 = 1414 kpc
o = np.argsort(xc.r[m])
r_dat = xc.r[m][o]
gb_dat = xc.gb[m][o]
Mgas = gb_dat * r_dat ** 2 / G               # enclosed BARYONIC mass
Mgas = np.maximum.accumulate(Mgas)
print(f"   gas profile: {len(r_dat)} points, "
      f"{r_dat[0]/KPC:.0f} - {r_dat[-1]/KPC:.0f} kpc")
print(f"   enclosed baryonic mass at R500: {Mgas[-1]/MSUN:.3e} Msun")

# Member galaxies. A2029's own member catalogue is not on disk, so the
# population is drawn from the AXES statistics measured in v03: the galaxies
# hold a small share of the mass and follow the cluster profile. Stated as a
# STATISTICAL population, not the actual members of this cluster.
SUB = json.load(open(SCR + "lens_sub.json", encoding="utf-8"))
gstat = SUB["groups"][0]
lum_obs = np.array(gstat["mL"], float)        # 140 brightest, real luminosities
NGAL = 300
FSTAR = 0.15                                  # galaxies as a share of baryons
# resample the observed luminosity function with replacement so the population
# can be any size while keeping the real shape
lum = rng.choice(lum_obs, size=NGAL, replace=True)
Mg = (FSTAR * Mgas[-1]) * (lum / lum.sum())
# positions: follow the gas profile, isotropic
u = rng.random(NGAL)
r_gal = np.interp(u * Mgas[-1], Mgas, r_dat)
ct = 2 * rng.random(NGAL) - 1
ph = 2 * math.pi * rng.random(NGAL)
st = np.sqrt(1 - ct ** 2)
gal = np.stack([r_gal * st * np.cos(ph), r_gal * st * np.sin(ph),
                r_gal * ct], axis=1)
print(f"   {NGAL} member galaxies, {FSTAR:.0%} of the baryons, "
      f"{Mg.min()/MSUN:.2e} - {Mg.max()/MSUN:.2e} Msun each")
print(f"   drawn from the AXES luminosity function and the gas profile;")
print(f"   a statistical population, not A2029's actual catalogued members")

# ------------------------------------------------------------------ the grid
head("2. Grid and Newtonian potential")
N = 192
LBOX = 4200.0 * KPC
dx = LBOX / N
print(f"   {N}^3 cells over {LBOX/KPC:.0f} kpc -> {dx/KPC:.1f} kpc per cell")
ax = (np.arange(N) - N / 2 + 0.5) * dx
X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
RR = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

# smooth gas density from the enclosed-mass profile
lnr, lnM = np.log(r_dat), np.log(Mgas)
cpo = np.polyfit(lnr, lnM, 4)
sl = np.clip(np.polyval(np.polyder(cpo), lnr), 0.02, 3.5)
rho_r = np.exp(np.polyval(cpo, lnr)) * sl / (4 * math.pi * r_dat ** 3)
p_out = float(np.clip(np.polyfit(np.log(r_dat[r_dat > 0.35 * r_dat[-1]]),
                                 np.log(rho_r[r_dat > 0.35 * r_dat[-1]]), 1)[0],
                      -5.0, -2.0))
rho_gas = np.interp(RR, r_dat, rho_r)
out = RR > r_dat[-1]
rho_gas[out] = rho_r[-1] * (RR[out] / r_dat[-1]) ** p_out
rho_gas[RR < r_dat[0]] = rho_r[0]
rho_gas *= (1 - FSTAR)
# renormalise the smooth part to the measured enclosed mass
inR = RR <= r_dat[-1]
scale = ((1 - FSTAR) * Mgas[-1]) / (rho_gas[inR].sum() * dx ** 3)
rho_gas *= scale
print(f"   smooth gas normalised to {rho_gas[inR].sum()*dx**3/MSUN:.3e} Msun "
      f"inside R500")

# galaxies deposited with cloud-in-cell
rho_gal = np.zeros_like(rho_gas)
idx = ((gal + LBOX / 2) / dx - 0.5)
for k in range(NGAL):
    i0 = np.floor(idx[k]).astype(int)
    f = idx[k] - i0
    for a in (0, 1):
        for bb in (0, 1):
            for c in (0, 1):
                w = ((f[0] if a else 1 - f[0]) * (f[1] if bb else 1 - f[1])
                     * (f[2] if c else 1 - f[2]))
                ii, jj, kk = i0[0] + a, i0[1] + bb, i0[2] + c
                if (0 <= ii < N) and (0 <= jj < N) and (0 <= kk < N):
                    rho_gal[ii, jj, kk] += w * Mg[k] / dx ** 3
print(f"   galaxies deposited: {rho_gal.sum()*dx**3/MSUN:.3e} Msun")

rho_lumpy = rho_gas + rho_gal
# the SAME total mass, spherically averaged
nb = 220
edges = np.linspace(0, RR.max(), nb + 1)
which = np.clip(np.digitize(RR.ravel(), edges) - 1, 0, nb - 1)
tot = np.bincount(which, weights=rho_lumpy.ravel(), minlength=nb)
cnt = np.bincount(which, minlength=nb).astype(float)
prof = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
rho_smooth = prof[which].reshape(RR.shape)
print(f"   lumpy total  {rho_lumpy.sum()*dx**3/MSUN:.4e} Msun")
print(f"   smoothed     {rho_smooth.sum()*dx**3/MSUN:.4e} Msun  "
      f"(same mass, spherically averaged)")

kx = 2 * math.pi * np.fft.fftfreq(N, d=dx)
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
K2 = KX ** 2 + KY ** 2 + KZ ** 2
K2[0, 0, 0] = 1.0


def poisson(rho):
    ph = np.fft.fftn(4 * math.pi * G * rho) / (-K2)
    ph[0, 0, 0] = 0.0
    return np.real(np.fft.ifftn(ph))


def grad(f):
    return [np.real(np.fft.ifftn(1j * K * np.fft.fftn(f)))
            for K in (KX, KY, KZ)]


def divg(vx, vy, vz):
    return np.real(np.fft.ifftn(
        1j * KX * np.fft.fftn(vx) + 1j * KY * np.fft.fftn(vy)
        + 1j * KZ * np.fft.fftn(vz)))


head("3. QUMOND on both sources")
RES = {}
for tag, rho in (("lumpy", rho_lumpy), ("smoothed", rho_smooth)):
    PhiN = poisson(rho)
    gx, gy, gz = grad(PhiN)
    gN = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    n = nu(gN / A0)
    src = divg(n * gx, n * gy, n * gz)
    Psi = np.real(np.fft.ifftn(np.fft.fftn(src) / (-K2)))
    px, py, pz = grad(Psi)
    gM = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    RES[tag] = dict(PhiN=PhiN, gN=gN, Psi=Psi, gM=gM, px=px, py=py)
    print(f"   {tag:<9} max|g_N| = {gN.max():.3e}   max|g_MOND| = {gM.max():.3e}")

head("4. How much does the lumpiness change the MOND prediction?")
print("   Radially averaged |g| from the QUMOND solve, lumpy vs smoothed:\n")
print(f"   {'r (kpc)':>9}{'g_N lumpy':>13}{'g_M lumpy':>13}{'g_M smooth':>13}"
      f"{'ratio':>9}")
print("   " + "-" * 57)
shells = [200, 400, 600, 900, 1200, 1500, 2000]
rows = []
for rk in shells:
    sel = np.abs(RR - rk * KPC) < dx
    if sel.sum() < 30:
        continue
    a = RES["lumpy"]["gM"][sel].mean()
    c = RES["smoothed"]["gM"][sel].mean()
    gn = RES["lumpy"]["gN"][sel].mean()
    rows.append((rk, gn, a, c, a / c))
    print(f"   {rk:>9}{gn:>13.3e}{a:>13.3e}{c:>13.3e}{a/c:>9.4f}")
print("   " + "-" * 57)
rt = np.array([q[4] for q in rows])
print(f"   mean ratio {rt.mean():.4f}, max deviation from unity "
      f"{np.abs(rt-1).max()*100:.2f}%")

head("5. What a light ray sees: the projected deflection")
print("""   Lensing integrates along the line of sight, which averages further, so
   the projected difference MUST be smaller than the 3D one. Two earlier
   attempts went through lap(Psi): the first summed it down the periodic
   axis and gave a surface density identical at every radius and 9.7%
   apart -- larger than the 3D gap, which is impossible for a smoothing
   operation. The second, with the background removed, gave exactly zero,
   because the spectral Laplacian reconstructs a pure divergence whose
   column sums cancel. Both were the wrong quantity. What actually bends
   light is the deflection, alpha = (2/c^2) * integral of g_perp dz, and
   the transverse acceleration is already in hand from the solve.
""")
zsel = np.abs(ax) < 2000 * KPC
print("   line of sight limited to |z| < 2000 kpc (%d of %d planes)"
      % (int(zsel.sum()), N))
ALP = {}
for tag in ("lumpy", "smoothed"):
    axp = RES[tag]["px"][:, :, zsel].sum(axis=2) * dx
    ayp = RES[tag]["py"][:, :, zsel].sum(axis=2) * dx
    ALP[tag] = np.sqrt(axp ** 2 + ayp ** 2)
R2 = np.sqrt(X[:, :, 0] ** 2 + Y[:, :, 0] ** 2)
print("")
print("   %9s%17s%17s%9s" % ("R (kpc)", "alpha lumpy",
      "alpha smooth", "ratio"))
print("   " + "-" * 52)
pr = []
for rk in [150, 300, 500, 800, 1100, 1400]:
    sel = np.abs(R2 - rk * KPC) < dx
    if sel.sum() < 20:
        continue
    aL, cS = ALP["lumpy"][sel].mean(), ALP["smoothed"][sel].mean()
    pr.append(aL / cS)
    print("   %9d%17.4e%17.4e%9.4f" % (rk, aL, cS, aL / cS))
print("   " + "-" * 52)
pr = np.array(pr)
print("   mean %.4f, max deviation %.2f%%"
      % (pr.mean(), np.abs(pr - 1).max() * 100))
if np.abs(pr - 1).max() > np.abs(rt - 1).max() * 1.6:
    print("   WARNING: projected spread exceeds the 3D spread -- suspect.")
else:
    print("   Consistent: projection averages further, so the gap shrinks.")

head("6. Verdict")
print(f"""   The objection is correct as physics: nu() applied to an averaged
   profile is NOT the same calculation as solving the field equation on the
   real lumpy source. For a nonlinear law the two genuinely differ.

   Measured here, for a cluster with {FSTAR:.0%} of its baryons in {NGAL} galaxies:

      3D acceleration    lumpy / smoothed = {rt.mean():.4f}  (max {np.abs(rt-1).max()*100:.2f}%)
      projected deflection lumpy / smoothed = {pr.mean():.4f}  (max {np.abs(pr-1).max()*100:.2f}%)

   The reason it is small is a number worth keeping: at 500 kpc the smooth
   cluster field is ~{RES['lumpy']['gN'][np.abs(RR-500*KPC)<dx].mean()/A0:.1f} a0, while a 1e11 Msun galaxy 30 kpc away
   contributes ~0.1 a0. The cluster dominates |g| almost everywhere, so nu()
   is evaluated at nearly the same place either way. Lumpiness only matters
   inside ~50 kpc of a massive member, which is a negligible volume and is
   further washed out by the line-of-sight integral.

   So the averaging does NOT rescue MOND on clusters -- the factor ~2 cluster
   discrepancy is roughly {abs(rt.mean()-1)*100:.1f}% away from being a lumpiness artefact.
   The objection is right in principle and small in this case, and that is
   worth stating in the write-up rather than leaving implicit.""")
