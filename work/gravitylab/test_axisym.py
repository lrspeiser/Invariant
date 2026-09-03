"""Validate the axisymmetric solver before it is allowed to score anything.

Same standard as section 6: it must reproduce a known analytic answer, conserve
flux, and converge. The reference is the exact Freeman razor-thin exponential
disk, approached by making the vertical scale height small.
"""
import numpy as np
import axisym as X

G, KPC, MSUN = 6.674e-11, 3.0856775814913673e19, 1.98892e30
BAR = "=" * 78
print(BAR + "\nAXISYMMETRIC SOLVER VALIDATION\n" + BAR)

Rd, Sigma0 = 3.0, 500 * MSUN / KPC**2 * 1e6   # 500 Msun/pc^2 -> kg/m^2
Mtot = 2 * np.pi * Sigma0 * (Rd * KPC) ** 2

print(f"\n   exponential disk: Rd = {Rd} kpc,  Sigma0 = 500 Msun/pc^2")
print(f"   total mass = {Mtot/MSUN:.3e} Msun\n")
print(f"   {'nR x nz':>12}{'box kpc':>12}{'hz kpc':>9}{'rel err vs Freeman':>21}")
print("   " + "-" * 54)
errs = []
for nR, nz, Rmax, zmax, hz in ((96, 48, 60.0, 30.0, 0.10),
                               (144, 72, 60.0, 30.0, 0.10),
                               (216, 108, 60.0, 30.0, 0.10)):
    g = X.Grid(nR, nz, Rmax, zmax)
    rho = X.exponential_disk(g.Rc / KPC, g.zc / KPC, Sigma0, Rd, hz)
    M = float(np.sum(rho * g.V))
    rho *= (Mtot / 2.0) / M   # grid is the z >= 0 half-space
    A = X.isotropic_A(rho.shape)
    bc = X.monopole_bc(g, Mtot)
    Psi, it, rel = X.solve_axi(rho, A, g, bc, tol=1e-12, maxiter=9000)
    vc = X.midplane_vc(Psi, g)
    vf = X.freeman_vc(g.Rc / KPC, Sigma0, Rd)
    m = (g.Rc / KPC > 1.5) & (g.Rc / KPC < 0.45 * Rmax)
    e = float(np.sqrt(np.mean((vc[m] - vf[m]) ** 2)) / np.sqrt(np.mean(vf[m] ** 2)))
    errs.append(e)
    print(f"   {f'{nR}x{nz}':>12}{f'{Rmax}x{zmax}':>12}{hz:>9.2f}{e:>21.4e}")
print("   " + "-" * 54)
order = np.log(errs[0] / errs[-1]) / np.log(216 / 96)
print(f"   convergence order: {order:.2f}   finest error: {errs[-1]:.3e}")
print(f"   {'PASS' if errs[-1] < 0.05 else 'FAIL'} -- matches the exact Freeman disk")

print("\n" + BAR + "\nFlux conservation on cylindrical surfaces\n" + BAR)
g = X.Grid(144, 72, 60.0, 30.0)
rho = X.exponential_disk(g.Rc / KPC, g.zc / KPC, Sigma0, Rd, 0.10)
rho *= (Mtot / 2.0) / float(np.sum(rho * g.V))
A = X.disk_axis_A(rho.shape, 1.8, 0.55)
Psi, _, _ = X.solve_axi(rho, A, g, X.monopole_bc(g, Mtot), tol=1e-13, maxiter=12000)
ARR, Azz, ARz = A
worst = 0.0
print()
for iR in (30, 50, 70, 90):
    for iz in (20, 35, 50):
        Menc = float(np.sum(rho[:iR, :iz] * g.V[:iR, :iz])) * 2
        dPdR = (Psi[iR, :iz] - Psi[iR - 1, :iz]) / g.dR
        FR = float(np.sum(0.5 * (ARR[iR] + ARR[iR - 1])[:iz] * dPdR
                          * g.AR[iR, :iz])) * 2
        dPdz = (Psi[:iR, iz] - Psi[:iR, iz - 1]) / g.dz
        Fz = float(np.sum(0.5 * (Azz[:iR, iz] + Azz[:iR, iz - 1]) * dPdz
                          * g.Az[:iR, 0])) * 2
        eps = abs(FR + Fz - 4 * np.pi * G * Menc) / (4 * np.pi * G * Menc)
        worst = max(worst, eps)
print(f"   worst eps_flux over 12 cylinders: {worst:.3e}   (threshold 1e-5)")
print(f"   {'PASS' if worst < 1e-5 else 'FAIL'}")

print("\n" + BAR + "\nDoes an anisotropic K actually change the vertical force?\n" + BAR)
print("   K = diag(k_perp, k_par) with the disk normal along z.\n")
print(f"   {'k_par':>8}{'k_perp':>9}{'V(3Rd) km/s':>14}{'K_z(3Rd)':>13}"
      f"{'A_dyn':>9}")
print("   " + "-" * 53)
gi = X.Grid(144, 72, 60.0, 30.0)
rho = X.exponential_disk(gi.Rc / KPC, gi.zc / KPC, Sigma0, Rd, 0.10)
rho *= (Mtot / 2.0) / float(np.sum(rho * gi.V))
bc = X.monopole_bc(gi, Mtot)
base = None
j = int(np.argmin(np.abs(gi.Rc / KPC - 3 * Rd)))
for kpar, kperp in ((1.0, 1.0), (1.0, 0.5), (0.5, 1.0), (2.0, 0.6), (0.6, 2.0)):
    A = X.disk_axis_A(rho.shape, kpar, kperp)
    Psi, _, _ = X.solve_axi(rho, A, gi, bc, tol=1e-11, maxiter=9000)
    v = X.midplane_vc(Psi, gi)[j] / 1e3
    kz = X.vertical_Kz(Psi, gi)[j]
    if base is None:
        base = (v, kz)
        ad = 1.0
    else:
        ad = (v / base[0]) ** 2 / (kz / base[1])
    print(f"   {kpar:>8.2f}{kperp:>9.2f}{v:>14.2f}{kz:>13.3e}{ad:>9.3f}")
print("   " + "-" * 53)
print("""   A_dyn = (g_R/g_R,N)/(K_z/K_z,N). If the tensor is doing real directional
   work these must depart from 1, and in opposite directions depending on which
   eigenvalue is suppressed. That is the signature the vertical test looks for.""")
