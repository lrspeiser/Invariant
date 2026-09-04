"""NGC 2685: a direction scan of the gravitational field of ONE baryonic system.

The polar-ring lane's headline is a bounded negative -- no polar-ring galaxy
anywhere has a numerically tabulated rotation curve in both planes. But it
surfaced one object that does something better for this purpose.

Jozsa et al. 2009 model NGC 2685's HI as 21 tilted rings, and tabulate for each
ring not only V_rot and the radius but the ring's INCLINATION, POSITION ANGLE and
full 3-D SPIN NORMAL (n_W, n_N, n_LOS) with errors. The position angle swings by
about 125 degrees between 0 and 31 kpc. So the rings sample a range of
ORIENTATIONS through the same baryonic mass distribution, which is exactly the
two-direction measurement section 9 of the programme asks for -- one baryonic
system, tracers in more than one plane -- and it is available as numbers rather
than as a figure.

Jozsa et al. show NGC 2685 is not a classical polar ring but a single warped
COHERENT disk. For this test that is an advantage, not a caveat: a coherent warp
means one connected gas distribution whose orbits change direction with radius,
rather than two dynamically independent components whose relative mass is a free
parameter.

THE TEST, AND THE CONFOUND THAT DECIDES WHETHER IT EXISTS

A scalar law -- Newton, MOND, the RAR -- predicts the boost nu = g_obs/g_bar to
depend on g_bar alone. A tensor law with a preferred axis predicts an additional
dependence on the angle between the orbit's normal and that axis.

But if the ring orientation is a monotone function of radius, then orientation IS
radius, the two cannot be separated, and the test does not exist. That is the
same collapse this programme has now found five times (rank-2, the potential-depth
identity, Freeman's formula, the QUMOND projector, spherical blindness), so it is
checked FIRST and the module reports the answer before computing anything else.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "env-data", "raw", "polar-rings",
                   "josza2009_table5_NGC2685_tiltedring.tsv")

G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30
A0 = 1.2e-10


def load():
    rows = []
    with open(TAB, encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            d = {}
            for k, v in zip(hdr, p):
                v = v.strip()
                d[k] = float(v) if v not in ("", "-") else np.nan
            rows.append(d)
    assert len(rows) == 21, f"expected 21 tilted rings, got {len(rows)}"
    return rows


def main():
    rows = load()
    r = np.array([d["r_kpc"] for d in rows])
    V = np.array([d["V_rot"] for d in rows])
    eV = np.array([d["e_V_rot"] for d in rows])
    inc = np.array([d["incl_deg"] for d in rows])
    pa = np.array([d["PA_deg"] for d in rows])
    nW = np.array([d["n_W"] for d in rows])
    nN = np.array([d["n_N"] for d in rows])
    nL = np.array([d["n_LOS"] for d in rows])
    sig = np.array([d["N_HI_faceon"] for d in rows])

    ok = np.isfinite(nW) & np.isfinite(V) & (r > 0) & (V > 0)
    print("=" * 78)
    print("NGC 2685 -- a direction scan through one baryonic system")
    print("=" * 78)
    print(f"\n   {len(rows)} tilted rings, {ok.sum()} usable "
          f"(finite spin normal, V_rot > 0, r > 0)")
    print(f"   radius {r[ok].min():.2f} - {r[ok].max():.2f} kpc")
    print(f"   PA swings {pa[ok].min():.1f} - {pa[ok].max():.1f} deg "
          f"(range {pa[ok].max()-pa[ok].min():.1f})")
    print(f"   inclination {inc[ok].min():.1f} - {inc[ok].max():.1f} deg")

    # --- the reference axis: the innermost well-determined ring's normal
    n = np.stack([nW[ok], nN[ok], nL[ok]], 1)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    ref = n[0]
    cosang = np.clip(np.abs(n @ ref), 0, 1)
    ang = np.degrees(np.arccos(cosang))
    rr, VV, eVV = r[ok], V[ok], eV[ok]

    print("\n   ring   r[kpc]   V_rot   incl    PA     angle to inner normal")
    for i in range(len(rr)):
        print(f"   {i:4d}  {rr[i]:7.2f}  {VV[i]:6.1f}  {inc[ok][i]:5.1f}  "
              f"{pa[ok][i]:6.1f}   {ang[i]:8.1f} deg")

    # ---------------------------------------------------- THE CONFOUND CHECK
    print("\n" + "-" * 78)
    print("   CONFOUND CHECK: is orientation just radius in disguise?")
    sp = np.corrcoef(np.argsort(np.argsort(rr)),
                     np.argsort(np.argsort(ang)))[0, 1]
    pe = np.corrcoef(rr, ang)[0, 1]
    print(f"      Spearman(r, angle)  = {sp:+.4f}")
    print(f"      Pearson (r, angle)  = {pe:+.4f}")
    # monotone? count sign changes in the angle sequence ordered by radius
    d = np.diff(ang)
    flips = int((np.sign(d[:-1]) != np.sign(d[1:])).sum())
    print(f"      sign changes in d(angle)/d(r): {flips} out of {len(d)-1}")
    print(f"      angle range: {ang.min():.1f} - {ang.max():.1f} deg")

    verdict = abs(sp) < 0.8 and flips >= 2
    if verdict:
        print("      => orientation is NOT monotone in radius. The two are")
        print("         separable and the direction test EXISTS on this object.")
    else:
        print("      => orientation tracks radius too closely. On this object")
        print("         a direction test would be a radius test wearing a")
        print("         different label, and must NOT be run.")

    # ------------------------------------------------- what a test would need
    print("\n" + "-" * 78)
    print("   WHAT A TEST STILL NEEDS, stated before any is attempted")
    print("      g_obs per ring is immediate: V_rot^2 / r.")
    gobs = (VV * 1e3) ** 2 / (rr * KPC)
    print(f"      g_obs / a0 spans {gobs.min()/A0:.3f} - {gobs.max()/A0:.3f}")
    print("      g_bar per ring is NOT immediate. The mass distribution is a")
    print("      warped disk, so g_bar depends on DIRECTION as well as radius,")
    print("      and computing it requires solving for the field of the")
    print("      tabulated HI surface density on the tabulated ring geometry --")
    print("      the axisymmetric solver cannot be used because the system is")
    print("      not axisymmetric. That is a 3-D solve on a warped source, and")
    print("      it is the reason this object has not already been used this way.")
    print("      M_HI = 1.7e9 Msun and L_I = 15.2e9 Lsun are tabulated, so the")
    print("      normalisation is available; the geometry is the work.")

    out = {
        "n_rings": len(rows), "n_usable": int(ok.sum()),
        "r_kpc": rr.tolist(), "V_rot": VV.tolist(), "e_V_rot": eVV.tolist(),
        "incl_deg": inc[ok].tolist(), "PA_deg": pa[ok].tolist(),
        "angle_to_inner_normal_deg": ang.tolist(),
        "g_obs_over_a0": (gobs / A0).tolist(),
        "spearman_r_angle": float(sp), "pearson_r_angle": float(pe),
        "sign_changes": flips,
        "orientation_separable_from_radius": bool(verdict),
    }
    with open(os.path.join(HERE, "ngc2685.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n   written: ngc2685.json")


if __name__ == "__main__":
    main()
