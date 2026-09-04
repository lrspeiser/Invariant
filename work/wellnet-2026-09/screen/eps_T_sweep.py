"""Run AQ -- the eps_T re-run owed from Run AG.

Family E's response is  K = exp[f0 I + fT That],  That = T0/sqrt(eps_T^2+|T0|^2).
The frozen eps_T = A0/(10 kpc) = 3.89e-31 s^-2 was never checked against the
|T0| that actually occurs, so E1/E2 may have been screened as near-inert laws.

This measures |T0|(r) on the screen's own galaxy, then re-runs the full Stage-1
screen for family E at eps_T spanning six decades around it.
"""
import json
import sys
import time

import numpy as np

import families as F
import screen as S

MSUN, KPC, A0 = F.MSUN, F.KPC, F.A0

# ---------------------------------------------------------------- 1. |T0|(r)
wx, wm = S.galaxy_cloud(Nq=32768)
Mtot = float(np.sum(wm))
print(f"galaxy cloud: {len(wm)} rows, M = {Mtot/MSUN:.3e} Msun\n")

print("|T0| against radius on the screen's own galaxy")
print(f"{'r (kpc)':>9} {'|T0| (s^-2)':>13} {'|T0|/eps_T frozen':>19} {'|That| frozen':>14}")
EPS_FROZEN = A0 / (10.0 * KPC)
radii = np.array([0.5, 1, 2, 5, 10, 20, 40, 80]) * KPC
rows = []
for r in radii:
    ndir = 64
    rng = np.random.default_rng(11)
    d = rng.normal(size=(ndir, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    T = S.tidal_analytic(d * r, wx, wm)
    tr = np.trace(T, axis1=-2, axis2=-1)
    T0 = T - tr[..., None, None] * np.eye(3) / 3.0
    nrm = np.sqrt((T0 * T0).sum((-1, -2)))
    med = float(np.median(nrm))
    hat = med / np.sqrt(EPS_FROZEN ** 2 + med ** 2)
    rows.append(dict(r_kpc=float(r / KPC), T0=med,
                     ratio=med / EPS_FROZEN, That=hat))
    print(f"{r/KPC:9.1f} {med:13.4e} {med/EPS_FROZEN:19.4f} {hat:14.4f}")

# a physically scaled eps_T: the median |T0| over the probe shell the screens use
Tref = float(np.median([x["T0"] for x in rows if 2 <= x["r_kpc"] <= 40]))
print(f"\nmedian |T0| over 2-40 kpc (the probe range) = {Tref:.4e} s^-2")
print(f"frozen eps_T                                = {EPS_FROZEN:.4e} s^-2")
print(f"frozen eps_T is {EPS_FROZEN/Tref:.1f}x the typical |T0| it normalises\n")

# ------------------------------------------------------ 2. the eps_T sweep
GRID = [("frozen", EPS_FROZEN),
        ("x1e-1", EPS_FROZEN * 1e-1),
        ("matched", Tref),
        ("x1e-2", EPS_FROZEN * 1e-2),
        ("x1e-3", EPS_FROZEN * 1e-3),
        ("x1e-4", EPS_FROZEN * 1e-4),
        ("zero", 0.0)]

out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "eps_T_frozen": EPS_FROZEN, "T0_profile": rows, "T0_ref": Tref,
       "runs": {}}

print("=" * 78)
print("Stage-1 screen, family E, eps_T swept")
print("=" * 78)
for fT, base in ((0.5, "E1"), (1.5, "E2")):
    for tag, eps in GRID:
        nm = f"{base}_epsT_{tag}"
        cand = F.Candidate(nm, "E", "tidal",
                           F._pE(fT=fT, eps_T=float(eps)), F._DIMS_E,
                           "K = exp[f0 I + fT That]  (eps_T sweep)")
        t = time.time()
        try:
            r = S.run_screen(cand, Nq=32768)
        except Exception as e:                              # noqa: BLE001
            print(f"{nm:24s} ERROR {e!r}")
            out["runs"][nm] = dict(error=repr(e))
            continue
        sc = r["screens"]

        def val(k):
            v = sc.get(k, {})
            return v.get("value") if isinstance(v, dict) else None

        out["runs"][nm] = dict(fT=fT, eps_T=float(eps), verdict=r["verdict"],
                               failed=r["failed"],
                               s6=val("S6_newtonian_limit"),
                               s8=val("S8_gain_bound"),
                               s10=val("S10_reciprocity"),
                               s11=val("S11_coarse_uniform"),
                               seconds=r["seconds"])
        g = val("S8_gain_bound")
        print(f"{nm:24s} {r['verdict']:6s} "
              f"S6={val('S6_newtonian_limit'):8.4f} "
              f"gain={g if g is None else round(g,4):<8} "
              f"S10={val('S10_reciprocity'):8.4f} "
              f"({time.time()-t:.0f}s)  failed={r['failed']}", flush=True)

# ------------------------------------------- 3. the bound that eps_T cannot move
print("\n" + "=" * 78)
print("the analytic gain ceiling, which is INDEPENDENT of eps_T")
print("=" * 78)
print("|That|_F <= 1 by construction, so the largest eigenvalue of That is <= 1")
print("and K = exp(fT That) has max eigenvalue <= exp(fT):\n")
for fT in (0.5, 1.5, 3.0, 5.0):
    print(f"   fT = {fT:4.1f}   max gain g/g_N <= exp(fT) = {np.exp(fT):9.3f}")
print("\nThe kinematic requirement for a flat curve at 40 kpc is a gain that")
print("GROWS without bound outward.  No eps_T changes that ceiling.")
out["gain_ceiling"] = {str(fT): float(np.exp(fT)) for fT in (0.5, 1.5, 3.0, 5.0)}

with open("eps_T_sweep.json", "w", newline="\n") as fh:
    json.dump(out, fh, indent=1, default=float)
print("\nwrote eps_T_sweep.json")
