"""JOB 3 -- select the development and negative-control samples by INDEPENDENT
geometry, using baryonic, X-ray and large-scale-environment maps ONLY.

THE RULE THIS FILE EXISTS TO ENFORCE.  The two-dimensional shear test in
`shear2d.py` scores the PHASE of the shear quadrupole against two reference
axes.  If either reference axis were derived from the shear itself the test
would be circular, and the phase statistic would have a non-zero null by
construction -- the same shared-quantity failure that retracted rho_p = -0.304
in this programme.  So every quantity used here is one of

    (a) X-ray:        the eFEDS/Bahar+2022 catalogue -- position, redshift,
                      R500, gas mass, temperature, and the Vikhlinin density
                      fit.  Spherically symmetric by construction, so it gives
                      MASS and SCALE but no axis.
    (b) member light: the projected POSITIONS of galaxies whose photometric
                      redshift places them at the cluster.  Positions only --
                      no shape, no ellipticity, no shear.
    (c) environment:  the projected tidal field at the target from the OTHER
                      catalogued X-ray systems, weighted by their gas masses.

and none of them is the shear that will be scored.  The member slice
(|z - z_cl| < 0.10 (1+z_cl)) and the scored background slice (z > z_cl + 0.2)
are DISJOINT by construction, so not one galaxy contributes both a selection
position and a scored shape.

WHAT IS SELECTED

  DEV   large projected member-light ellipticity, measured at >= 2.5 sigma,
        a well-defined external tidal axis, a stated MISALIGNMENT between the
        two (|sin 2 Delta| >= 0.5, i.e. 15 deg <= |Delta| <= 75 deg), and
        enough background sources to measure a quadrupole.
  CTRL  the NEGATIVE CONTROL: near-round in member light (e <= 0.08 and
        0.05 and consistent with zero), same depth and background limits.  A
        detector that reports the same phase signal here as in DEV is
        reporting a systematic.

THE FROZEN ARTEFACT.  This script writes `selection.json` and prints its
SHA-256.  `shear2d.py` recomputes that hash and refuses to run if it differs,
so the sample cannot be edited after a residual has been seen.

CONVENTION.  Position angles use the DECADE/DES ellipticity frame, whose first
axis points WEST:  phi = atan2(d_dec, -d_ra cos dec).  That was MEASURED in
`../efeds-hsc/acquire_decade.py` (the other three sign choices give a negative
tangential shear) and it is used identically for member positions, neighbour
positions and source positions here, so all phase differences are frame
independent.  The frame is a parity flip of the usual east-of-north
convention; every statistic below is a product of two quantities that flip
together, hence invariant.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
EFEDS = os.path.join(LANE, "efeds-hsc")
for p in (EFEDS, os.path.join(LANE, "lead01")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline as P                                            # noqa: E402
import efeds_hsc as E                                           # noqa: E402
import lead01 as L                                              # noqa: E402

CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

TAP = "https://datalab.noirlab.edu/tap/sync"
UA = "research/1.0 (gravity programme)"
CTX = ssl.create_default_context()

# --------------------------------------------------------------- DECLARED CUTS
# Every number below was fixed before any shear quadrupole was computed.
DZ_MEM = 0.10           # member slice half-width, in units of (1 + z_cl)
DZ_BG = 0.20            # background margin, identical to Chiu+2022 / acquire_decade
R_MEM_MIN_MPC = 0.15    # inner member radius: avoid the BCG and blending
R_MEM_MAX_MPC = 1.00    # outer member radius
R_BG_MIN_MPC = 0.20 / 0.7
R_BG_MAX_MPC = 3.50 / 0.7
THETA_CAP_DEG = 1.5
Z_MIN, Z_MAX = 0.10, 0.60
N_MEM_MIN = 25
N_BG_MIN = 500
E_DEV_MIN = 0.15        # development sample: member-light ellipticity
E_DEV_SNR = 2.5
E_CTRL_MAX = 0.05       # negative control: near-round, on the DEBIASED
E_CTRL_SNR = 1.5        # amplitude, and consistent with zero at this level
#: NOTE, recorded because it is a change made after the parent sample was
#: measured but BEFORE any shear was computed.  The control cut was first
#: written on the RAW m = 2 amplitude at <= 0.08.  With a median of 43 members
#: the noise floor of that amplitude is sqrt(2/N) = 0.22, so a raw cut at 0.08
#: does not select round clusters -- it selects downward noise fluctuations.
#: The cut is therefore on the NOISE-DEBIASED amplitude, which is what
#: "near-round" actually means.  No shear had been measured when this was
#: changed; the change is on the selection side only.
SIN2D_MIN = 0.5         # |sin 2 Delta| -- the stated axis misalignment
ENV_NEIGH_MIN = 2       # neighbours needed for an external tidal axis
ENV_DZ = 0.02           # neighbour redshift window, in (1 + z)
ENV_RMAX_MPC = 30.0     # neighbour projected search radius
EDGE_MARGIN_DEG = 0.35  # keep the aperture inside the sampled footprint

COLS = "ra,dec,dnf_z,mcal_w_noshear"
SEL = ("mcal_flags = 0 AND flags_foreground = 0 AND flags_footprint = 1 "
       "AND mcal_sel_noshear > 0 AND dnf_z > 0 AND dnf_z < 3")
QUERY_TEMPLATE = (f"SELECT {COLS} FROM delve_dr3.decade_shear WHERE "
                  "'t' = q3c_radial_query(ra, dec, <RA>, <DEC>, <RAD>) "
                  f"AND {SEL}")


def tap(query, retries=4):
    body = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": query}).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(TAP, data=body,
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300, context=CTX) as r:
                raw = r.read()
            if raw.lstrip().startswith(b"<?xml"):
                raise RuntimeError(raw[:400].decode("utf-8", "replace"))
            return raw
        except Exception:                                       # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(3 * (k + 1))


def parse_csv(raw):
    """Parse a TAP CSV response without materialising a decoded copy.

    The first version of this built a list of split Python strings, which for
    the largest low-redshift apertures raised MemoryError on a response the
    machine had ample RAM for.  pandas parses straight from the byte buffer.
    """
    import io

    import pandas as pd
    df = pd.read_csv(io.BytesIO(raw))
    head = list(df.columns)
    if len(df) == 0:
        return head, np.zeros((0, len(head)))
    return head, df.to_numpy(dtype=float)


def phi_west(ra_s, de_s, ra_c, de_c):
    """Position angle in the DECADE ellipticity frame, and separation in rad."""
    d2r = math.pi / 180.0
    dra = (ra_s - ra_c) * math.cos(de_c * d2r)      # east-positive degrees
    dde = de_s - de_c
    return np.arctan2(dde, -dra), np.hypot(dra, dde) * d2r


def m2_moment(phi, w):
    """Amplitude and phase of the m = 2 azimuthal modulation of a point set.

    For N points with weights w drawn from a surface density
    n(phi) = n0 [1 + e cos 2(phi - phi_0)] the estimators

        c = 2 <w cos 2phi> / <w>,   s = 2 <w sin 2phi> / <w>

    are unbiased for (e cos 2phi_0, e sin 2phi_0).  The 1 sigma error on each
    of c and s for an isotropic parent is sqrt(2 / N_eff), with
    N_eff = (sum w)^2 / sum w^2.
    """
    sw = float(np.sum(w))
    if sw <= 0 or phi.size == 0:
        return 0.0, 0.0, np.inf, 0.0
    c = 2.0 * float(np.sum(w * np.cos(2 * phi))) / sw
    s = 2.0 * float(np.sum(w * np.sin(2 * phi))) / sw
    neff = sw ** 2 / float(np.sum(w ** 2))
    return c, s, math.sqrt(2.0 / neff), neff


def wrap90(a_deg):
    """Wrap an axis-angle difference into (-90, 90] degrees."""
    return (a_deg + 90.0) % 180.0 - 90.0


# ------------------------------------------------------------ environment map
def environment_axis(recs, i):
    """Projected tidal axis at system i from the OTHER catalogued X-ray systems.

    Two weightings are carried, because the standing brief asks for four
    environmental variables and these are two of them:

        tidal   w = M_gas / d^3   -- the projected tidal tensor; opposing
                                     neighbours ADD in the traceless part, so
                                     this defines an axis, not a direction
        wellsum w = M_gas / d^2   -- the directionless inverse-square well
                                     strength, which does not cancel

    The AXIS is the same object in both cases (a spin-2 quantity); the two
    weightings differ in how strongly nearby systems dominate.  The primary is
    the tidal one, declared here.
    """
    r0 = recs[i]
    z0 = r0["z"]
    out = {}
    for tag, pwr in (("tidal", 3.0), ("wellsum", 2.0)):
        cs = ss = norm = 0.0
        n_use = 0
        dmin = np.inf
        for j, rj in enumerate(recs):
            if j == i:
                continue
            if abs(rj["z"] - z0) > ENV_DZ * (1.0 + z0):
                continue
            phi, th = phi_west(np.array([rj["RA"]]), np.array([rj["DE"]]),
                               r0["RA"], r0["DE"])
            d = float(th[0]) * r0["DA"] / P.MPC          # projected Mpc
            if not (0.05 < d < ENV_RMAX_MPC):
                continue
            m = rj["Mgas500_pub"]
            if not (m > 0):
                continue
            w = m / d ** pwr
            cs += w * math.cos(2 * float(phi[0]))
            ss += w * math.sin(2 * float(phi[0]))
            norm += w
            n_use += 1
            dmin = min(dmin, d)
        if norm <= 0 or n_use < ENV_NEIGH_MIN:
            out[tag] = dict(n=n_use, amp=0.0, pa_deg=float("nan"),
                            d_nearest_Mpc=float(dmin) if n_use else float("nan"))
            continue
        amp = math.hypot(cs, ss) / norm
        pa = 0.5 * math.degrees(math.atan2(ss, cs))
        out[tag] = dict(n=n_use, amp=amp, pa_deg=pa, norm=norm,
                        d_nearest_Mpc=float(dmin))
    return out


def neighbour_uniform_shear(recs, i, src_beta=0.35):
    """Predicted UNIFORM reduced shear at system i from catalogued neighbours.

    This is not a nuisance to be waved away.  A neighbouring cluster produces a
    nearly uniform shear across the target's aperture, and a uniform shear
    (gamma_1, gamma_2) enters the tangential ellipticity as
    e_t(phi) = -(gamma_1 cos 2phi + gamma_2 sin 2phi), which is EXACTLY the
    m = 2 mode the two-dimensional test measures -- and it is aligned with the
    external axis by construction.  So it is a first-class false positive for
    the external-axis hypothesis and it is predicted here, per cluster, from
    the neighbours' own gas masses under the baryons + RAR forward model.

    The estimate uses the point-mass deep-MOND shear of each neighbour at the
    target's separation, which for g = sqrt(G M a0)/r gives a projected
    convergence and shear of order kappa ~ sqrt(G M a0) / (c^2 ... ) -- carried
    through with the pipeline's own Sigma_crit.
    """
    r0 = recs[i]
    z0 = r0["z"]
    g1 = g2 = 0.0
    for j, rj in enumerate(recs):
        if j == i or abs(rj["z"] - z0) > ENV_DZ * (1.0 + z0):
            continue
        phi, th = phi_west(np.array([rj["RA"]]), np.array([rj["DE"]]),
                           r0["RA"], r0["DE"])
        d = float(th[0]) * r0["DA"]                     # metres, projected
        if not (0.3 * P.MPC < d < ENV_RMAX_MPC * P.MPC):
            continue
        mg = rj["Mgas500_pub"] * 1e12 * P.MSUN
        if not (mg > 0):
            continue
        mb = 1.15 * mg                                  # gas + stars, Run Z
        # deep-MOND point mass: g = sqrt(G M a0)/r, so the enclosed effective
        # 2-D mass grows like r and Sigma ~ sqrt(G M a0)/(2 pi G r)
        gN = P.G * mb / d ** 2
        gmond = gN / (1.0 - math.exp(-math.sqrt(gN / P.A0)))
        sig = gmond / (2.0 * math.pi * P.G)             # kg/m^2, thin-lens
        scr = P.CLIGHT ** 2 / (4.0 * math.pi * P.G * r0["DA"] * src_beta)
        kap = sig / scr
        # a distant isolated mass gives |gamma| ~ kappa, tangential about it,
        # i.e. an m = 2 pattern whose axis lies along the separation vector
        g1 += -kap * math.cos(2 * float(phi[0]))
        g2 += -kap * math.sin(2 * float(phi[0]))
    return math.hypot(g1, g2), g1, g2


# ---------------------------------------------------------------- per-cluster
def measure_one(rec, raw):
    """Member-light ellipticity, mask control, and background counts."""
    head, a = parse_csv(raw)
    if a.shape[0] < 100:
        return None
    col = {n: i for i, n in enumerate(head)}
    ra, de = a[:, col["ra"]], a[:, col["dec"]]
    zs = a[:, col["dnf_z"]]
    w = a[:, col["mcal_w_noshear"]]
    z_l = rec["z"]
    phi, th = phi_west(ra, de, rec["RA"], rec["DE"])
    Rm = th * rec["DA"] / P.MPC

    mem = (np.abs(zs - z_l) < DZ_MEM * (1.0 + z_l)) \
        & (Rm > R_MEM_MIN_MPC) & (Rm < R_MEM_MAX_MPC)
    # the mask/footprint control: everything in the SAME annulus that is NOT a
    # member.  Its m = 2 moment measures the geometric anisotropy of the
    # coverage, which is then removed from the member moment.
    msk = (np.abs(zs - z_l) > 0.15 * (1.0 + z_l)) \
        & (Rm > R_MEM_MIN_MPC) & (Rm < R_MEM_MAX_MPC)
    bg = (zs > z_l + DZ_BG) & (Rm > R_BG_MIN_MPC) & (Rm < R_BG_MAX_MPC)

    n_mem = int(mem.sum())
    n_msk = int(msk.sum())
    n_bg = int(bg.sum())
    if n_mem < 5 or n_msk < 20:
        return dict(n_mem=n_mem, n_mask=n_msk, n_bg=n_bg, usable=False)

    c_m, s_m, sig_m, neff_m = m2_moment(phi[mem], np.ones(n_mem))
    c_k, s_k, sig_k, neff_k = m2_moment(phi[msk], np.ones(n_msk))
    c = c_m - c_k
    s = s_m - s_k
    sig = math.hypot(sig_m, sig_k)
    e = math.hypot(c, s)
    # the m = 2 amplitude of an isotropic parent is Rayleigh distributed with
    # mode sigma, so subtract the noise bias in quadrature before comparing
    e_deb = math.sqrt(max(e ** 2 - 2.0 * sig ** 2, 0.0))
    pa = 0.5 * math.degrees(math.atan2(s, c))

    # background surface density in the scored annulus, per square arcmin
    area_am2 = math.pi * ((R_BG_MAX_MPC * P.MPC / rec["DA"]) ** 2
                          - (R_BG_MIN_MPC * P.MPC / rec["DA"]) ** 2) \
        * (180 * 60 / math.pi) ** 2
    return dict(n_mem=n_mem, n_mask=n_msk, n_bg=n_bg, usable=True,
                e_mem=e, e_mem_debiased=e_deb, e_mem_err=sig,
                e_mem_snr=e / sig if sig > 0 else 0.0,
                pa_mem_deg=pa,
                e_mask=math.hypot(c_k, s_k),
                pa_mask_deg=0.5 * math.degrees(math.atan2(s_k, c_k)),
                n_bg_per_arcmin2=n_bg / area_am2,
                w_bg_mean=float(np.mean(w[bg])) if n_bg else float("nan"))


def main():
    print("=" * 78)
    print("SELECTION -- development and near-round control samples, chosen by")
    print("             INDEPENDENT geometry and frozen before any scoring")
    print("=" * 78)
    recs, cuts = E.load_efeds()
    print(f"\n   parent catalogue: {len(recs)} eFEDS systems with a "
          f"Bahar+2022 density fit")
    print(f"   ingest cuts: {cuts}")

    # ---- pass 1: environment, purely from the X-ray catalogue
    print("\n   1. external tidal axis from the catalogued X-ray systems")
    env = []
    for i in range(len(recs)):
        e = environment_axis(recs, i)
        amp, g1, g2 = neighbour_uniform_shear(recs, i)
        e["neighbour_shear"] = dict(amp=amp, g1=g1, g2=g2)
        env.append(e)
    ok_env = sum(1 for e in env if np.isfinite(e["tidal"]["pa_deg"]))
    print(f"      {ok_env}/{len(recs)} systems have >= {ENV_NEIGH_MIN} "
          f"catalogued neighbours within {ENV_RMAX_MPC:g} Mpc and "
          f"|dz| < {ENV_DZ}(1+z)")
    ns = np.array([e["neighbour_shear"]["amp"] for e in env])
    print(f"      predicted neighbour uniform shear: median {np.median(ns):.2e}, "
          f"90th pct {np.percentile(ns, 90):.2e}, max {ns.max():.2e}")

    # ---- pass 2: member light, from DECADE POSITIONS only
    print("\n   2. member-light ellipticity from DECADE galaxy POSITIONS")
    print("      (no shape, no ellipticity, no shear is read in this pass)")
    cache_path = os.path.join(CACHE, "selection_positions.json")
    cached = {}
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path, encoding="utf-8"))
        print(f"      resuming from cache: {len(cached)} systems")
    t0 = time.time()
    nq = 0
    for k, rec in enumerate(recs):
        if rec["id"] in cached:
            continue
        rad = min(THETA_CAP_DEG,
                  R_BG_MAX_MPC * P.MPC / rec["DA"] * 180 / math.pi)
        q = (f"SELECT {COLS} FROM delve_dr3.decade_shear WHERE "
             f"'t' = q3c_radial_query(ra, dec, {rec['RA']:.6f}, "
             f"{rec['DE']:.6f}, {rad:.6f}) AND {SEL}")
        try:
            raw = tap(q)
        except Exception as exc:                                # noqa: BLE001
            print(f"      {rec['id']} QUERY FAILED: {exc}")
            continue
        nq += 1
        if nq == 1:
            with open(os.path.join(CACHE, "selection_raw_example.csv"),
                      "wb") as g:
                g.write(raw)
            with open(os.path.join(CACHE, "selection_raw_example.query.txt"),
                      "w", encoding="utf-8") as g:
                g.write(q)
        try:
            m = measure_one(rec, raw)
        except Exception as exc:                                # noqa: BLE001
            print(f"      {rec['id']} PARSE FAILED ({len(raw)} bytes): {exc}")
            m = None
        cached[rec["id"]] = m if m is not None else dict(usable=False,
                                                         n_mem=0, n_mask=0,
                                                         n_bg=0)
        cached[rec["id"]]["radius_deg"] = rad
        if k % 25 == 0:
            json.dump(cached, open(cache_path, "w", encoding="utf-8"))
            print(f"      {k+1}/{len(recs)}  {rec['id']}  "
                  f"{time.time()-t0:.0f}s")
    json.dump(cached, open(cache_path, "w", encoding="utf-8"))
    print(f"      {nq} systems queried, {time.time()-t0:.0f}s")

    # ---- assemble
    rows = []
    for i, rec in enumerate(recs):
        m = cached.get(rec["id"])
        if m is None or not m.get("usable"):
            continue
        e = env[i]
        pa_ext = e["tidal"]["pa_deg"]
        d = wrap90(pa_ext - m["pa_mem_deg"]) if np.isfinite(pa_ext) else np.nan
        d_mask = wrap90(pa_ext - m["pa_mask_deg"]) \
            if np.isfinite(pa_ext) else np.nan
        rows.append(dict(
            id=rec["id"], RA=rec["RA"], DE=rec["DE"], z=rec["z"],
            DA_Mpc=rec["DA"] / P.MPC, R500_Mpc=rec["R500"] / P.MPC,
            Mgas500=rec["Mgas500_pub"], T_keV=rec["T"],
            radius_deg=m["radius_deg"],
            n_mem=m["n_mem"], n_mask=m["n_mask"], n_bg=m["n_bg"],
            n_bg_per_arcmin2=m["n_bg_per_arcmin2"],
            e_mem=m["e_mem"], e_mem_debiased=m["e_mem_debiased"],
            e_mem_err=m["e_mem_err"], e_mem_snr=m["e_mem_snr"],
            pa_mem_deg=m["pa_mem_deg"],
            e_mask=m["e_mask"], pa_mask_deg=m["pa_mask_deg"],
            env_n=e["tidal"]["n"], env_amp=e["tidal"]["amp"],
            pa_ext_deg=pa_ext,
            env_amp_wellsum=e["wellsum"]["amp"],
            pa_ext_wellsum_deg=e["wellsum"]["pa_deg"],
            d_nearest_Mpc=e["tidal"].get("d_nearest_Mpc", float("nan")),
            neighbour_shear=e["neighbour_shear"]["amp"],
            misalign_deg=d, sin2delta=math.sin(2 * math.radians(d))
            if np.isfinite(d) else float("nan"),
            misalign_mask_deg=d_mask))

    print(f"\n   {len(rows)} systems have both a member measurement and an "
          f"environment axis")

    # ---- the frozen selection function
    def edge_ok(r):
        return (126.0 + EDGE_MARGIN_DEG < r["RA"] < 146.0 - EDGE_MARGIN_DEG
                and -3.0 + EDGE_MARGIN_DEG < r["DE"] < 6.0 - EDGE_MARGIN_DEG)

    def common(r):
        return (Z_MIN <= r["z"] <= Z_MAX and r["n_mem"] >= N_MEM_MIN
                and r["n_bg"] >= N_BG_MIN and edge_ok(r)
                and np.isfinite(r["pa_ext_deg"]))

    dev = [r for r in rows if common(r)
           and r["e_mem_debiased"] >= E_DEV_MIN
           and r["e_mem_snr"] >= E_DEV_SNR
           and abs(r["sin2delta"]) >= SIN2D_MIN]
    ctrl = [r for r in rows if common(r)
            and r["e_mem_debiased"] <= E_CTRL_MAX
            and r["e_mem_snr"] <= E_CTRL_SNR]
    dev_ids = {r["id"] for r in dev}
    ctrl = [r for r in ctrl if r["id"] not in dev_ids]

    print(f"\n   SELECTION FUNCTION, declared and now frozen:")
    print(f"      common:  {Z_MIN} <= z <= {Z_MAX}, n_mem >= {N_MEM_MIN}, "
          f"n_bg >= {N_BG_MIN},")
    print(f"               aperture >= {EDGE_MARGIN_DEG} deg inside the eFEDS "
          f"box, external axis defined")
    print(f"      DEV:     e_mem(debiased) >= {E_DEV_MIN}, SNR >= {E_DEV_SNR}, "
          f"|sin 2 Delta| >= {SIN2D_MIN}")
    print(f"      CTRL:    e_mem(debiased) <= {E_CTRL_MAX} and SNR <= "
          f"{E_CTRL_SNR} (near-round negative control)")
    print(f"\n      DEV  : {len(dev)} systems")
    print(f"      CTRL : {len(ctrl)} systems")

    if dev:
        mis = np.array([abs(r["misalign_deg"]) for r in dev])
        print(f"      DEV misalignment |Delta|: median {np.median(mis):.1f} deg, "
              f"range {mis.min():.1f}-{mis.max():.1f} deg")
        ee = np.array([r["e_mem_debiased"] for r in dev])
        print(f"      DEV member ellipticity: median {np.median(ee):.3f}, "
              f"range {ee.min():.3f}-{ee.max():.3f}")
        nb = np.array([r["n_bg"] for r in dev])
        print(f"      DEV background sources: median {np.median(nb):.0f}, "
              f"total {nb.sum():.0f}")

    # ---- the systematic that could couple selection to the phase statistic
    fin = [r for r in rows if np.isfinite(r["pa_ext_deg"])]
    dm = np.array([math.sin(2 * math.radians(r["misalign_mask_deg"]))
                   for r in fin])
    dd = np.array([r["sin2delta"] for r in fin])
    rho = float(np.corrcoef(dm, dd)[0, 1]) if len(fin) > 3 else float("nan")
    print(f"\n   SYSTEMATIC CHECK.  If the survey MASK were aligned with the")
    print(f"   external axis, an error in pa_mem would correlate with "
          f"sin 2 Delta")
    print(f"   and could fake the phase statistic.  corr(sin2(pa_ext-pa_mask), "
          f"sin2 Delta) = {rho:+.3f}")
    print(f"   over {len(fin)} systems.  |rho| below ~0.2 means the mask "
          f"cannot drive the result.")

    out = dict(
        generated_utc=dt.datetime.now(dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        lane="work/wellnet-2026-09/axis-2d",
        parent=dict(catalogue="Bahar+2022 eFEDS (VizieR J/A+A/661/A7) x "
                              "DELVE DR3 delve_dr3.decade_shear positions",
                    n_parent=len(recs), ingest_cuts=cuts,
                    n_measured=len(rows)),
        selection_function=dict(
            DZ_MEM=DZ_MEM, DZ_BG=DZ_BG, R_MEM_MIN_MPC=R_MEM_MIN_MPC,
            R_MEM_MAX_MPC=R_MEM_MAX_MPC, R_BG_MIN_MPC=R_BG_MIN_MPC,
            R_BG_MAX_MPC=R_BG_MAX_MPC, Z_MIN=Z_MIN, Z_MAX=Z_MAX,
            N_MEM_MIN=N_MEM_MIN, N_BG_MIN=N_BG_MIN, E_DEV_MIN=E_DEV_MIN,
            E_DEV_SNR=E_DEV_SNR, E_CTRL_MAX=E_CTRL_MAX,
            E_CTRL_SNR=E_CTRL_SNR, SIN2D_MIN=SIN2D_MIN,
            ENV_NEIGH_MIN=ENV_NEIGH_MIN, ENV_DZ=ENV_DZ,
            ENV_RMAX_MPC=ENV_RMAX_MPC, EDGE_MARGIN_DEG=EDGE_MARGIN_DEG,
            query_template=QUERY_TEMPLATE,
            phase_convention="phi = atan2(d_dec, -d_ra cos dec)"),
        mask_systematic_rho=rho,
        neighbour_shear_median=float(np.median(ns)),
        neighbour_shear_p90=float(np.percentile(ns, 90)),
        dev=dev, ctrl=ctrl, all_measured=rows)
    path = os.path.join(HERE, "selection.json")
    blob = json.dumps(out, indent=1, sort_keys=True).encode("utf-8")
    with open(path, "wb") as f:
        f.write(blob)
    h = hashlib.sha256(blob).hexdigest()
    with open(path + ".sha256", "w", encoding="utf-8") as f:
        f.write(h + "\n")
    print(f"\n   written: selection.json")
    print(f"   SHA-256: {h}")
    print("   shear2d.py verifies this hash and refuses to run if the sample")
    print("   was edited after a residual was seen.")


if __name__ == "__main__":
    main()
