"""Lead 01: does potential depth carry information WITHIN one class?

The galaxies-to-clusters ladder (Run R) gave 0.766 dex of |Phi_b| leverage at
fixed g_bar and still could not decide the potential-depth hypothesis, because
86% of that leverage was the class label -- remove it and only 0.286 dex remains,
less than SPARC alone. Every individual rung offered 0.10-0.36 dex, and the
X-ray group rungs offered the LEAST, because those tables give two overdensity
radii at a fixed ratio so r barely moves at fixed g_bar.

eFEDS removes that cap. Bahar et al. 2022 (A&A 661, A7 = arXiv:2110.09534) fit a
Vikhlinin+2006 electron-density model to 542 eFEDS groups and clusters and
tabulate its parameters, so g_bar(r) is continuous and

    log|Phi_b| = log g_bar + log r + log S,   S(r) = |Phi_b| / (g_bar r)

has real spread in BOTH r and S, from profile-shape diversity, inside a single
instrument and a single pipeline.

THE MODEL, taken from the paper's own table caption rather than assumed:

    n_e^2(r) = n0^2 (r/rs)^-alpha [1+(r/rs)^2]^(-3beta+alpha/2) [1+(r/rs)^3]^(-eps/3)

Note that the VizieR column labelled `n0` is the paper's n0^2, in 10^-7 cm^-6 --
the units give it away, and taking it as n0 would put every density off by its
own square root.

TWO OBSERVABLES, BOTH FROM MEASUREMENTS, NEITHER ASSUMING DARK MATTER:

    g_bar(r) = G M_b(<r) / r^2,  M_b from integrating the fitted n_e(r)
    g_obs(r) = -(kT / mu m_p r) dln n_e / dln r      (hydrostatic, isothermal)

The second is the reason this lane can be run at all: eFEDS publishes a measured
temperature, and hydrostatic equilibrium with a measured density profile gives an
acceleration with no halo model anywhere in it. Isothermality is an assumption,
declared, and this programme has already measured its size -- a realistic
declining T(r) moves the amplitude by 0.88-0.94.

THE GATE: reproduce the paper's own published M_gas,500 from the density
parameters. If that fails, nothing downstream is trustworthy, and the run stops.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

G = 6.674e-11
KPC = 3.0856775814913673e19
MPC = 1000.0 * KPC
MSUN = 1.98892e30
M_P = 1.67262192e-27
KEV = 1.602176634e-16
MU = 0.6           # mean molecular weight
MU_E = 1.14        # gas mass per electron, in m_p
CLIGHT = 2.99792458e8
A0 = 1.2e-10

# Bahar+2022 cosmology (flat LCDM); validated below against published Mgas500
OM, OL, H0 = 0.3, 0.7, 70.0 * 1e3 / MPC


def E(z):
    return math.sqrt(OM * (1 + z) ** 3 + OL)


def d_comoving(z, n=2048):
    zz = np.linspace(0.0, z, n)
    return (CLIGHT / H0) * np.trapezoid(1.0 / np.sqrt(OM * (1 + zz) ** 3 + OL), zz)


def d_angular(z):
    return d_comoving(z) / (1.0 + z)


# ------------------------------------------------------------------ ingest
def read_tsv(path):
    rows, hdr = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if hdr is None:
                if len(p) > 3 and p[0].strip() == "recno":
                    hdr = [x.strip() for x in p]
                continue
            if len(p) != len(hdr):
                continue
            if set("".join(p).strip()) <= set("- "):
                continue
            d = {k: v.strip() for k, v in zip(hdr, p)}
            try:
                float(d["recno"])
            except ValueError:
                continue
            rows.append(d)
    return hdr, rows


def num(d, k):
    v = d.get(k, "")
    try:
        return float(v)
    except ValueError:
        return float("nan")


# ------------------------------------------------------- the density model
def ne_of_u(u, n0sq, eps, beta, alpha):
    """n_e(r) in cm^-3 for u = r/rs. n0sq is the tabulated n0^2 in 1e-7 cm^-6."""
    u = np.maximum(u, 1e-8)
    lg = (np.log(n0sq * 1e-7)
          - alpha * np.log(u)
          + (-3.0 * beta + alpha / 2.0) * np.log1p(u ** 2)
          - (eps / 3.0) * np.log1p(u ** 3))
    return np.sqrt(np.exp(lg))


def dlnne_dlnr(u, eps, beta, alpha):
    """Analytic logarithmic slope of n_e, exact for the model above."""
    u2 = u ** 2
    u3 = u ** 3
    dlnne2 = (-alpha
              - (6.0 * beta - alpha) * u2 / (1.0 + u2)
              - eps * u3 / (1.0 + u3))
    return 0.5 * dlnne2


def profiles(rs_m, n0sq, eps, beta, alpha, r_out_m, nr=600):
    """Radial grid, gas density, enclosed gas mass, and the log slope."""
    r = np.geomspace(1e-3 * r_out_m, r_out_m, nr)
    u = r / rs_m
    ne = ne_of_u(u, n0sq, eps, beta, alpha) * 1e6        # cm^-3 -> m^-3
    rho = MU_E * M_P * ne                                # kg/m^3
    integ = 4.0 * math.pi * r ** 2 * rho
    M = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1])
                                        * np.diff(r))])
    return r, ne, rho, M, dlnne_dlnr(u, eps, beta, alpha)


def phi_from_g(r, g):
    """|Phi_b|(r) = Int_r^Rmax g dr' + g(Rmax) Rmax, the Run N/R convention."""
    seg = 0.5 * (g[1:] + g[:-1]) * np.diff(r)
    inner = np.concatenate([[0.0], np.cumsum(seg)])
    inner = inner[-1] - inner
    return inner + g[-1] * r[-1]


def main():
    h1, d1 = read_tsv(os.path.join(HERE, "efeds_bahar2022_table1_density.tsv"))
    h2, d2 = read_tsv(os.path.join(HERE, "efeds_bahar2022_table2.tsv"))
    assert len(d1) == 542, f"table1 rows {len(d1)}"
    assert len(d2) == 542, f"table2 rows {len(d2)}"
    t2 = {r["ID"]: r for r in d2}
    print("=" * 78)
    print("LEAD 01 -- eFEDS resolved group profiles, within one class")
    print("=" * 78)
    print(f"\n   ingested {len(d1)} density fits and {len(d2)} property rows")

    # ---- cuts, declared here BEFORE any residual is computed
    CUTS = {"finite_params": 0, "finite_z_R500_T": 0,
            "Tcex_frac_err_le_0.5": 0, "beta_gt_third": 0}
    recs = []
    for r1 in d1:
        cid = r1["ID"]
        r2 = t2.get(cid)
        if r2 is None:
            continue
        n0sq, rs_as = num(r1, "n0"), num(r1, "rs")
        eps, beta, alpha = num(r1, "epsilon"), num(r1, "beta"), num(r1, "alpha")
        if not all(np.isfinite([n0sq, rs_as, eps, beta, alpha])) or n0sq <= 0 \
           or rs_as <= 0:
            continue
        CUTS["finite_params"] += 1
        z, R500_am = num(r2, "z"), num(r2, "R500")
        T = num(r2, "Tcex500")
        eT = max(num(r2, "e_Tcex500"), num(r2, "E_Tcex500"))
        if not np.isfinite(T) or T <= 0:
            T, eT = num(r2, "T500"), max(num(r2, "e_T500"), num(r2, "E_T500"))
        if not all(np.isfinite([z, R500_am, T])) or z <= 0 or R500_am <= 0 \
           or T <= 0:
            continue
        CUTS["finite_z_R500_T"] += 1
        if not np.isfinite(eT) or eT / T > 0.5:
            continue
        CUTS["Tcex_frac_err_le_0.5"] += 1
        # beta <= 1/3 makes the enclosed mass diverge faster than r; the model
        # is then extrapolating rather than describing, so it is cut
        if not (beta > 1.0 / 3.0):
            continue
        CUTS["beta_gt_third"] += 1
        DA = d_angular(z)
        arcsec = math.pi / (180.0 * 3600.0)
        rs_m = rs_as * arcsec * DA
        R500_m = R500_am * 60.0 * arcsec * DA
        recs.append(dict(id=cid, z=z, T=T, eT=eT, rs=rs_m, R500=R500_m,
                         n0sq=n0sq, eps=eps, beta=beta, alpha=alpha,
                         Mgas500_pub=num(r2, "Mgas500"),
                         l_Mgas=r2.get("l_Mgas500", "").strip()))
    print("\n   cuts, declared before residuals:")
    prev = len(d1)
    for k, v in CUTS.items():
        print(f"      {k:26s} {v:4d}   (dropped {prev - v})")
        prev = v
    print(f"   RETAINED {len(recs)}")

    # ------------------------------------------------------------- THE GATE
    print("\n   GATE: reproduce the published M_gas,500 from the density fit")
    rat = []
    for R in recs:
        r, ne, rho, M, sl = profiles(R["rs"], R["n0sq"], R["eps"], R["beta"],
                                     R["alpha"], R["R500"])
        R["r"], R["ne"], R["M"], R["slope"] = r, ne, M, sl
        R["Mgas500_mine"] = M[-1] / MSUN / 1e12          # 10^12 Msun
        if R["l_Mgas"] != "<" and np.isfinite(R["Mgas500_pub"]) \
           and R["Mgas500_pub"] > 0:
            rat.append(R["Mgas500_mine"] / R["Mgas500_pub"])
    rat = np.array(rat)
    if rat.size < 20:
        print(f"      only {rat.size} systems have a non-upper-limit Mgas500 "
              "-- gate underpowered")
    med = float(np.median(rat)) if rat.size else float("nan")
    sc = float(np.std(np.log10(rat))) if rat.size else float("nan")
    print(f"      n = {rat.size},  median mine/published = {med:.4f},  "
          f"scatter {sc:.4f} dex")
    ok = rat.size >= 20 and 0.8 < med < 1.25 and sc < 0.15
    print(f"      => {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("      Stopping. The chain does not reproduce the paper's own")
        print("      gas masses, so nothing downstream can be trusted.")
        json.dump({"gate_passed": False, "median_ratio": med, "scatter": sc,
                   "n_gate": int(rat.size), "n_retained": len(recs)},
                  open(os.path.join(HERE, "lead01_results.json"), "w"), indent=1)
        return

    print("\n   (gate passed -- proceeding to the leverage measurement)")
    json.dump({"gate_passed": True, "median_ratio": med, "scatter": sc,
               "n_gate": int(rat.size), "n_retained": len(recs),
               "cuts": CUTS},
              open(os.path.join(HERE, "lead01_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
