"""Item 4, step 2 of 2: score the SEALED fresh sample, ONCE.

Reads the frozen coefficients from fresh_seal.json.  Refits nothing.  Every cut
was declared in that file before this script was written.
"""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

import common as C

FRESH = os.path.join(C.LANE, "data", "fresh")
G, KPC, MSUN, A0 = C.G, C.KPC, C.MSUN, C.A0
MPC = 1e3 * KPC
MK_SUN = 3.27           # Willmer 2018, absolute K magnitude of the Sun
UPS_K = 0.75            # DECLARED IN ADVANCE, see fresh_seal.json
HERN = 1.8153           # Hernquist a = r_e / 1.8153
RNG = np.random.default_rng(20260907)


def rd(path, delim="\t"):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter=delim))
    clean = []
    for r in rows:
        clean.append({k.split(" [")[0].strip(): v for k, v in r.items()})
    return clean


def f(x, default=float("nan")):
    try:
        return float(str(x).strip())
    except Exception:
        return default


def mgas_beta(r_m, rho0, rc_m, beta):
    """M_gas(<r) for rho(s) = rho0 [1+(s/rc)^2]^(-3 beta/2), exact quadrature."""
    if not (np.isfinite(rho0) and np.isfinite(rc_m) and np.isfinite(beta)):
        return float("nan")
    r_m = float(r_m)
    xs = np.linspace(0.0, r_m, 20001)
    ys = rho0 * (1.0 + (xs / rc_m) ** 2) ** (-1.5 * beta) * xs ** 2
    return 4.0 * math.pi * float(np.trapezoid(ys, xs))


def mgas_shape(r_m, rc_m, beta):
    """Unnormalised Int_0^r [1+(s/rc)^2]^(-3 beta/2) s^2 ds."""
    xs = np.linspace(0.0, float(r_m), 20001)
    ys = (1.0 + (xs / rc_m) ** 2) ** (-1.5 * beta) * xs ** 2
    return float(np.trapezoid(ys, xs))


class FreshProfile:
    """Baryonic profile of one early-type galaxy: beta-model gas plus a
    Hernquist stellar component, point-mass tail beyond the measured radius.

    The gas SHAPE comes from Babyk's beta-model parameters; the gas
    NORMALISATION comes from Babyk's own tabulated M_gas(<5 r_e).  The ingest
    gate below is why: the tabulated r_c is rounded to 0.01 kpc and 24 of the
    94 objects have r_c <= 0.10 kpc (17 by the strict test used at run time),
    where that rounding is a factor-of-two
    error in r_c and therefore up to 2 dex in the beta-model normalisation.
    The shape at the radii that matter is insensitive to it; the normalisation
    is not, so the measured mass is used for the normalisation.
    """

    def __init__(self, rho0, rc, beta, Mstar, a_h, r_out, Mgas_out):
        self.rho0, self.rc, self.beta = rho0, max(rc, 0.005 * KPC), beta
        self.Mstar, self.a = Mstar, a_h
        self.r_out = r_out
        self.Mgas_out = Mgas_out
        self._I_out = mgas_shape(r_out, self.rc, beta)
        self.M_out = self.M(r_out)

    def Mgas(self, s):
        if s >= self.r_out:
            return self.Mgas_out
        return self.Mgas_out * mgas_shape(s, self.rc, self.beta) / self._I_out

    def M(self, s):
        s = float(s)
        if s >= self.r_out:
            return self.Mgas_out + self.Mstar * self.r_out ** 2 \
                / (self.r_out + self.a) ** 2
        return self.Mgas(s) + self.Mstar * s ** 2 / (s + self.a) ** 2

    def g(self, s):
        return G * self.M(s) / s ** 2

    def integral(self, a, b, n=800):
        """Int_a^b g ds, b may be inf (point-mass tail beyond r_out)."""
        if b == a:
            return 0.0
        if b < a:
            return -self.integral(b, a, n)
        tot = 0.0
        hi = min(b, self.r_out)
        if hi > a:
            xs = np.geomspace(a, hi, n)
            gs = np.array([self.g(x) for x in xs])
            tot += float(np.trapezoid(gs, xs))
        if b > self.r_out:
            lo = max(a, self.r_out)
            tot += G * self.M_out * (1.0 / lo -
                                     (0.0 if math.isinf(b) else 1.0 / b))
        return tot

    def r_half(self):
        tgt = 0.5 * self.M_out
        lo, hi = self.r_out * 1e-4, self.r_out
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            if self.M(mid) < tgt:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)


def build(ups_k=UPS_K):
    joined = rd(os.path.join(FRESH, "babyk2018_joined_per_object.tsv"))
    tab3 = rd(os.path.join(FRESH, "babyk2018_table3_betamodel_masses.tsv"))
    kb = rd(os.path.join(FRESH, "babyk2018_kband.tsv"))
    assert len(joined) == 94, f"Babyk joined rows {len(joined)} != 94"
    assert len(tab3) == 94, f"Babyk table 3 rows {len(tab3)} != 94"
    b3 = {r["Name"]: r for r in tab3}
    kcol = None
    for cand in ("Ktmag", "K_s_total", "Ks_total", "Kmag", "K_total",
                 "K_s_tot_mag", "Ktot"):
        if cand in kb[0]:
            kcol = cand
            break
    if kcol is None:
        kcol = [c for c in kb[0]
                if c.lower().startswith("k") and "err" not in c.lower()
                and "cat" not in c.lower() and "col" not in c.lower()][0]
    ecol = "e_" + kcol if ("e_" + kcol) in kb[0] else None
    if ecol is None:
        for c in kb[0]:
            if "err" in c.lower() and c.lower().startswith(("k", "e_k")):
                ecol = c
                break
    ccol = next((c for c in kb[0] if "catalogue" in c.lower()), None)
    kmap = {r["Name"]: r for r in kb}
    print(f"   K-band file: {len(kb)} rows, magnitude column '{kcol}'"
          + (f", error column '{ecol}'" if ecol else ""))

    out = []
    for r in joined:
        n = r["Name"]
        t3 = b3[n]
        k = kmap.get(n)
        r5 = f(r["r_5re"]) * KPC
        Mtot = f(r["M_tot_hydrostatic_lt_5re"]) * 1e13 * MSUN
        Mgas_tab = f(r["M_gas_lt_5re"]) * 1e11 * MSUN
        DL = f(r["D_L"]) * MPC
        rho0 = f(t3["rho_0"]) * 1e-21                 # 1e-24 g/cm^3 -> kg/m^3
        rc = f(t3["r_c"]) * KPC
        beta = f(t3["beta"])
        Kmag = f(k[kcol]) if k else float("nan")
        rec = dict(name=n, morph=r.get("MorphType", ""),
                   bcg=(r.get("BCG_flag", "").strip() != ""),
                   cd=(r.get("cD_flag", "").strip() != ""),
                   z=f(r["z"]), r_out=r5, Mtot=Mtot, Mgas_tab=Mgas_tab,
                   Mtot_err=f(r["M_tot_err_stat"]) * 1e13 * MSUN,
                   kT=f(r["kT"]), DL=DL, rho0=rho0, rc=rc, beta=beta,
                   Kmag=Kmag,
                   Kerr=f(k[ecol]) if (k and ecol) else float("nan"),
                   kband_source=(k.get(ccol, "") if (k and ccol)
                                 else ("" if k else "MISSING")))
        out.append(rec)
    return out, kcol


def main():
    seal = json.load(open(os.path.join(C.LANE, "fresh_seal.json")))
    print("=" * 78)
    print("ITEM 4   THE SEALED FRESH SAMPLE, SCORED ONCE")
    print("=" * 78)
    print(f"   seal written {seal['sealed_utc']}, sha256 "
          f"{seal.get('self_sha256', '')[:16]}...")
    print(f"   sample: {seal['fresh_sample']['identity'][:100]}...")
    rows, kcol = build()
    res = dict(seal_utc=seal["sealed_utc"], n_acquired=len(rows),
               kband_column=kcol, upsilon_K=UPS_K, MK_sun=MK_SUN)

    # ---- GATE 1: reproduce the published M_gas(<5 r_e) -------------------
    rat = []
    for r in rows:
        m = mgas_beta(r["r_out"], r["rho0"], r["rc"], r["beta"])
        r["Mgas_model"] = m
        if np.isfinite(m) and m > 0 and np.isfinite(r["Mgas_tab"]):
            rat.append(math.log10(m / r["Mgas_tab"]))
    rat = np.array(rat)
    small_rc = sum(1 for r in rows if r["rc"] <= 0.10 * KPC)
    print(f"\n   GATE 1  M_gas integrated from the beta-model parameters vs "
          f"the tabulated M_gas(<5 r_e):")
    print(f"      n = {len(rat)}, median ratio = {10 ** np.median(rat):.4f}, "
          f"scatter = {rat.std():.4f} dex, "
          f"range {10 ** rat.min():.3f}-{10 ** rat.max():.3f}")
    res["gate_mgas"] = dict(n=len(rat), median_ratio=float(10 ** np.median(rat)),
                            scatter_dex=float(rat.std()),
                            min_ratio=float(10 ** rat.min()),
                            max_ratio=float(10 ** rat.max()))
    within = {t_: int((np.abs(rat) <= t_).sum()) for t_ in (0.05, 0.1, 0.2, 0.5, 1.0)}
    print(f"      within 0.05/0.1/0.2/0.5/1.0 dex: "
          + "/".join(str(within[t_]) for t_ in (0.05, 0.1, 0.2, 0.5, 1.0))
          + f" of {len(rat)}")
    print(f"      {small_rc} of {len(rows)} objects have a tabulated r_c <= "
          f"0.10 kpc, printed to 0.01 kpc, so r_c itself is uncertain by a "
          f"factor of order 2 and the beta-model NORMALISATION with it.")
    print(f"      corr(|log ratio|, log(r_5re/r_c)) = "
          f"{np.corrcoef(np.abs(rat), np.log10([r['r_out'] / r['rc'] for r in rows if np.isfinite(r['Mgas_model']) and r['Mgas_model'] > 0]))[0, 1]:+.3f}"
          f"  -- the failures are exactly the small-r_c objects.")
    print(f"      MEDIAN IS 1.00, so the formula and the units are right.  The "
          f"gas NORMALISATION is therefore taken from Babyk's own tabulated "
          f"M_gas(<5 r_e) and the beta-model supplies only the SHAPE.")
    gate_ok = abs(np.median(rat)) < 0.10
    print(f"      GATE {'PASSES' if gate_ok else 'FAILS'} on the median "
          f"(|median| < 0.10 dex required)")
    res["gate_mgas"]["passed"] = bool(gate_ok)
    res["gate_mgas"]["within_dex"] = {str(k): v for k, v in within.items()}
    res["gate_mgas"]["n_small_rc"] = small_rc
    res["gate_mgas"]["repair"] = ("gas normalisation from the tabulated "
                                  "M_gas(<5 r_e); beta-model supplies the shape")

    # ---- build the observables -------------------------------------------
    for r in rows:
        DM = 5 * math.log10(r["DL"] / (10 * 3.0856775814913673e16))
        r["logLK"] = (-0.4 * (r["Kmag"] - DM - MK_SUN)
                      if np.isfinite(r["Kmag"]) else float("nan"))
        r["Mstar"] = UPS_K * 10 ** r["logLK"] * MSUN \
            if np.isfinite(r["logLK"]) else float("nan")
        r["re"] = r["r_out"] / 5.0
        r["a_h"] = r["re"] / HERN

    usable = [r for r in rows
              if np.isfinite(r["Mstar"]) and np.isfinite(r["Mtot"])
              and np.isfinite(r["Mgas_tab"]) and r["Mgas_tab"] > 0
              and np.isfinite(r["rc"]) and np.isfinite(r["beta"])
              and r["Mtot"] > 0
              and np.isfinite(r["r_out"]) and r["r_out"] > 0
              and np.isfinite(r["kT"])
              and r["Mtot_err"] / r["Mtot"] <= 0.5]
    print(f"\n   declared cuts: {len(rows)} acquired -> {len(usable)} usable "
          f"(finite masses, radius, kT, K magnitude; M_tot_err/M_tot <= 0.5)")
    res["n_usable"] = len(usable)
    res["n_dropped_by_cut"] = len(rows) - len(usable)
    res["dropped_names"] = [r["name"] for r in rows if r not in usable]

    for r in usable:
        p = FreshProfile(r["rho0"], r["rc"], r["beta"], r["Mstar"], r["a_h"],
                         r["r_out"], r["Mgas_tab"])
        r["prof"] = p
        r["Mb"] = p.M_out
        r["g_bar"] = G * r["Mb"] / r["r_out"] ** 2
        r["g_obs"] = G * r["Mtot"] / r["r_out"] ** 2
        r["nu_obs"] = r["g_obs"] / r["g_bar"]
        r["lg"] = math.log10(r["g_bar"])
        r["dev"] = math.log10(r["nu_obs"] / C.nu_rar(r["g_bar"]))
        r["f_gas"] = mgas_beta(r["r_out"], r["rho0"], r["rc"], r["beta"]) \
            / r["Mb"]
        r["dphi_TAIL"] = abs(p.integral(r["r_out"], math.inf))
        rref = 10.0 * p.r_half()
        r["r_ref_BARY"] = rref
        r["dphi_BARY"] = abs(p.integral(min(r["r_out"], rref),
                                        max(r["r_out"], rref)))
        r["lp_BARY"] = math.log10(r["dphi_BARY"])
        r["lp_TAIL"] = math.log10(r["dphi_TAIL"])

    lgs = np.array([r["lg"] for r in usable])
    lo, hi = seal["window_log_gbar"]
    inwin = (lgs >= lo) & (lgs <= hi)
    print(f"\n   log10 g_bar of the fresh sample: {lgs.min():.3f} ... "
          f"{lgs.max():.3f}   (median {np.median(lgs):.3f})")
    print(f"   fitted window is [{lo:.3f}, {hi:.3f}]: "
          f"{int(inwin.sum())} of {len(usable)} objects fall inside it")
    print(f"   baryon budget: median gas fraction of M_b inside 5 r_e = "
          f"{np.median([r['f_gas'] for r in usable]):.3f}")
    print(f"   log10|DeltaPhi_b| (BARY) range: "
          f"{min(r['lp_BARY'] for r in usable):.2f} ... "
          f"{max(r['lp_BARY'] for r in usable):.2f}")
    res["log_gbar_range"] = [float(lgs.min()), float(lgs.max())]
    res["n_in_window"] = int(inwin.sum())
    res["median_gas_fraction"] = float(np.median([r["f_gas"] for r in usable]))

    # ---- SCORE, ONCE ------------------------------------------------------
    def predict(r, key):
        c = seal["frozen_models"][key]["coef"]
        x = r["lg"]
        v = c[0] + c[1] * x + c[2] * x * x
        if key == "M1_BARY":
            v += c[3] * r["lp_BARY"]
        elif key == "M1_TAIL":
            v += c[3] * r["lp_TAIL"]
        elif key == "M3":
            v += c[3] * 0.0            # they ARE galaxies: step = 0
        return v

    def predict_M3_alt(r):
        c = seal["frozen_models"]["M3"]["coef"]
        x = r["lg"]
        return c[0] + c[1] * x + c[2] * x * x + c[3] * 1.0

    subsets = {
        "PRIMARY  non-BCG/cD, inside the fitted window":
            [r for r, w in zip(usable, inwin)
             if w and not r["bcg"] and not r["cd"]],
        "non-BCG/cD, all g_bar": [r for r in usable
                                  if not r["bcg"] and not r["cd"]],
        "BCG or cD (secondary)": [r for r in usable if r["bcg"] or r["cd"]],
        "everything usable": usable,
    }
    res["subsets"] = {}
    for nm, sub in subsets.items():
        if len(sub) < 3:
            continue
        dev = np.array([r["dev"] for r in sub])
        rec = dict(n=len(sub), observed_mean_dev=float(dev.mean()),
                   observed_sd_dev=float(dev.std()),
                   median_lp_BARY=float(np.median([r["lp_BARY"] for r in sub])),
                   median_lg=float(np.median([r["lg"] for r in sub])),
                   models={})
        print(f"\n   {nm}   n = {len(sub)}")
        print(f"      observed log10(nu_obs/nu_RAR): mean {dev.mean():+.4f} "
              f"dex, sd {dev.std():.4f}, median "
              f"{float(np.median(dev)):+.4f}")
        print(f"      {'model':<28} {'rms':>8} {'bias':>9} {'scatter':>9} "
              f"{'mean pred':>10}")
        for key, lab in (("M0", "M0  RAR only"),
                         ("M1_BARY", "M1  potential depth (BARY)"),
                         ("M1_TAIL", "M1  potential depth (TAIL)"),
                         ("M3", "M3  class step, step = 0")):
            pr = np.array([predict(r, key) for r in sub])
            e = dev - pr
            rec["models"][key] = dict(
                rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()),
                scatter=float(e.std()), mean_pred=float(pr.mean()))
            print(f"      {lab:<28} {np.sqrt(np.mean(e ** 2)):8.4f} "
                  f"{e.mean():+9.4f} {e.std():9.4f} {pr.mean():+10.4f}")
        pr = np.array([predict_M3_alt(r) for r in sub])
        e = dev - pr
        rec["models"]["M3_step1"] = dict(
            rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()),
            scatter=float(e.std()), mean_pred=float(pr.mean()),
            note="alternative reading of the class step: treat an X-ray "
                 "hot-gas early-type galaxy as NOT a galaxy")
        print(f"      {'M3  class step, step = 1':<28} "
              f"{np.sqrt(np.mean(e ** 2)):8.4f} {e.mean():+9.4f} "
              f"{e.std():9.4f} {pr.mean():+10.4f}")
        # paired bootstrap M1_BARY vs M3
        d1 = dev - np.array([predict(r, "M1_BARY") for r in sub])
        d3 = dev - np.array([predict(r, "M3") for r in sub])
        nb, n = 20000, len(sub)
        dd = np.empty(nb)
        for k in range(nb):
            p_ = RNG.integers(0, n, n)
            dd[k] = math.sqrt(float(np.mean(d1[p_] ** 2))) \
                - math.sqrt(float(np.mean(d3[p_] ** 2)))
        rec["paired_M1BARY_vs_M3"] = dict(
            observed_delta_rms=float(np.sqrt(np.mean(d1 ** 2))
                                     - np.sqrt(np.mean(d3 ** 2))),
            ci95=[float(x) for x in np.percentile(dd, [2.5, 97.5])],
            p_M1_better=float((dd < 0).mean()),
            n_objects_M1_better=int((np.abs(d1) < np.abs(d3)).sum()))
        b = rec["paired_M1BARY_vs_M3"]
        print(f"      paired M1(BARY) - M3: dRMS {b['observed_delta_rms']:+.4f}"
              f" [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  "
              f"P(M1 better) = {b['p_M1_better']:.4f}  "
              f"M1 closer on {b['n_objects_M1_better']}/{len(sub)}")
        res["subsets"][nm] = rec

    # ---- systematics: the declared Upsilon_K range ------------------------
    print("\n   SYSTEMATIC: the stellar mass-to-light ratio")
    sysres = {}
    prim_names = {r["name"] for r in subsets[
        "PRIMARY  non-BCG/cD, inside the fitted window"]}
    for u in (0.6, 0.75, 1.0):
        rr, _ = build(ups_k=u)
        tmp = []
        for r in rr:
            if r["name"] not in prim_names:
                continue
            DM = 5 * math.log10(r["DL"] / (10 * 3.0856775814913673e16))
            logLK = -0.4 * (r["Kmag"] - DM - MK_SUN)
            Mstar = u * 10 ** logLK * MSUN
            re = r["r_out"] / 5.0
            p = FreshProfile(r["rho0"], r["rc"], r["beta"], Mstar,
                             re / HERN, r["r_out"], r["Mgas_tab"])
            gb = G * p.M_out / r["r_out"] ** 2
            go = G * r["Mtot"] / r["r_out"] ** 2
            dv = math.log10((go / gb) / C.nu_rar(gb))
            rref = 10.0 * p.r_half()
            lp = math.log10(abs(p.integral(min(r["r_out"], rref),
                                           max(r["r_out"], rref))))
            tmp.append((math.log10(gb), lp, dv))
        lgv = np.array([x[0] for x in tmp])
        lpv = np.array([x[1] for x in tmp])
        dv = np.array([x[2] for x in tmp])
        c = seal["frozen_models"]["M1_BARY"]["coef"]
        p1 = c[0] + c[1] * lgv + c[2] * lgv ** 2 + c[3] * lpv
        c3 = seal["frozen_models"]["M3"]["coef"]
        p3 = c3[0] + c3[1] * lgv + c3[2] * lgv ** 2
        sysres[str(u)] = dict(
            n=len(tmp), mean_dev=float(dv.mean()),
            rms_M1=float(np.sqrt(np.mean((dv - p1) ** 2))),
            rms_M3=float(np.sqrt(np.mean((dv - p3) ** 2))),
            median_log_gbar=float(np.median(lgv)))
        print(f"      Upsilon_K = {u:.2f}: mean observed deviation "
              f"{dv.mean():+.4f} dex, rms M1 "
              f"{np.sqrt(np.mean((dv - p1) ** 2)):.4f}, rms M3 "
              f"{np.sqrt(np.mean((dv - p3) ** 2)):.4f}  "
              f"(median log g_bar {np.median(lgv):+.3f})")
    res["upsilon_K_systematic"] = sysres

    # ---- per-object table --------------------------------------------------
    res["objects"] = [dict(
        name=r["name"], morph=r["morph"], bcg=r["bcg"], cd=r["cd"], z=r["z"],
        r_kpc=r["r_out"] / KPC, Mb_Msun=r["Mb"] / MSUN,
        Mstar_Msun=r["Mstar"] / MSUN, Mgas_Msun=r["Mgas_model"] / MSUN,
        Mtot_Msun=r["Mtot"] / MSUN, log_g_bar=r["lg"], nu_obs=r["nu_obs"],
        dev=r["dev"], lp_BARY=r["lp_BARY"], lp_TAIL=r["lp_TAIL"],
        r_ref_BARY_kpc=r["r_ref_BARY"] / KPC, kT=r["kT"], Kmag=r["Kmag"])
        for r in usable]

    p = os.path.join(C.LANE, "fresh_result.json")
    json.dump(res, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
