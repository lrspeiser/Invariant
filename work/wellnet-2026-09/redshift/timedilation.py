"""
THE TIME-DILATION GATE.

This runs BEFORE any fit to redshifts.  A path-dependent redshift mechanism
must stretch TIME intervals as well as photon frequencies.  If it does not, it
is excluded outright and the regression is moot.

WHAT IS MEASURED
----------------
White et al. (2024), "The Dark Energy Survey Supernova Program: slow supernovae
show cosmological time dilation out to z ~ 1", arXiv:2406.05050,
MNRAS 533, 3365, doi:10.1093/mnras/stae2008.

    1504 DES type-Ia supernovae, 0.1 <~ z <~ 1.2.
    Parameterise    dt_obs = dt_em (1 + z)^b
    Result          b = 1.003 +/- 0.005 (stat) +/- 0.010 (sys)

Note what b is and is not.  It is the ratio of the observed DURATION stretch to
the observed WAVELENGTH stretch of the same objects.  Both are observed; no
distance, no H0, no cosmology enters.  That is exactly the quantity a
path-redshift mechanism has to reproduce.

Corroboration from a population with entirely different systematics:
Lewis & Brewer (2023), Nature Astronomy 7, 1265, arXiv:2306.04053 -- 190
quasars to z = 4 as damped random walks give n = 1.28 +/- 0.29.

THE DECOMPOSITION
-----------------
Let the observed redshift factor split multiplicatively into an expansion part
and a path part,

    (1 + z) = (1 + z_exp) (1 + z_path)

and let the path mechanism stretch durations by (1 + z_path)^eta, with

    eta = 1   the mechanism is geometric: it acts on the null-geodesic affine
              structure, so frequency and time stretch together
    eta = 0   the mechanism drains photon energy but leaves the arrival-time
              spacing of successive photons alone ("tired light" of every kind:
              drag, scattering-free absorption/re-emission, photon decay,
              secular energy loss to a medium or a field)

Then

    dt_obs / dt_em = (1 + z_exp)(1 + z_path)^eta
    b              = 1 - f (1 - eta),        f = ln(1+z_path) / ln(1+z)

so the measurement constrains the PRODUCT f(1 - eta) = 1 - b, and nothing else.
It has FULL power against eta = 0 and ZERO power against eta = 1.  Saying "DES
excludes tired light" is right; saying "DES excludes path-dependent redshift"
is wrong, and this module keeps the two separate.

TRANSLATION TO THE LANE'S COEFFICIENT
-------------------------------------
The lane's law is  ln(1+z) = c1 D + c2 I_q + ... .  Over a long random sight
line the void path length is the volume-filling fraction times the distance,
I_q -> F_v D, so

    f = c2 F_v / (c1 + c2 F_v)   ~=  (c2/c1) F_v   for small c2/c1

and the DES bound on f converts directly into a bound on c2/c1 for eta = 0.

A SECOND, INDEPENDENT GATE THAT DOES REACH eta = 1
--------------------------------------------------
Derived here, not taken from a publication, and labelled as such throughout.
Any achromatic path-dependent redshift also redshifts CMB photons, so

    dT/T = -d ln(1+z) = -c2 dI_q

where dI_q is the sky-to-sky variation of the void path length in the FOREGROUND
volume that we have actually mapped.  Using only the mapped volume is
conservative: the unmapped remainder of the 14 Gpc path adds variance, it does
not cancel it.  The observed CMB anisotropy is ~1e-5 of T0, so this bounds c2
whatever eta is.  Assumptions are listed in the output JSON.

Outputs  timedilation.json  and provenance in  raw/ + manifests/ .
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
MAN = os.path.join(HERE, "manifests")
VOIDLANE = os.path.abspath(os.path.join(HERE, "..", "void-data"))

C_KMS = 299792.458
UA = "wellnet-redshift-lane/1.0 (research; leonard@horizon3.net)"


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch(url, name, note=""):
    """Download to raw/<name>, write manifests/<name>.manifest.json."""
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(MAN, exist_ok=True)
    dest = os.path.join(RAW, name)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        status = r.status
        body = r.read()
    with open(dest, "wb") as fh:
        fh.write(body)
    man = {
        "file": name,
        "source_url": url,
        "query_issued": f"HTTP GET {url}",
        "http_status": int(status),
        "retrieved_utc": utc_now(),
        "sha256": sha256_file(dest),
        "bytes": os.path.getsize(dest),
        "row_count": None,
        "columns": None,
        "notes": note,
    }
    with open(os.path.join(MAN, name + ".manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    print(f"  fetched {name}: HTTP {status}, {len(body)} bytes, "
          f"sha256 {man['sha256'][:16]}...", flush=True)
    return dest, body


def assert_in(body, needles, label):
    """Silent-extraction-failure guard: echo the identifier back."""
    txt = body.decode("utf-8", "replace")
    missing = [n for n in needles if n not in txt]
    if missing:
        raise RuntimeError(f"{label}: expected strings absent from the "
                           f"downloaded page: {missing!r}")
    print(f"  verified {label}: all {len(needles)} identifiers echoed back",
          flush=True)
    return txt


# --------------------------------------------------------------------------
# 1. acquisition
# --------------------------------------------------------------------------
def acquire():
    prov = {}
    _, b1 = fetch("https://export.arxiv.org/abs/2406.05050",
                  "arxiv_2406.05050_abs.html",
                  "DES-SN5YR time dilation, White et al. 2024, abstract page")
    t1 = assert_in(
        b1,
        ["2406.05050", "1504", "time dilation", "1.003", "0.005", "0.010"],
        "White et al. 2024 abstract")
    prov["des_time_dilation"] = {
        "arxiv": "2406.05050",
        "abstract_contains_b_value": "1.003" in t1,
        "abstract_contains_stat_err": "0.005" in t1,
        "abstract_contains_sys_err": "0.010" in t1,
        "abstract_contains_n_sne": "1504" in t1,
    }

    _, b2 = fetch("http://export.arxiv.org/api/query?id_list=2406.05050",
                  "arxiv_api_2406.05050.xml",
                  "arXiv API metadata record for the DES time-dilation paper")
    assert_in(b2, ["2406.05050", "cosmological time dilation"], "arXiv API record")

    _, b3 = fetch("https://export.arxiv.org/abs/2306.04053",
                  "arxiv_2306.04053_abs.html",
                  "Lewis & Brewer 2023 quasar time dilation, abstract page")
    assert_in(b3, ["2306.04053", "quasar"], "Lewis & Brewer 2023 abstract")

    return prov


# --------------------------------------------------------------------------
# 2. the measurements, transcribed with their sources
# --------------------------------------------------------------------------
MEASUREMENTS = {
    "DES_SN5YR_White2024": {
        "reference": "White R. M. T. et al. 2024, MNRAS 533, 3365 "
                     "(arXiv:2406.05050), doi:10.1093/mnras/stae2008",
        "probe": "type Ia supernova light-curve width vs stacked template",
        "n_objects": 1504,
        "z_range": [0.1, 1.2],
        "parameterisation": "dt_obs = dt_em (1+z)^b",
        "b": 1.003,
        "sigma_b_stat": 0.005,
        "sigma_b_sys": 0.010,
        "sigma_b_total": float(np.hypot(0.005, 0.010)),
        "note": "systematic dominates; the constraint is only as good as the "
                "DES treatment of intrinsic-width evolution and of "
                "selection on light-curve stretch",
    },
    "QSO_LewisBrewer2023": {
        "reference": "Lewis G. F. & Brewer B. J. 2023, Nature Astronomy 7, "
                     "1265 (arXiv:2306.04053), doi:10.1038/s41550-023-02029-2",
        "probe": "quasar variability as a damped random walk, SDSS + PS1",
        "n_objects": 190,
        "z_range": [0.0, 4.0],
        "parameterisation": "characteristic timescale ~ (1+z)^n",
        "b": 1.28,
        "sigma_b_stat": 0.29,
        "sigma_b_sys": 0.0,
        "sigma_b_total": 0.29,
        "note": "independent population, entirely different systematics; "
                "weaker but immune to SN-specific selection effects",
    },
}


def bound_on_f(meas, n_sigma):
    """One-sided upper bound on f(1-eta) = 1 - b at n_sigma."""
    return (1.0 - meas["b"]) + n_sigma * meas["sigma_b_total"]


# --------------------------------------------------------------------------
# 3. mechanism taxonomy, each with its eta and its verdict
# --------------------------------------------------------------------------
MECHANISMS = [
    dict(key="M1_photon_energy_drain",
         name="Photon energy drain / tired light (drag on the photon, secular "
              "energy loss to a medium or a background field, photon decay)",
         eta=0.0,
         why="The mechanism removes energy from each photon independently. "
             "Two photons emitted dt apart still arrive dt apart in the "
             "observer frame; only their individual frequencies drop.",
         maps_to="c2 I_q with any sign"),
    dict(key="M2_scattering",
         name="Scattering off a medium (Compton, Raman, plasma, 'photon-photon "
              "in a void')",
         eta=0.0,
         why="Same duration argument as M1, and every known scattering channel "
             "also randomises direction.",
         maps_to="c2 I_q, plus an unavoidable angular blur"),
    dict(key="M3_chromatic_drain",
         name="Frequency-dependent energy loss",
         eta=0.0,
         why="Same as M1 for durations, and additionally predicts a "
             "wavelength-dependent redshift, i.e. distorted spectra and "
             "band-dependent light-curve widths.",
         maps_to="c2(nu) I_q"),
    dict(key="M4_geometric_path_stretch",
         name="Geometric path stretch: the void contributes to the conformal "
              "factor along the null geodesic, so the affine spacing of "
              "successive wavefronts AND of successive events both dilate",
         eta=1.0,
         why="Frequency and time are stretched by the same factor by "
             "construction. Time dilation has NO power against this class.",
         maps_to="c2 I_q, c3 I_T, c5 I_q^2, c6 I_q I_T"),
    dict(key="M5_partial_geometric",
         name="Mixed / partially geometric mechanism",
         eta=None,
         why="Any eta strictly between 0 and 1 is constrained in the "
             "combination f(1-eta) only.",
         maps_to="c2 I_q with a free eta"),
]


def main():
    print("=" * 72)
    print("TIME-DILATION GATE")
    print("=" * 72)
    out = {"generated_utc": utc_now(),
           "lane": "work/wellnet-2026-09/redshift",
           "independence_statement":
               "This is a logically independent hypothesis. Nothing in the "
               "galaxy or cluster gravity work is evidence for or against it, "
               "and this result does not bear on that work."}

    print("\n[1] acquisition")
    out["provenance"] = acquire()
    out["measurements"] = MEASUREMENTS

    # ---- 2. the exclusion in (f, eta) --------------------------------
    print("\n[2] exclusion in the (f, eta) plane")
    des = MEASUREMENTS["DES_SN5YR_White2024"]
    qso = MEASUREMENTS["QSO_LewisBrewer2023"]

    excl = {}
    for label, m in (("DES_SN5YR", des), ("QSO", qso)):
        d = {"b": m["b"], "sigma_b_total": m["sigma_b_total"]}
        # pure energy drain carrying ALL of the redshift: f = 1, eta = 0 -> b = 0
        d["b_predicted_if_f1_eta0"] = 0.0
        d["sigma_from_b_eq_0"] = float(m["b"] / m["sigma_b_total"])
        for ns, tag in ((1.645, "95pct_onesided"), (2.0, "2sigma"),
                        (3.0, "3sigma")):
            d[f"upper_bound_on_f_times_1_minus_eta_{tag}"] = float(
                bound_on_f(m, ns))
        excl[label] = d
        print(f"  {label}: b = {m['b']:.3f} +/- {m['sigma_b_total']:.4f}; "
              f"a purely non-dilating mechanism carrying all of z is "
              f"{d['sigma_from_b_eq_0']:.1f} sigma away")
    out["exclusion_in_f_eta"] = excl

    # ---- 3. translate to c2/c1 ---------------------------------------
    print("\n[3] translation to c2/c1 for eta = 0")
    # volume filling fraction of catalogued voids along real sight lines,
    # measured in the void-data lane, both algorithms and both surveys
    Fv = {
        "DESIVAST_VoidFinder": 0.4953,   # NGC path-averaged void fraction
        "DESIVAST_REVOLVER": 0.5228,
        "SDSS_VAST_VoidFinder": 0.5881,
    }
    out["void_volume_filling_fraction_used"] = Fv
    tr = {}
    for ns, tag in ((1.645, "95pct_onesided"), (2.0, "2sigma"), (3.0, "3sigma")):
        fmax = bound_on_f(des, ns)
        row = {}
        for k, F in Fv.items():
            # f = (c2/c1) F / (1 + (c2/c1) F)  ->  c2/c1 = f / (F (1-f))
            row[k] = float(fmax / (F * (1.0 - fmax)))
        tr[tag] = row
        print(f"  {tag}: f(1-eta) < {fmax:.4f}  ->  c2/c1 < "
              + ", ".join(f"{k.split('_')[0]} {v*100:.2f}%"
                          for k, v in row.items()))
    out["c2_over_c1_upper_bound_eta0"] = tr

    # ---- 4. compare against what the redshift fit could ever reach ----
    print("\n[4] comparison with the redshift regression's own reach")
    with open(os.path.join(VOIDLANE, "results.json")) as fh:
        vres = json.load(fh)
    reach = {
        "SDSS_arm_statistical_3sigma_c2_over_c1": 0.028,
        "DESI_REVOLVER_statistical_3sigma_c2_over_c1": float(
            vres["power"]["REVOLVER"]["transverse_min_detectable_c2_over_c1_3sigma"]),
        "DESI_VoidFinder_statistical_3sigma_c2_over_c1": float(
            vres["power"]["VoidFinder"]["transverse_min_detectable_c2_over_c1_3sigma"]),
        "cross_pipeline_transverse_residual_r": 0.153,
        "note": "the statistical floors above ignore the cross-pipeline "
                "disagreement; with transverse-residual r = 0.153 between two "
                "independently built VoidFinder catalogues the realistic "
                "threshold is tens of percent, not a few percent",
    }
    td_3sig = tr["3sigma"]["SDSS_VAST_VoidFinder"]
    reach["time_dilation_3sigma_c2_over_c1_eta0"] = td_3sig
    reach["time_dilation_is_tighter_than_best_statistical_floor_at_3sigma"] = \
        bool(td_3sig < reach["SDSS_arm_statistical_3sigma_c2_over_c1"])
    reach["like_for_like_3sigma"] = {
        "time_dilation_eta0": td_3sig,
        "best_void_regression_statistical": reach[
            "SDSS_arm_statistical_3sigma_c2_over_c1"],
        "ratio_td_over_regression": float(
            td_3sig / reach["SDSS_arm_statistical_3sigma_c2_over_c1"]),
        "reading": "at matched 3-sigma confidence the two are within a factor "
                   "of two of each other, with the void regression nominally "
                   "the tighter of the two ON STATISTICS ALONE. The comparison "
                   "reverses by an order of magnitude once the cross-pipeline "
                   "systematic floor is included, because the time-dilation "
                   "bound has no such floor.",
    }
    out["comparison_with_regression_reach"] = reach
    print(f"  DES 3-sigma bound on c2/c1 (eta=0): {td_3sig*100:.2f}%")
    print(f"  best statistical 3-sigma reach of this dataset: "
          f"{reach['SDSS_arm_statistical_3sigma_c2_over_c1']*100:.2f}% "
          f"(systematic floor: tens of percent)")

    # ---- 5. the CMB smoothness gate (own derivation) ------------------
    print("\n[5] CMB smoothness gate -- reaches eta = 1 as well "
          "(own derivation, not a published limit)")
    sd_dIq = {}
    d = pd.read_csv(os.path.join(VOIDLANE, "path_integrals_analysed.csv"))
    s = pd.read_csv(os.path.join(VOIDLANE, "path_integrals_sdss.csv"))
    sd_dIq["DESIVAST_VoidFinder"] = float(d["dI_q_VoidFinder"].std())
    sd_dIq["DESIVAST_REVOLVER"] = float(d["dI_q_REVOLVER"].std())
    sd_dIq["SDSS_VAST_VoidFinder"] = float(
        s.loc[s["r_end_mpch"] >= 100.0, "dI_q_SDSS"].std())
    T0 = 2.7255
    dT_rms_K = 110e-6            # generous: total CMB rms excluding the dipole
    frac = dT_rms_K / T0
    C1_FID = 3.335641e-4          # H0/c with h = 1, per Mpc/h
    cmb = {"assumptions": [
        "the mechanism is achromatic and acts on CMB photons exactly as on "
        "supernova photons",
        "the sky-to-sky spread of foreground void path length is at least the "
        "spread measured inside the surveyed volume (conservative: the "
        "unmapped remainder of the path adds variance)",
        "no cancellation against another term of the law",
        "the whole observed CMB anisotropy is attributed to the mechanism, "
        "which is generous by roughly the ratio of the primordial to the "
        "hypothesised signal",
    ],
        "T0_K": T0, "assumed_dT_rms_K": dT_rms_K,
        "dT_over_T": float(frac),
        "sd_of_transverse_void_path_length_mpch": sd_dIq,
        "bounds": {}}
    for k, v in sd_dIq.items():
        c2max = frac / v
        cmb["bounds"][k] = {
            "c2_max_per_mpch": float(c2max),
            "c2_over_c1_max": float(c2max / C1_FID),
        }
        print(f"  {k}: sd(dI_q) = {v:.1f} Mpc/h -> |c2|/c1 < "
              f"{c2max / C1_FID * 100:.2f}%")
    cmb["applies_to_eta"] = "all eta, including eta = 1"
    cmb["sharper_test_not_done_here"] = (
        "cross-correlate the void path-length map with a CMB temperature map. "
        "That measurement already exists in the literature as void-ISW "
        "stacking and finds amplitudes of a few micro-kelvin, consistent with "
        "the general-relativistic integrated Sachs-Wolfe prediction; it would "
        "tighten the bound below by roughly the ratio of the total CMB rms to "
        "the stacked void signal.")
    out["cmb_smoothness_gate"] = cmb

    # ---- 6. verdicts --------------------------------------------------
    print("\n[6] verdicts")
    fmax_2s = bound_on_f(des, 2.0)
    verdicts = []
    for m in MECHANISMS:
        v = dict(m)
        if m["eta"] == 0.0:
            v["excluded_by_time_dilation"] = True
            v["significance_if_carrying_all_of_z_sigma"] = float(
                des["b"] / des["sigma_b_total"])
            v["max_allowed_fraction_of_ln1pz_2sigma"] = float(fmax_2s)
            v["max_allowed_c2_over_c1_2sigma"] = float(
                fmax_2s / (Fv["SDSS_VAST_VoidFinder"] * (1.0 - fmax_2s)))
            v["verdict"] = (
                "EXCLUDED as the origin of cosmological redshift at "
                f"{des['b'] / des['sigma_b_total']:.0f} sigma. Survives only "
                f"as a sub-{fmax_2s*100:.1f}% additive contamination of "
                "ln(1+z), which is below this dataset's realistic sensitivity.")
        elif m["eta"] == 1.0:
            v["excluded_by_time_dilation"] = False
            v["verdict"] = (
                "NOT constrained by time dilation -- b = 1 identically for "
                "any f. This is the only class the redshift regression can "
                "address. It is however constrained by the CMB smoothness "
                "gate above at the ~0.4% level in c2/c1, which is again "
                "tighter than this dataset's realistic sensitivity.")
        else:
            v["excluded_by_time_dilation"] = "partially"
            v["verdict"] = (
                f"Constrained only in the product f(1-eta) < {fmax_2s:.4f} "
                "(2 sigma). A mechanism with eta -> 1 escapes entirely; one "
                "with eta -> 0 is excluded unless f is tiny.")
        verdicts.append(v)
        print(f"  {m['key']}: {v['verdict'][:96]}")
    out["mechanism_verdicts"] = verdicts

    # ---- 7. the headline ----------------------------------------------
    out["headline"] = {
        "gate_passed_by": ["M4_geometric_path_stretch"],
        "gate_failed_by": ["M1_photon_energy_drain", "M2_scattering",
                           "M3_chromatic_drain"],
        "statement": (
            "Every mechanism that drains photon energy without acting on the "
            "metric is excluded outright: DES-SN5YR measures b = 1.003 +/- "
            "0.011, and such a mechanism carrying all of the redshift predicts "
            f"b = 0, which is {des['b']/des['sigma_b_total']:.0f} sigma away. "
            "As a sub-dominant additive term it is capped at "
            f"f < {fmax_2s*100:.1f}% of ln(1+z) (2 sigma), i.e. c2/c1 < "
            f"{fmax_2s/(Fv['SDSS_VAST_VoidFinder']*(1-fmax_2s))*100:.1f}%, "
            "comparable to the best STATISTICAL reach of the void data "
            "(2.8% at 3 sigma) and about an order of magnitude below its "
            "realistic systematic floor. The only surviving class is a "
            "genuinely geometric path stretch (eta = 1), against which the "
            "time-dilation test has zero power by construction -- and that "
            "class is separately capped near 0.3-0.4% in c2/c1 by the "
            "smoothness of the CMB across the local void structure, roughly "
            "an order of magnitude tighter than the void regression's best "
            "statistical reach. THE REDSHIFT FIT IS THEREFORE A BOUNDED "
            "FEASIBILITY STUDY, NOT A DISCOVERY TEST."),
    }

    with open(os.path.join(HERE, "timedilation.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote timedilation.json")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
