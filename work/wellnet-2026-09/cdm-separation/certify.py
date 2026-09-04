"""certify.py -- a Stage 4 identifiability certificate for every candidate.

Method requirement of this lane's brief: obtain a Stage 4 certificate for every
candidate/statistic pair before trusting it.  Seven checks, all required, from
``work/wellnet-2026-09/stage4/certificate.py``; typed identifiers per
``prospective.py``, so no logic anywhere depends on a human-readable name.

Each candidate is certified at MORE THAN ONE amplitude on purpose.  A statistic
certified at one amplitude and refused at another has named the amplitude at
which the answer changes, which is what the brief asks for; a single verdict
would hide it.

Two wiring decisions are declared rather than buried:

  * C2's control lever is a MEASURED one, not a hypothetical normalisation
    shift.  For a new-gravity detector the control is the dark-matter universe
    (how far does a halo move this statistic?) and the target is the tensor
    amplitude range; for a CDM discriminator the two are exchanged.  Both are
    read off the arms, so C2 is answering "could the other mechanism have made
    this?" rather than "could a fitted constant have made this?".

  * C7 in the Stage 4 module is a CORRELATION between signature vectors, and a
    correlation saturates -- this programme has recorded an injected slope of
    -0.25 driving a correlation to -0.92.  Where the natural signature is an
    amplitude sequence, two mechanisms that differ by a factor of fifty in
    amplitude still correlate at 0.98.  So every signature fed to C7 here is a
    RESPONSE PATTERN ACROSS CONDITIONS (statistics, or amplitude x alignment),
    never a single monotone amplitude sequence; the correlation is then
    measuring what C7 means by it.  The module itself is used unmodified.
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "stage4")))

import certificate as C          # noqa: E402  (the Stage 4 gate itself)

RES = os.path.join(HERE, "results")
STATS = ("S_bar", "S_ext", "S_diff", "S_morph", "S_shape", "S_45")

# the amplitude at which each family becomes observable AT ALL, from Run BF's
# own scans (universes/results/E9_equivalence_at_threshold.json)
BF_THRESH_A = 0.0200293
BF_FID_A = 0.5
# projected collisionless-halo convergence ellipticities: the low end of what a
# cluster analysis would predict, and a typical one.  Declared, not fitted.
E_HALO = (0.30, 0.45)


def load(name):
    return json.load(io.open(os.path.join(RES, name), encoding="utf-8"))


def c2_measured(target_effect, control_effect, target_name, control_name):
    """C2 with a MEASURED control lever.  Typed, and with no inf ratio."""
    t, c = abs(float(target_effect)), abs(float(control_effect))
    if t <= 0.0:
        return dict(passed=False, target=t, control=c, ratio=None,
                    detail=(f"statistic INSENSITIVE to {target_name}; the "
                            f"question of whether {control_name} reproduces it "
                            f"does not arise"))
    r = c / t
    return dict(passed=r < 1.0, target=t, control=c, ratio=float(r),
                detail=(f"{control_name} reproduces {r:.3f}x the effect that "
                        f"{target_name} produces ({c:.3f} against {t:.3f})"))


def main():
    P = load("P_power.json")
    F = load("F_forward.json")
    C6 = load("C6_out_of_grammar.json")

    tscan = P["P6_tensor_scan"]
    amps = sorted(float(k) for k in tscan)

    def tfn(stat):
        ys = [tscan[k][stat]["mean"] for k in sorted(tscan, key=float)]
        return (lambda a, ys=ys: float(np.interp(a, amps, ys)))

    escan = F["F5_e_scan"]
    es = sorted(float(k) for k in escan)

    def efn(stat):
        ys = [escan[k][stat] for k in sorted(escan, key=float)]
        return (lambda e, ys=ys: float(np.interp(e, es, ys)))

    # responsiveness
    resp_A = {s: P["P6_responsiveness"][s]["slope"] for s in STATS}
    resp_e = {s: F["F5_responsiveness_vs_e"][s]["slope"] for s in STATS}
    resp_A_fw = {s: F["F5_responsiveness_vs_A"][s]["slope"] for s in STATS}
    g_resp = P["P6_responsiveness"]["G_ext"]["slope"]
    g45_resp = P["P6_G45_responsiveness"]["G_45"]["slope"]

    # noise widths: the null the alternative is tested against, measured on the
    # UNTOUCHED audit half of the generator that supplies the responsiveness
    sd_scalar = {s: P["P1_sizing"]["scalar_null"][s]["null_sd"] for s in STATS}
    sd_scalar["G_ext"] = P["P1_sizing"]["scalar_null"]["G_ext"]["null_sd"]
    sd_none_fw = {s: F["F1_arms"]["none"][s]["sd"] for s in STATS}

    # measured effect sizes on Run BF's own arms
    m = {s: {a: P["P2_rates_vs_scalar_null"][s][a]["mean"]
             for a in ("U03_mond", "U02_cdm", "U05_fid", "U10_systematics")}
         for s in STATS}
    m["G_ext"] = {a: P["P2_rates_vs_scalar_null"]["G_ext"][a]["mean"]
                  for a in ("U03_mond", "U02_cdm", "U05_fid", "U10_systematics")}

    def eff(stat, arm):
        return m[stat][arm] - m[stat]["U03_mond"]

    # C7 signatures: RESPONSE PATTERNS ACROSS THE STATISTIC SET
    sig_tensor = np.array([resp_A_fw[s] for s in STATS])
    sig_halo = np.array([resp_e[s] for s in STATS])
    sig_u10 = np.array([m[s]["U10_systematics"] for s in STATS])
    sig_sys = np.array([P["P2_rates_vs_scalar_null"][s]["U02_cdm_3xsys"]["mean"]
                        - P["P2_rates_vs_scalar_null"][s]["U02_cdm"]["mean"]
                        for s in STATS])
    nuis_for_tensor = {"triaxial collisionless halo": sig_halo,
                       "systematics-only universe": sig_u10,
                       "3x systematics increment": sig_sys}
    nuis_for_halo = {"external-axis tensor": sig_tensor,
                     "systematics-only universe": sig_u10,
                     "3x systematics increment": sig_sys}

    # galaxy C7: the pattern across AMPLITUDE x ALIGNMENT.  A tensor's m=3 is
    # locked to the external axis whatever the halo does, so its pattern is
    # flat in the alignment direction; a triaxial galaxy halo's is not.
    G = F["F6_galaxy"]
    gq = (0.05, 0.1, 0.2)
    sig_gal = np.array([G[f"tensor_q{q:g}"]["mean"] for q in gq] * 2)
    nuis_gal = {
        "disc-aligned triaxial halo":
            np.array([G[f"halo_q{q:g}_mis25_flss0"]["mean"] for q in gq]
                     + [G[f"halo_q{q:g}_mis25_flss1"]["mean"] for q in gq]),
        "isotropic triaxial halo":
            np.array([G[f"halo_q{q:g}_mis90_flss0"]["mean"] for q in gq]
                     + [G[f"halo_q{q:g}_mis25_flss0.5"]["mean"] for q in gq]),
    }

    cases = {}

    # ---------------------------------------------------- tensor, clusters
    for amp, tag in ((BF_THRESH_A, "AT_BF_THRESHOLD"), (0.1, "AT_A0.1"),
                     (0.5, "AT_FIDUCIAL"), (1.0, "AT_A1.0")):
        sz = P["P1_sizing"]["scalar_null"]["S_ext"]["nominal_0.05"]
        cases[f"CAND.TENSOR.CLUSTER_QUAD.{tag}"] = dict(
            _meta=dict(statistic="S_ext", alternative="external-axis tensor",
                       amplitude=amp, null_family="scalar / Newtonian"),
            C1_responsive=C.c1_responsive(tfn("S_ext"), np.array(amps)),
            C2_not_a_restatement=c2_measured(
                target_effect=resp_A["S_ext"] * amp,
                control_effect=eff("S_ext", "U02_cdm"),
                target_name="the tensor at this amplitude",
                control_name="a dark-matter universe (its OWN halo ellipticity)"),
            C3_exchangeable=dict(
                passed=abs(sz["realised_fpr_two_sided"]["rate"] - 0.05) < 0.02,
                detail=(f"untouched audit half: realised FPR "
                        f"{sz['realised_fpr_two_sided']['rate']:.3f} at nominal "
                        f"0.05, {P['P1_sizing']['scalar_null']['S_ext']['nominal_0.01']['realised_fpr_two_sided']['rate']:.3f} "
                        f"at nominal 0.01; null mean "
                        f"{P['P1_sizing']['scalar_null']['S_ext']['null_mean']:+.3f} "
                        f"+- {sd_scalar['S_ext']:.3f}")),
            C4_powered=C.c4_powered(abs(resp_A["S_ext"]), amp, sd_scalar["S_ext"]),
            C5_support=C.c5_support((0.20, 2.20), (0.09, 2.40)),
            C6_out_of_grammar=C.c6_out_of_grammar(
                F["F4_out_of_grammar_ring"]["0.15"]["S_ext"]["upper"]["rate"]),
            C7_nuisance_distinct=C.c7_nuisance_distinct(sig_tensor,
                                                        nuis_for_tensor))

    # ---------------------------------------------------- tensor, galaxies
    for amp, tag in ((BF_THRESH_A, "AT_BF_THRESHOLD"), (0.1, "AT_A0.1"),
                     (0.5, "AT_FIDUCIAL")):
        sz = P["P1_sizing"]["scalar_null"]["G_ext"]["nominal_0.05"]
        cases[f"CAND.TENSOR.GALAXY_M3.{tag}"] = dict(
            _meta=dict(statistic="G_ext", alternative="external-axis tensor",
                       amplitude=amp, null_family="scalar / Newtonian"),
            C1_responsive=C.c1_responsive(tfn("G_ext"), np.array(amps)),
            C2_not_a_restatement=c2_measured(
                target_effect=g_resp * amp, control_effect=g45_resp * amp,
                target_name="the tensor on the correct external axis",
                control_name="the same estimator on an axis rotated 45 degrees"),
            C3_exchangeable=dict(
                passed=abs(sz["realised_fpr_two_sided"]["rate"] - 0.05) < 0.02,
                detail=(f"untouched audit half: realised FPR "
                        f"{sz['realised_fpr_two_sided']['rate']:.3f} at nominal "
                        f"0.05; null mean "
                        f"{P['P1_sizing']['scalar_null']['G_ext']['null_mean']:+.3f} "
                        f"+- {sd_scalar['G_ext']:.3f}")),
            C4_powered=C.c4_powered(abs(g_resp), amp, sd_scalar["G_ext"]),
            C5_support=C.c5_support((1.0, 5.0), (0.0, 5.2)),
            # the INDEPENDENT galaxy model uses a different rotation-curve law
            # and a different radial turn-on for the modulation, so recovering
            # its injection is an out-of-family test
            C6_out_of_grammar=C.c6_out_of_grammar(
                G["tensor_q0.2"]["two_sided"]["rate"]),
            C7_nuisance_distinct=C.c7_nuisance_distinct(sig_gal, nuis_gal))

    # ---------------------------------------------------- CDM discriminators
    for stat, cid in (("S_bar", "CAND.CDM.BARYON_AXIS_QUAD"),
                      ("S_diff", "CAND.CDM.SIGNED_CONTRAST"),
                      ("S_morph", "CAND.CDM.MORPHOLOGY_SLOPE"),
                      ("S_shape", "CAND.CDM.RADIAL_SHAPE")):
        for e, tag in ((E_HALO[0], "AT_E0.30"), (E_HALO[1], "AT_E0.45")):
            sz = P["P1_sizing"]["cdm_null"][stat]["nominal_0.05"]
            cases[f"{cid}.{tag}"] = dict(
                _meta=dict(statistic=stat,
                           alternative="triaxial collisionless halo",
                           amplitude=e,
                           null_family="the surviving modified-gravity universes"),
                C1_responsive=C.c1_responsive(efn(stat), np.array(es)),
                C2_not_a_restatement=c2_measured(
                    target_effect=eff(stat, "U02_cdm"),
                    control_effect=eff(stat, "U05_fid"),
                    target_name="a dark-matter universe",
                    control_name="the tensor universe at its fiducial amplitude"),
                C3_exchangeable=dict(
                    passed=abs(sz["realised_fpr_two_sided"]["rate"] - 0.05) < 0.02,
                    detail=(f"untouched audit half: realised FPR "
                            f"{sz['realised_fpr_two_sided']['rate']:.3f} at "
                            f"nominal 0.05, "
                            f"{P['P1_sizing']['cdm_null'][stat]['nominal_0.01']['realised_fpr_two_sided']['rate']:.3f}"
                            f" at nominal 0.01; null mean "
                            f"{P['P1_sizing']['cdm_null'][stat]['null_mean']:+.3f}"
                            f" +- {P['P1_sizing']['cdm_null'][stat]['null_sd']:.3f}")),
                C4_powered=C.c4_powered(abs(resp_e[stat]), e, sd_none_fw[stat]),
                C5_support=C.c5_support((0.20, 2.20), (0.09, 2.40)),
                C6_out_of_grammar=C.c6_out_of_grammar(
                    C6["recovery_at_A0.1"][stat]),
                C7_nuisance_distinct=C.c7_nuisance_distinct(sig_halo,
                                                            nuis_for_halo))

    out = {"cases": {}, "n_issued": 0, "n_refused": 0,
           "wiring": {
               "C2": "measured control lever: the other mechanism's arm",
               "C4_noise_sd_tensor": "scalar-null sd on the untouched audit half",
               "C4_noise_sd_cdm": "the empty-universe arm of the independent model",
               "C7": "response pattern across the statistic set, not an "
                     "amplitude sequence (a correlation saturates)",
               "C2_cluster_caveat": (
                   "the control effect for S_ext is the dark-matter arm's mean "
                   "on Run BF's generator, -3.79.  That mean is LIBRARY "
                   "SPECIFIC: the shared 18-cluster scene library happens to "
                   "have mean cos 2(pa_bar - axis_ext) = -0.369, so a "
                   "baryon-aligned quadrupole projects onto the external axis "
                   "with a fixed non-zero coefficient in every corpus.  In the "
                   "independent forward model, which redraws both axes per "
                   "cluster, the halo's effect on S_ext is consistent with "
                   "zero and only its VARIANCE is inflated.  The stricter of "
                   "the two is used here.")}}
    for cid, spec in cases.items():
        meta = spec.pop("_meta")
        ok = C.certify(f"{cid}   [{meta['statistic']} vs {meta['alternative']}, "
                       f"amplitude {meta['amplitude']:g}]", spec)
        out["cases"][cid] = dict(meta=meta, issued=ok, checks=spec,
                                 failed=[k for k, v in spec.items()
                                         if not v["passed"]])
        out["n_issued"] += int(ok)
        out["n_refused"] += int(not ok)
    print(f"\n{out['n_issued']} certificates issued, {out['n_refused']} refused")
    p = os.path.join(RES, "C_certificates.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
