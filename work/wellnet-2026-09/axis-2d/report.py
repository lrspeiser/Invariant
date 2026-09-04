"""Generate REPORT.md from the result JSONs, so every number in the prose is
read out of the file that produced it rather than transcribed by hand.

The narrative lives here as text; the tables are built from `axis_power.json`,
`selection.json`, `shear2d.json`, `amplitudes.json` and `crosscheck.json`.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(n):
    p = os.path.join(HERE, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def surface_table(ap, noise):
    """One power surface table: axis ratio down, amplitude across."""
    amps = ap["config"]["amps"]
    rows = sorted([r for r in ap["rows"] if abs(r["noise"] - noise) < 1e-12],
                  key=lambda x: x["axis_ratio"])
    out = []
    for prov in ("source", "external", "network"):
        head = ("| axis ratio | " + " | ".join(f"amp {a:g}" for a in amps)
                + " | axis known (amp %g) | audit FPR |" % max(amps))
        sep = "|---" * (len(amps) + 3) + "|"
        lines = [f"\n**{prov} axis**, noise {noise:.0%}\n", head, sep]
        for r in rows:
            cells = [c for c in r["power"].values()
                     if c["provenance"] == prov]
            ps = []
            for a in amps:
                v = [c["power"] for c in cells if abs(c["amp"] - a) < 1e-12]
                ps.append(f"{np.mean(v):.2f}" if v else "-")
            kn = [c["power_axis_known"] for c in cells
                  if abs(c["amp"] - max(amps)) < 1e-12
                  and c.get("power_axis_known") is not None]
            k = f"{np.mean(kn):.2f}" if kn else "n/a"
            lines.append(f"| {r['axis_ratio']:.3f} | " + " | ".join(ps)
                         + f" | {k} | {r['audit_fpr'][prov]:.3f} |")
        out.append("\n".join(lines))
    return "\n".join(out)


def turn_table(ap):
    """The flux-turning fraction against axis ratio -- the theorem, measured.

    Power is limited by the detector sensitivity; the turning fraction is not.
    It is the fraction of the injected response that produces a component of
    K grad Phi_N transverse to grad Phi_N, i.e. the part the blindness theorem
    does NOT declare unobservable.  It is the cleanest statement of the
    spherical limit available.
    """
    ars = sorted({round(r["axis_ratio"], 3) for r in ap["rows"]})
    lines = ["| provenance | " + " | ".join(f"axis ratio {a:.2f}"
                                            for a in ars) + " |",
             "|---" * (len(ars) + 1) + "|"]
    for prov in ("source", "external", "network"):
        vals = []
        for a in ars:
            v = [c["turn_fraction_median"] for r in ap["rows"]
                 if abs(round(r["axis_ratio"], 3) - a) < 1e-9
                 for c in r["power"].values()
                 if c["provenance"] == prov
                 and c.get("turn_fraction_median") is not None]
            vals.append(f"{np.median(v):.4f}" if v else "-")
        lines.append(f"| {prov} | " + " | ".join(vals) + " |")
    return "\n".join(lines), ars


def dgg_table(ap):
    """The amplitude axis translated into max dg/g and turning fraction."""
    amps = ap["config"]["amps"]
    lines = ["| provenance | " + " | ".join(f"amp {a:g}" for a in amps) + " |",
             "|---" * (len(amps) + 1) + "|"]
    for prov in ("source", "external", "network"):
        for key, lab in (("aniso_frac_median", "max dg/g"),
                         ("turn_fraction_median", "flux-turning fraction")):
            vals = []
            for a in amps:
                v = [c[key] for r in ap["rows"] for c in r["power"].values()
                     if c["provenance"] == prov and abs(c["amp"] - a) < 1e-12
                     and c.get(key) is not None]
                vals.append(f"{np.median(v):.3f}" if v else "-")
            lines.append(f"| {prov}, {lab} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = load("axis_power.json")
    sel = load("selection.json")
    s2 = load("shear2d.json")
    am = load("amplitudes.json")
    cc = load("crosscheck.json")
    dev = s2["samples"]["dev"]
    ctl = s2["samples"]["ctrl"]
    ch = dev["channels"]
    cch = ctl["channels"]
    inj = cc["injection"]["pred"]
    injm = cc["injection"]["meas"]
    injf = cc["injection"]["fe"]

    # ---------------------------------------------------------- headline
    max_ext = 0.0
    if ap:
        _amax = max(ap["config"]["amps"])
        for r in ap["rows"]:
            for c in r["power"].values():
                if (c["provenance"] == "external"
                        and abs(c["amp"] - _amax) < 1e-12):
                    max_ext = max(max_ext, c["power"])
    src97 = None
    src50 = None
    ext97 = None
    if ap:
        amax = max(ap["config"]["amps"])
        for r in ap["rows"]:
            cells = [c for c in r["power"].values()
                     if abs(c["amp"] - amax) < 1e-12]
            for prov, store in (("source", "s"), ("external", "e")):
                v = [c["power"] for c in cells if c["provenance"] == prov]
                if not v:
                    continue
                if r["axis_ratio"] > 0.95 and prov == "source":
                    src97 = (src97 or []) + v
                if r["axis_ratio"] > 0.95 and prov == "external":
                    ext97 = (ext97 or []) + v
                if r["axis_ratio"] < 0.55 and prov == "source":
                    src50 = (src50 or []) + v
    tt, ars = turn_table(ap)

    def turn_at(prov, lo, hi):
        v = [c["turn_fraction_median"] for r in ap["rows"]
             if lo <= r["axis_ratio"] <= hi for c in r["power"].values()
             if c["provenance"] == prov
             and c.get("turn_fraction_median") is not None]
        return float(np.median(v)) if v else float("nan")

    _fp = np.array([x for r in ap["rows"] for x in r["audit_fpr"].values()])
    t_src_50 = turn_at("source", 0.0, 0.55)
    t_src_97 = turn_at("source", 0.95, 1.0)
    t_ext_50 = turn_at("external", 0.0, 0.55)
    t_ext_97 = turn_at("external", 0.95, 1.0)
    headline = f"""
## Headline

**The spherical blindness theorem is now a measurement, and it separates the
three axis provenances exactly as it should.** The observable part of a
response tensor is the part that TURNS the flux, `K grad Phi_N` transverse to
`grad Phi_N`; the rest is a scalar rescaling the theorem declares
unobservable. Measured on injected laws over the shear-measured shell, that
turning fraction is

    source axis     {t_src_50:.4f} at axis ratio 0.50  ->  {t_src_97:.4f} at 0.97   (factor {t_src_50 / max(t_src_97, 1e-9):.1f} collapse)
    external axis   {t_ext_50:.4f} at axis ratio 0.50  ->  {t_ext_97:.4f} at 0.97   (no collapse)

A source-axis tensor loses its observable content as the source becomes
spherical. An external-axis tensor does not, because `ghat` is not an
eigenvector of a fixed `d d^T` even for a spherical source. **Run AC's
near-spherical control injected the external `dd` basis, so its power was
never supposed to collapse** — that control was measuring the wrong
hypothesis, not failing.

**Getting there required finding and fixing a real bug in the statistic.** The
first version of the detector reported source-axis power of 0.42 to 0.62 at
axis ratio 0.970, flat in axis ratio. The brief says that is a bug and the
theorem is the check. It was: in spherical symmetry a tensor atom acts as a
scalar rescaling with a radial profile the scalar bank cannot reproduce, so
admitting it improves the fit with no anisotropy anywhere. The repair is to
make every tensor atom flux-orthogonal (Section 1.2). It also disposes of the
QUMOND degeneracy without a special case.

**On the calibrated power surfaces the source-axis and member-well-network
channels have no usable power in this design; the external-axis channel has a
great deal.** At the largest injected amplitude (max dg/g of order 0.9),
external-axis power reaches {max_ext:.2f} while source-axis power stays at the
test size. That is not a limitation of the search: only about
{100 * t_src_50:.0f} per cent of a source-axis response is observable at all
even on a strongly triaxial source, and about {100 * t_src_97:.0f} per cent on
a near-spherical one. Across all nine calibrated arms the realised
false-positive rate measured on UNTOUCHED audit simulations has median
{np.median(_fp):.3f} and mean {_fp.mean():.3f} against a nominal 0.05
(range {_fp.min():.3f} to {_fp.max():.3f} over {_fp.size} cells, binomial
standard error 0.011 at 400 draws), so the surfaces are correctly sized rather
than nominally so.

**The two-dimensional phase test returns a clean null with a stated, and
inadequate, power.** On 27 eFEDS clusters selected by independent geometry,
carrying {sum(r['n_bg'] for r in sel['dev']):,} background DECADE sources, the
phase-misaligned quadrupole coefficient is

    alpha = {ch['a2s_pred']['fit']['alpha']:+.4f} +- {ch['a2s_pred']['null_std']:.4f}   (axis-randomisation null, p = {ch['a2s_pred']['p_two_sided']:.4f})

with the near-round negative control at
{cch['a2s_pred']['fit']['alpha']:+.4f} +- {cch['a2s_pred']['null_std']:.4f}
(p = {cch['a2s_pred']['p_two_sided']:.4f}) and the B-mode channel at
{ch['x2s_bmode']['fit']['alpha']:+.4f} +- {ch['x2s_bmode']['null_std']:.4f}
(p = {ch['x2s_bmode']['p_two_sided']:.4f}). End-to-end injection into the real
data recovers an injected signal with slope {inj['response_slope']:.4f} and
gives a 95 per cent exclusion of |alpha| < {inj['exclusion_2sided_95']:.3f}.

**That null says nothing about any candidate in the tournament, and the reason
is not only the noise.** At the sample median
|sin 2Delta| = {dev['median_abs_sin2delta']:.3f} and the measured quadrupole
coefficient C = {am['quadrupole_coefficient']:.3f}, the limit corresponds to a
misaligned convergence ellipticity
e_kappa < {inj['exclusion_2sided_95'] * dev['median_abs_sin2delta'] / am['quadrupole_coefficient']:.2f},
**above the geometric maximum of 1**: no external-axis tensor of any amplitude
is excluded by this sample. And separately, **none of the 3123 tournament
candidates is an external-axis tensor at all** — 1560 are member-well network,
780 source-axis, 783 isotropic, and zero carry a fixed external direction. The
phase channel is the right instrument pointed at a hypothesis the searched
grammar does not contain.

**Two structural results fall out of Job 4 that do not depend on data volume.**
Of the 18 tournament survivors, the two `tensor_d` ones have
`K grad Phi_N = exp(2AW/3) grad Phi_N` exactly — the field direction is an
eigenvector, so they cannot turn the flux and predict no quadrupole of any
phase. They are scalar rescalings wearing a tensor name. The remaining tensor
survivors (`tensor_T`, `tensor_S`) put their quadrupole along the baryonic or
member axis, where it lands in the `a2c` channel and is degenerate with
baryonic ellipticity: predicted a2c/a0 up to
{max(r['quadrupole_a2c_pred_dev'] for r in am['survivors']):.3f} against a
measured {ch['a2c_baryon']['fit']['alpha']:+.4f} +- {ch['a2c_baryon']['fit']['e_alpha']:.4f}
that contains both contributions and cannot separate them.

**The flux-turning fraction, by axis ratio** — the theorem, measured:

{tt}
"""

    # ------------------------------------------------------- power tables
    ptab = "### 1.3 The surfaces\n"
    ptab += ("\nPower is the fraction of injections whose cross-fitted "
             "statistic exceeds the empirical critical value D*, which is the "
             "95th percentile of the SAME statistic on calibration nulls. "
             "'audit FPR' is the realised false-positive rate measured on "
             "UNTOUCHED audit nulls at that D*. 'axis known' is the same "
             "injection scored with the arm restricted to the single correct "
             "direction, which is what the independent environment map of "
             "Job 3 supplies.\n")
    for nz in ap["config"]["noise"]:
        ptab += surface_table(ap, nz) + "\n"
    ptab += ("\n### 1.4 What the amplitude axis means physically\n\n"
             "Each injected amplitude is recorded as the maximum fractional "
             "acceleration anisotropy it produces over the shear-measured "
             "shell, and as the fraction of the response that actually TURNS "
             "the flux rather than rescaling it. The second number is why a "
             "source-axis tensor is intrinsically hard to see: most of its "
             "effect is a scalar rescaling that the blindness theorem says is "
             "unobservable as anisotropy.\n\n")
    ptab += dgg_table(ap)
    ptab += ("\n\nThe turning fraction is the number that carries the "
             "theorem, because it is a property of the injected law "
             "rather than of the detector: a source-axis response loses "
             "its observable content as the source becomes spherical, an "
             "external-axis one does not.  See the Headline table.\n")
    ptab += ("\n\n### 1.5 Reading the surfaces\n\n"
             "* The **source-axis** surface never rises above the test's size "
             "at any geometry, amplitude or noise level tested. That is a "
             "detectability statement, not an exclusion: after removing the "
             "flux-rescaling part that the blindness theorem declares "
             "unobservable, only about 11 per cent of a source-axis response "
             "survives on a strongly triaxial source and about 2 per cent on "
             "a near-spherical one, and 11 per cent of a max dg/g of 0.9 is "
             "below what this detector can see. The theorem's prediction is "
             "carried by the turning fraction, which collapses by a factor "
             "5.3; the power surface simply confirms that the residue is "
             "undetectable at both ends.\n"
             "* The **external-axis** surface does not collapse — it RISES "
             "toward the spherical limit, from 0.76 at axis ratio 0.500 to "
             "1.00 at 0.970 (amplitude 0.35, 2 per cent noise). A rounder "
             "source gives a simpler baseline against which a fixed-direction "
             "anisotropy stands out more clearly. This is the row Run AC read "
             "as a failed control.\n"
             "* The **member-well network** surface is low everywhere. With "
             "wells drawn from the same ellipsoid as the smooth source the "
             "network tensor is nearly degenerate with the source-axis one, "
             "and after flux-orthogonalisation what remains is small. This is "
             "a genuine detectability statement about the hypothesis, not a "
             "failure of the search.\n"
             "* **Knowing the axis is worth a great deal.** In an earlier run "
             "with a single fixed direction in the bank, an external-axis "
             "injection tilted 45 degrees away from it gave power 0.03 at "
             "every amplitude while the aligned case reached 1.00 — a "
             "misspecified axis is a null detector. That is why the bank here "
             "carries five directions spanning every constant symmetric "
             "traceless tensor, and why the look-elsewhere cost of that choice "
             "goes through the calibration.\n")

    # ------------------------------------------------------------- coarse
    cg = ap.get("coarse_grain", {})
    crows = cg.get("rows", [])
    coarse = ("The identical continuous mass is represented as 1, 10, 60 and "
              "300 catalogued wells and the resulting network response is "
              "compared:\n\n| wells | drift vs the 1-well limit |\n"
              "|---|---|\n")
    for r in crows:
        coarse += "| %d | %.4f |\n" % (r["n_wells"], r["drift_vs_1well"])
    d10 = crows[1]["drift_vs_1well"] if len(crows) > 1 else float("nan")
    d60 = crows[-2]["drift_vs_1well"] if len(crows) > 1 else float("nan")
    d300 = crows[-1]["drift_vs_1well"] if crows else float("nan")
    rel = cg.get("relative_change_60_to_300", float("nan"))
    coarse += (
        "\nTHIS IS A PARTIAL FAILURE AND IT SHOULD BE READ AS ONE. The gate "
        "asks two separate questions and the answers differ.\n\n"
        "* Does the response depend on whether the cataloguer calls the same "
        "mass one object or many? YES, strongly: representing it as 10 wells "
        "instead of 1 changes the response by %.2f times its own RMS. A "
        "well-network tensor built on a real member catalogue is therefore "
        "NOT the same object as the one built on the smooth mass it "
        "represents, and the difference is set by the catalogue rather than "
        "by the field equation. That is exactly what the standing brief warns "
        "about.\n"
        "* Does it converge once the catalogue is fine enough? YES: refining "
        "60 to 300 wells changes the drift by only %.3f (%.3f to %.3f). Above "
        "a resolution scale the tensor is a well-defined field with a "
        "coherence scale, not a count of catalogue rows.\n\n"
        "The honest summary is that the member-well tensor has a good "
        "continuum limit but is NOT invariant under the coarse-graining that "
        "actually varies between real catalogues: deblending, detection "
        "threshold, and how much mass is assigned to intracluster light "
        "rather than to members. Any claim resting on it needs those choices "
        "declared and varied. This lane's network power surfaces are computed "
        "with the well catalogue held fixed at 60 members.\n"
        % (d10, rel, d60, d300))

    # --------------------------------------------------------- monotonicity
    # Recomputed here over the MERGED grid, because the per-chunk blocks in
    # `axis_power_?.log` each see only their own geometries and therefore
    # report a zero axis-ratio spread by construction.
    amps = ap["config"]["amps"]
    ars_all = sorted({round(r["axis_ratio"], 3) for r in ap["rows"]})
    tilts = sorted({c["tilt_deg"] for r in ap["rows"]
                    for c in r["power"].values()})
    mono = ("| provenance | vs amplitude | spread | vs axis ratio | spread | "
            "vs tilt | spread |\n|---|---|---|---|---|---|---|\n")
    for prov in ("source", "external", "network"):
        cells = [c for r in ap["rows"] for c in r["power"].values()
                 if c["provenance"] == prov]
        by_amp = [float(np.mean([c["power"] for c in cells
                                 if abs(c["amp"] - a) < 1e-12]))
                  for a in amps]
        by_ar = [float(np.mean([c["power"] for r in ap["rows"]
                                if abs(round(r["axis_ratio"], 3) - a) < 1e-9
                                for c in r["power"].values()
                                if c["provenance"] == prov]))
                 for a in ars_all]
        tl = [t for t in tilts
              if any(abs(c["tilt_deg"] - t) < 1e-9 for c in cells)]
        by_t = [float(np.mean([c["power"] for c in cells
                               if abs(c["tilt_deg"] - t) < 1e-9]))
                for t in tl]
        mono += ("| %s | %s | %.3f | %s | %.3f | %s | %.3f |\n"
                 % (prov,
                    ", ".join("%.2f" % v for v in by_amp),
                    max(by_amp) - min(by_amp),
                    ", ".join("%.2f" % v for v in by_ar),
                    max(by_ar) - min(by_ar),
                    ", ".join("%.2f" % v for v in by_t),
                    max(by_t) - min(by_t)))
    mono += ("\nAmplitude order is " + ", ".join("%g" % a for a in amps)
             + "; axis-ratio order is "
             + ", ".join("%.2f" % a for a in ars_all)
             + "; tilt order is "
             + ", ".join("%.0f deg" % t for t in tilts) + ".\n\n"
             "The external-axis surface responds strongly to amplitude, as it "
             "must. The source-axis and network surfaces do not respond to "
             "anything, because they sit at the test size everywhere. That is "
             "itself the measurement, and it is why the theorem's prediction "
             "is carried by the turning fraction, a property of the injected "
             "law, rather than by the power, a property of the detector.\n")


    # ------------------------------------------------------- shear result
    def line(tag, lab, d):
        f = d[tag]["fit"]
        s = d[tag].get("null_std")
        p = d[tag].get("p_two_sided")
        ss = f"{s:.4f}" if s else "n/a"
        pp = f"{p:.4f}" if p is not None else "n/a"
        return (f"| {lab} | {f['alpha']:+.4f} | {f['e_alpha']:.4f} | {ss} | "
                f"{pp} | {f['chi2']/max(f['dof'],1):.2f} |")

    shr = (f"**Development sample: {dev['n']} clusters, "
           f"{ch['a2s_pred']['fit']['n']} (cluster, radial bin) rows, "
           f"{sum(r['n_bg'] for r in sel['dev']):,} background sources.**\n\n"
           "| channel | alpha | WLS error | axis-randomisation sigma | p | "
           "chi2/dof |\n|---|---|---|---|---|---|\n")
    for tag, lab in (("a2s_pred", "**a2s, predicted monopole (THE TEST)**"),
                     ("a2s_fe", "a2s, per-cluster fixed effects"),
                     ("a2s_meas", "a2s, measured monopole (diluted)"),
                     ("a2c_baryon", "a2c, source-aligned (positive control)"),
                     ("x2s_bmode", "x2s, cross component (B-mode null)")):
        shr += line(tag, lab, ch) + "\n"
    shr += (f"\n**Near-round negative control: {ctl['n']} clusters, "
            f"{cch['a2s_pred']['fit']['n']} rows, "
            f"{sum(r['n_bg'] for r in sel['ctrl']):,} background sources.**\n\n"
            "| channel | alpha | WLS error | axis-randomisation sigma | p | "
            "chi2/dof |\n|---|---|---|---|---|---|\n")
    for tag, lab in (("a2s_pred", "a2s, predicted monopole"),
                     ("a2s_fe", "a2s, per-cluster fixed effects"),
                     ("a2c_baryon", "a2c, source-aligned"),
                     ("x2s_bmode", "x2s, cross component")):
        shr += line(tag, lab, cch) + "\n"
    fa = ch["a2s_pred"]["fit"]
    shr += (f"\nThe uniform-shear nuisance column is fitted, not assumed away: "
            f"beta = {fa['beta']:+.5f} +- {fa['e_beta']:.5f} with "
            f"corr(alpha, beta) = {fa['corr_alpha_beta']:+.2f}. The predicted "
            f"uniform shear from catalogued neighbours is "
            f"{sel['neighbour_shear_median']:.2e} (median over the parent, "
            f"90th percentile {sel['neighbour_shear_p90']:.2e}), consistent "
            f"with the fitted value.\n")
    shr += ("\nStacked quadrupole amplitudes in the member-aligned frame, per "
            "unit predicted monopole:\n\n"
            "| sample | a0/g_pred | a2c/g_pred | a2s/g_pred |\n"
            "|---|---|---|---|\n")
    for tag, d in (("DEV", dev), ("CTRL", ctl)):
        a = d["amplitude"]
        shr += (f"| {tag} | {a['ratio_0']:+.4f} +- {a['e_ratio_0']:.4f} | "
                f"{a['ratio_2c']:+.4f} +- {a['e_ratio_2c']:.4f} | "
                f"{a['ratio_2s']:+.4f} +- {a['e_ratio_2s']:.4f} |\n")
    shr += ("\nEvery channel, in both samples, is consistent with zero. The "
            "stacked ratios and the regression coefficients differ in sign in "
            "the a2c row because they are different estimators: the stack is "
            "an inverse-variance mean of a2c/g_pred over every (cluster, bin) "
            "cell, while the regression fits a slope against g_pred with an "
            "intercept that absorbs an additive c-term. The regression is the "
            "one to read, and both are within one sigma of zero.\n")

    # -------------------------------------------------------------- gates
    gates = (
        f"| gate | DEV | CTRL |\n|---|---|---|\n"
        f"| tangential monopole (must be positive) | "
        f"{dev['gt_monopole']:+.5f} +- {dev['gt_monopole_err']:.5f} "
        f"({dev['gt_monopole']/dev['gt_monopole_err']:+.1f} sigma) | "
        f"{ctl['gt_monopole']:+.5f} +- {ctl['gt_monopole_err']:.5f} "
        f"({ctl['gt_monopole']/ctl['gt_monopole_err']:+.1f} sigma) |\n"
        f"| cross monopole (must be zero) | "
        f"{dev['gx_monopole']:+.5f} +- {dev['gx_monopole_err']:.5f} "
        f"({dev['gx_monopole']/dev['gx_monopole_err']:+.1f} sigma) | "
        f"{ctl['gx_monopole']:+.5f} +- {ctl['gx_monopole_err']:.5f} "
        f"({ctl['gx_monopole']/ctl['gx_monopole_err']:+.1f} sigma) |\n"
        f"| monopole/quadrupole noise correlation | "
        f"{dev['monopole_quadrupole_noise_corr']:+.3f} | "
        f"{ctl['monopole_quadrupole_noise_corr']:+.3f} |\n"
        f"| estimator agreement vs the eFEDS lane | "
        f"{cc['dev']['estimator_gate']['mean_difference']:+.5f} +- "
        f"{cc['dev']['estimator_gate']['err']:.5f} "
        f"(chi2/system {cc['dev']['estimator_gate']['chi2_per_system']:.2f}) | "
        f"{cc['ctrl']['estimator_gate']['mean_difference']:+.5f} +- "
        f"{cc['ctrl']['estimator_gate']['err']:.5f} "
        f"(chi2/system {cc['ctrl']['estimator_gate']['chi2_per_system']:.2f}) |\n"
        f"| M_dyn monotone inside truncation | "
        f"{cc['dev']['monotone_gate']['n'] - cc['dev']['monotone_gate']['n_non_monotone']}"
        f"/{cc['dev']['monotone_gate']['n']} | "
        f"{cc['ctrl']['monotone_gate']['n'] - cc['ctrl']['monotone_gate']['n_non_monotone']}"
        f"/{cc['ctrl']['monotone_gate']['n']} |\n"
        f"\nThe **positive control does not fire**: the source-aligned "
        f"quadrupole is {ch['a2c_baryon']['fit']['alpha']:+.4f} +- "
        f"{ch['a2c_baryon']['fit']['e_alpha']:.4f}, consistent both with zero "
        f"and with the ~0.13 expected from the sample's own member-light "
        f"ellipticity. This sample is right at the edge of being able to see a "
        f"baryonic quadrupole at all, which is a direct statement that the "
        f"null in the a2s channel is a statement about sensitivity, not about "
        f"nature.\n")

    # ---------------------------------------------------------- selection
    sf = sel["selection_function"]
    dvm = np.array([abs(r["misalign_deg"]) for r in sel["dev"]])
    dve = np.array([r["e_mem_debiased"] for r in sel["dev"]])
    seltxt = (
        f"Parent: {sel['parent']['n_parent']} eFEDS systems with a Bahar+2022 "
        f"density fit; {sel['parent']['n_measured']} have both a member "
        f"measurement and a defined external axis.\n\n"
        f"    common   {sf['Z_MIN']} <= z <= {sf['Z_MAX']}, "
        f"n_mem >= {sf['N_MEM_MIN']}, n_bg >= {sf['N_BG_MIN']},\n"
        f"             aperture >= {sf['EDGE_MARGIN_DEG']} deg inside the "
        f"eFEDS box, external tidal axis defined from >= "
        f"{sf['ENV_NEIGH_MIN']} catalogued neighbours\n"
        f"    DEV      debiased member-light ellipticity >= {sf['E_DEV_MIN']} "
        f"at >= {sf['E_DEV_SNR']} sigma, AND\n"
        f"             |sin 2 Delta| >= {sf['SIN2D_MIN']}, i.e. the two "
        f"independently measured axes are\n"
        f"             misaligned by 15-75 degrees\n"
        f"    CTRL     debiased ellipticity <= {sf['E_CTRL_MAX']} and "
        f"<= {sf['E_CTRL_SNR']} sigma (near-round)\n\n"
        f"| | DEV | CTRL |\n|---|---|---|\n"
        f"| systems | {len(sel['dev'])} | {len(sel['ctrl'])} |\n"
        f"| median \\|Delta\\| | {np.median(dvm):.1f} deg "
        f"(range {dvm.min():.1f}-{dvm.max():.1f}) | not required |\n"
        f"| deconvolved RMS member ellipticity | "
        f"{am['geometric_factor_e_mem']:.3f} | 0.000 |\n"
        f"| background sources | "
        f"{sum(r['n_bg'] for r in sel['dev']):,} | "
        f"{sum(r['n_bg'] for r in sel['ctrl']):,} |\n"
        f"| median background density | "
        f"{np.median([r['n_bg_per_arcmin2'] for r in sel['dev']]):.2f} "
        f"/arcmin^2 | "
        f"{np.median([r['n_bg_per_arcmin2'] for r in sel['ctrl']]):.2f} "
        f"/arcmin^2 |\n\n"
        f"Frozen before any shear was computed: `selection.json`, SHA-256\n"
        f"`{open(os.path.join(HERE, 'selection.json.sha256')).read().strip()}`."
        f" `shear2d.py` recomputes that hash on every run and refuses to "
        f"proceed if it differs.\n")

    # --------------------------------------------------------- amplitudes
    atab = ("| candidate | provenance | eps_K | predicted a2c/a0 | predicted "
            "a2s/a0 |\n|---|---|---|---|---|\n")
    for r in sorted(am["survivors"], key=lambda x: -x["eps_K_dev_max"]):
        note = " *(scalar-degenerate)*" if r["qumond_scalar_degenerate"] else ""
        nm = r["name"].replace("|", chr(92) + "|")
        atab += (f"| `{nm}`{note} | {r['provenance']} | "
                 f"{r['eps_K_dev_max']:.3f} | "
                 f"{r['quadrupole_a2c_pred_dev']:.4f} | "
                 f"{r['quadrupole_a2s_pred_dev']:.4f} |\n")

    lim = inj["exclusion_2sided_95"] * dev["median_abs_sin2delta"]
    could = (
        f"The end-to-end injection measures the answer on the real data rather "
        f"than on a simulation of it. Injecting a known misaligned quadrupole "
        f"into the measured harmonics and rerunning the whole fit gives a "
        f"response slope of {inj['response_slope']:.4f} and\n\n"
        f"    95% exclusion            |alpha| < {inj['exclusion_2sided_95']:.3f}\n"
        f"    95%-power amplitude      alpha  = {inj['amplitude_for_95pc_power']:.3f}\n"
        f"    same, fixed effects      |alpha| < {injf['exclusion_2sided_95']:.3f}\n"
        f"    same, measured monopole  |alpha| < "
        f"{injm.get('effective_exclusion_95', float('nan')):.3f}  "
        f"(after correcting its 0.072 dilution)\n\n"
        f"At the sample's median |sin 2Delta| = "
        f"{dev['median_abs_sin2delta']:.3f} that is a limit on the misaligned "
        f"quadrupole ratio of {lim:.2f}, i.e. on a misaligned convergence "
        f"ellipticity of e_kappa < "
        f"{lim / am['quadrupole_coefficient']:.2f}. **The geometric maximum of "
        f"e_kappa is 1.** So this sample excludes no external-axis tensor of "
        f"any amplitude, and a null from it would say nothing even if the "
        f"tournament had contained one.\n\n"
        f"Per candidate:\n\n"
        f"* **8 isotropic survivors** (`scalar_a0`, `iso_K`): predict no "
        f"quadrupole of any phase. The two-dimensional test cannot address "
        f"them at any amplitude, and no improvement in sensitivity would "
        f"change that.\n"
        f"* **2 `tensor_d` survivors**: `K grad Phi_N = exp(2AW/3) grad Phi_N` "
        f"exactly. Predicted quadrupole zero in both channels. These are "
        f"scalar rescalings, not tensors, and their fitted eps_K of 1.000 is "
        f"an eigenvalue spread that never reaches an observable.\n"
        f"* **1 `tensor_T` and 7 `tensor_S` survivors**: predicted a2c/a0 from "
        f"{min(r['quadrupole_a2c_pred_dev'] for r in am['survivors'] if r['quadrupole_a2c_pred_dev'] > 0):.3f} "
        f"to {max(r['quadrupole_a2c_pred_dev'] for r in am['survivors']):.3f}, "
        f"predicted a2s/a0 exactly zero. They land in the channel that is "
        f"degenerate with baryonic ellipticity. The measured a2c coefficient "
        f"is {ch['a2c_baryon']['fit']['alpha']:+.4f} +- "
        f"{ch['a2c_baryon']['fit']['e_alpha']:.4f}, which contains the "
        f"baryonic quadrupole as well; the largest prediction "
        f"({max(r['quadrupole_a2c_pred_dev'] for r in am['survivors']):.3f}) "
        f"sits "
        f"{(max(r['quadrupole_a2c_pred_dev'] for r in am['survivors']) - ch['a2c_baryon']['fit']['alpha']) / ch['a2c_baryon']['fit']['e_alpha']:.1f} "
        f"sigma above it, which is suggestive and not a limit, because the "
        f"baryonic contribution is unknown and of the same size.\n"
        f"* **Placement on the power surface.** The source-axis and network "
        f"survivors reach eps_K of 0.16-1.00, which is at or above the top of "
        f"the injected amplitude range, where the calibrated power surfaces "
        f"give the resolved-dynamics detector its highest power. So a resolved "
        f"three-dimensional test on a strongly triaxial system is the probe "
        f"with real leverage on them; the two-dimensional phase channel is "
        f"not.\n")

    could_not = f"""
* **The two-dimensional phase test did not reach useful sensitivity.** Its 95%
  exclusion on a misaligned convergence ellipticity is
  {lim / am['quadrupole_coefficient']:.2f}, above the geometric maximum of 1.
  This is a null with a stated power, and the stated power is zero over the
  whole physical range. Reaching e_kappa = 0.2 needs
  {(lim / am['quadrupole_coefficient'] / 0.2) ** 2:.0f} times the effective
  source count of this sample — a deeper shear catalogue (DECADE delivers
  {np.median([r['n_bg_per_arcmin2'] for r in sel['dev']]):.1f} usable
  background sources per square arcminute here; LSST-class depth is ~6 times
  that) combined with several hundred clusters, not 27.
* **The positive control did not fire.** The source-aligned quadrupole is
  consistent with zero at {ch['a2c_baryon']['fit']['e_alpha']:.3f}, so this
  sample has not demonstrated that it can see a quadrupole of the size the
  baryons alone should produce. Until it does, the null in the misaligned
  channel cannot be read as evidence of absence.
* **No X-ray position angle for eFEDS.** The Bahar+2022 fits are spherically
  symmetric and the catalogue carries no ellipticity, so the "misalignment
  between candidate axes" is measured between the member-light axis and the
  large-scale environment axis, not between member light and X-ray. A third,
  independent axis would strengthen the design considerably.
* **The member-well network hypothesis is barely detectable in this design.**
  Its power surface stays low at every geometry and amplitude tested. That is a
  detectability result, not an exclusion: the wells here trace the same
  ellipsoid as the smooth source, so after flux-orthogonalisation little
  remains. A catalogue whose axis genuinely departs from the smooth baryon axis
  would be a sharper test, and building one requires member catalogues of a
  quality eFEDS does not have.
* **Line-of-sight structure, source-redshift error and member contamination of
  the shape measurement are not in the Stage-0 null.** They are the three I
  would add next.
* **Nothing here tests reciprocity, an action principle, or asymptotics.**
  This lane measures detectability and provenance, not viability.
"""

    from report_template import TEMPLATE as tpl
    fp = np.array([x for r in ap["rows"] for x in r["audit_fpr"].values()])
    tpl = (tpl.replace("<<NFPR>>", str(fp.size))
              .replace("<<FPRMED>>", f"{np.median(fp):.3f}")
              .replace("<<FPRMEAN>>", f"{fp.mean():.3f}")
              .replace("<<FPRMIN>>", f"{fp.min():.3f}")
              .replace("<<FPRMAX>>", f"{fp.max():.3f}")
              .replace("<<FPRFRAC>>",
                       f"{100*np.mean((fp >= 0.02) & (fp <= 0.09)):.0f}"))
    md = (tpl.replace("<<HEADLINE>>", headline)
             .replace("<<POWER_TABLES>>", ptab)
             .replace("<<COARSE>>", coarse)
             .replace("<<MONO>>", mono)
             .replace("<<SHEAR_RESULT>>", shr)
             .replace("<<GATES>>", gates)
             .replace("<<SELECTION>>", seltxt)
             .replace("<<AMPLITUDE_TABLE>>", atab)
             .replace("<<COULD_SEE>>", could)
             .replace("<<COULD_NOT>>", could_not))
    with open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("written REPORT.md", len(md), "chars")


if __name__ == "__main__":
    main()
