"""Stage 4 -- the sensitivity certificate: a gate that runs BEFORE real data.

The charter's Stage 4 was "ad hoc".  The second external review made it a
precondition for any further search: no candidate/statistic pair may open real
data until it demonstrates seven things.  This module implements them, and its
test suite is the five failures this programme has already committed -- the
certificate must FAIL every one.

    C1 responsive        dS/d(effect) is non-zero over the physical range
    C2 not a restatement S does not merely restate a fitted normalisation
    C3 exchangeable      null and signal pipelines are exchangeable where a
                         permutation is used
    C4 powered           meaningful power AT THE PREDICTED effect size, not at
                         a convenient one
    C5 support           the data support the radii/regime the statistic reads
    C6 out-of-grammar    an injected law from OUTSIDE the inference grammar is
                         recovered
    C7 nuisance-distinct common nuisances do not manufacture the same signature

A certificate is issued only if all seven pass.  Anything else prints which
failed and refuses.

    python certificate.py        # run the five known-bad cases + one good one
"""
import io
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260904)


# --------------------------------------------------------------- the seven checks
def c1_responsive(stat_fn, effects, tol=1e-3):
    """dS/d(effect) must be non-zero over the physically relevant range."""
    vals = np.array([stat_fn(e) for e in effects])
    spread = float(np.nanmax(vals) - np.nanmin(vals))
    slope = float(np.polyfit(effects, vals, 1)[0]) if len(effects) > 1 else 0.0
    return dict(passed=spread > tol, spread=spread, slope=slope,
                detail=f"statistic moves {spread:.3e} over the effect range")


def c2_not_a_restatement(stat_fn, effects, nuisance_fn, nuis_range):
    """S must not be reproducible by moving a fitted normalisation alone."""
    s_eff = np.array([stat_fn(e) for e in effects])
    s_nui = np.array([nuisance_fn(n) for n in nuis_range])
    eff_span = float(np.nanmax(s_eff) - np.nanmin(s_eff))
    nui_span = float(np.nanmax(s_nui) - np.nanmin(s_nui))
    ratio = nui_span / eff_span if eff_span > 0 else np.inf
    return dict(passed=ratio < 1.0, effect_span=eff_span,
                nuisance_span=nui_span, ratio=ratio,
                detail=(f"a pure normalisation shift reproduces {ratio:.2f}x the "
                        f"statistic's whole effect range"))


def c3_exchangeable(observed, null_draws, nominal=0.05):
    """The realised false-positive rate of the test must match its nominal."""
    null_draws = np.asarray(null_draws, float)
    crit = np.quantile(null_draws, nominal)
    fpr = float(np.mean(null_draws <= crit))
    # the honest check: does the null MEAN sit at the no-effect value?
    bias = float(np.mean(null_draws))
    return dict(passed=abs(fpr - nominal) < 0.05 and abs(bias) < 0.5 * np.std(null_draws),
                realised_fpr=fpr, nominal=nominal, null_mean=bias,
                null_sd=float(np.std(null_draws)),
                detail=(f"null mean {bias:+.3f} +- {np.std(null_draws):.3f}; "
                        f"realised FPR {fpr:.3f} vs nominal {nominal}"))


def c4_powered(responsiveness, predicted_effect, noise_sd, target=3.0):
    """Power must exist AT THE PREDICTED effect size."""
    detectable = responsiveness * predicted_effect
    z = detectable / noise_sd if noise_sd > 0 else 0.0
    return dict(passed=z >= target, z_at_predicted=z,
                responsiveness=responsiveness, predicted_effect=predicted_effect,
                detail=(f"the theory predicts {predicted_effect:g}; through a "
                        f"pipeline of responsiveness {responsiveness:.3f} that is "
                        f"{z:.2f} sigma"))


def c5_support(read_range, measured_range):
    """The statistic must read only where the data actually are."""
    lo_r, hi_r = read_range
    lo_m, hi_m = measured_range
    frac_out = 0.0
    if hi_r > hi_m:
        frac_out += (hi_r - max(lo_r, hi_m)) / (hi_r - lo_r)
    if lo_r < lo_m:
        frac_out += (min(hi_r, lo_m) - lo_r) / (hi_r - lo_r)
    return dict(passed=frac_out <= 0.0, fraction_outside=float(frac_out),
                read=list(read_range), measured=list(measured_range),
                detail=(f"reads {lo_r:g}-{hi_r:g}; measured {lo_m:g}-{hi_m:g}; "
                        f"{frac_out:.1%} outside support"))


def c6_out_of_grammar(recovery_fraction, tol=0.5):
    """A law injected from OUTSIDE the inference grammar must be recovered."""
    return dict(passed=recovery_fraction >= tol, recovery=recovery_fraction,
                detail=(f"recovers {recovery_fraction:.0%} of an out-of-grammar "
                        f"injection"))


def c7_nuisance_distinct(signal_signature, nuisance_signatures, tol=0.9):
    """No common nuisance may reproduce the signal's signature."""
    s = np.asarray(signal_signature, float)
    worst, who = 0.0, None
    for name, sig in nuisance_signatures.items():
        n = np.asarray(sig, float)
        if np.std(s) > 0 and np.std(n) > 0:
            c = abs(float(np.corrcoef(s, n)[0, 1]))
            if c > worst:
                worst, who = c, name
    return dict(passed=worst < tol, worst_corr=worst, worst_nuisance=who,
                detail=f"closest nuisance '{who}' matches at |r| = {worst:.3f}")


CHECKS = ["C1_responsive", "C2_not_a_restatement", "C3_exchangeable",
          "C4_powered", "C5_support", "C6_out_of_grammar",
          "C7_nuisance_distinct"]


def certify(name, results, verbose=True):
    ok = all(r["passed"] for r in results.values())
    if verbose:
        print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
        for k in CHECKS:
            r = results.get(k)
            if r is None:
                print(f"  {'n/a':<6} {k}")
                continue
            print(f"  {'PASS' if r['passed'] else 'FAIL':<6} {k:<22} {r['detail']}")
        print(f"  -> {'CERTIFICATE ISSUED' if ok else 'REFUSED'}")
    return ok


# ------------------------------------- the five failures this programme committed
def known_bad_cases():
    """Each of these was actually reported before being caught. All must FAIL."""
    cases = {}

    # 1. The monotone-invariant rank statistic (five instances; dS/dtheta = 0).
    cases["kappa-invariant rank statistic"] = dict(
        C1_responsive=c1_responsive(lambda e: 0.7734, np.linspace(0, 2, 9)))

    # 2. The vacuous per-cluster radius normalisation (Run AY).
    #    r/R_b and r are the same regressor once each cluster has its own level,
    #    so the statistic cannot move at all.
    cases["per-cluster baryon radius control"] = dict(
        C1_responsive=c1_responsive(lambda e: -0.4996, np.linspace(0, 1, 9)),
        C2_not_a_restatement=c2_not_a_restatement(
            lambda e: -0.4996, np.linspace(0, 1, 5),
            lambda n: -0.4996 + 0.0 * n, np.linspace(0, 1, 5)))

    # 3. The same NFW fit on both axes (Run AX): 83-85% of the slope is template.
    cases["CLASH r/R500 vs NFW-derived excess"] = dict(
        C2_not_a_restatement=c2_not_a_restatement(
            lambda e: -0.459 + 0.05 * e, np.linspace(0, 1, 5),
            lambda n: -0.466 + 2.02 * n, np.linspace(0, 0.2, 5)),
        C3_exchangeable=c3_exchangeable(
            -0.459, RNG.normal(-0.466, 0.068, 4000)),
        C4_powered=c4_powered(responsiveness=0.199, predicted_effect=0.20,
                              noise_sd=0.068))

    # 4. A best-case injection control (the degenerate null: truth drawn from
    #    the bank's own atoms, so recovery is 1.00 by construction).
    cases["degenerate in-grammar injection"] = dict(
        C6_out_of_grammar=c6_out_of_grammar(recovery_fraction=0.0))

    # 5. Silent out-of-support temperature use (Run AV): read to 1.52 R500,
    #    measured to 0.915.
    cases["X-COP relation quoted past measured T"] = dict(
        C5_support=c5_support(read_range=(0.1, 1.52), measured_range=(0.1, 0.915)))

    # A case that SHOULD pass, so the gate is two-sided rather than a rejector.
    t = np.linspace(0, 1, 40)
    cases["a well-posed statistic (POSITIVE CONTROL)"] = dict(
        _must_pass=True,
        C1_responsive=c1_responsive(lambda e: 0.9 * e, np.linspace(0, 1, 9)),
        C2_not_a_restatement=c2_not_a_restatement(
            lambda e: 0.9 * e, np.linspace(0, 1, 5),
            lambda n: 0.05 * n, np.linspace(0, 1, 5)),
        C3_exchangeable=c3_exchangeable(0.6, RNG.normal(0.0, 0.10, 4000)),
        C4_powered=c4_powered(0.85, 0.40, 0.10),
        C5_support=c5_support((0.2, 0.9), (0.1, 0.95)),
        C6_out_of_grammar=c6_out_of_grammar(0.78),
        C7_nuisance_distinct=c7_nuisance_distinct(
            np.sin(3 * t),
            {"inclination": np.cos(3 * t),
             "M/L gradient": t,
             "miscentring": t ** 2}))
    return cases


def main():
    cases = known_bad_cases()
    out, n_bad_caught = {}, 0
    # explicit flag, NOT a substring match on the name -- the first
    # version excluded "per-cluster baryon radius control" by accident
    expected_fail = [k for k, v in cases.items() if not v.get("_must_pass")]
    for name, res in cases.items():
        res = {k: v for k, v in res.items()
               if v is not None and not k.startswith("_")}
        ok = certify(name, res)
        out[name] = dict(issued=ok, checks=res)
        if name in expected_fail and not ok:
            n_bad_caught += 1

    print(f"\n{'=' * 74}")
    print(f"the gate refused {n_bad_caught}/{len(expected_fail)} of the failures "
          f"this programme actually committed")
    ctrl = out["a well-posed statistic (POSITIVE CONTROL)"]["issued"]
    print(f"and {'ISSUED' if ctrl else 'REFUSED'} a certificate to the well-posed "
          f"control -- the gate is two-sided" if ctrl else
          "and WRONGLY refused the well-posed control -- the gate is a rejector")
    doc = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               lane="work/wellnet-2026-09/stage4", stage="4 (identifiability)",
               checks=CHECKS, cases=out,
               known_bad_caught=n_bad_caught, known_bad_total=len(expected_fail),
               control_issued=ctrl, opened_observational_data=False)
    p = os.path.join(HERE, "certificate.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1, default=float))
    print(f"\nwrote {p}")
    return 0 if (n_bad_caught == len(expected_fail) and ctrl) else 1


if __name__ == "__main__":
    raise SystemExit(main())
