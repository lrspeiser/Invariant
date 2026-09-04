"""Prospective validation of the Stage 4 certificate.

Refusing five KNOWN historical failures is a regression test, not evidence the
gate catches anything new.  This module runs it against mechanisms it was NOT
designed against, and reports honestly where it does not catch them -- a gap
found here is the point of the exercise, not a failure of it.

Every case carries a STABLE TYPED IDENTIFIER.  No logic anywhere may depend on a
human-readable name: the development harness selected its expected-failure set by
substring-matching "control" in a case title and silently excused a case whose
title contained that word.

    python prospective.py
"""
import io
import json
import os

import numpy as np

import certificate as C

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(90402026)


def c2_typed(target_resp, control_resp):
    """C2 without the uninterpretable inf ratio.

    The development version printed 'inf x the effect range' when the target
    responsiveness was zero.  A ratio against zero is not a number; the verdict
    is that the statistic is insensitive to the claimed effect.
    """
    if abs(target_resp) <= 0.0:
        return dict(passed=False, target_responsiveness=float(target_resp),
                    control_responsiveness=float(control_resp), ratio=None,
                    detail=(f"target responsiveness = {target_resp:.3g}; control "
                            f"responsiveness = {control_resp:.3g}; verdict: "
                            f"statistic INSENSITIVE to the claimed effect"))
    ratio = abs(control_resp) / abs(target_resp)
    return dict(passed=ratio < 1.0, target_responsiveness=float(target_resp),
                control_responsiveness=float(control_resp), ratio=float(ratio),
                detail=(f"a control lever reproduces {ratio:.2f}x the target's "
                        f"responsiveness"))


# --------------------------------------------------------- the prospective suite
# Mechanisms NOT present in the development set.
def suite():
    t = np.linspace(0.05, 1.0, 60)
    cases = {}

    # New mechanism 1: the statistic moves with the effect ONLY because effect
    # and statistic share a denominator.  Nothing in the dev set had this.
    denom = 1.0 + 0.4 * t
    cases["CERT.SHARED_DENOMINATOR.001"] = dict(
        _must_fail=True, _mech="statistic and effect share a denominator",
        C2_not_a_restatement=c2_typed(target_resp=1.0, control_resp=0.97),
        C7_nuisance_distinct=C.c7_nuisance_distinct(
            signal_signature=1.0 / denom,
            nuisance_signatures={"the shared denominator": 1.0 / denom,
                                 "unrelated": np.sin(5 * t)}))

    # New mechanism 2: a SELECTION FUNCTION reproduces the signal shape.
    sel = np.exp(-((t - 0.5) ** 2) / 0.05)
    cases["CERT.SELECTION_MIMIC.001"] = dict(
        _must_fail=True, _mech="the selection function has the signal's shape",
        C7_nuisance_distinct=C.c7_nuisance_distinct(
            signal_signature=sel * (1 + 0.02 * RNG.normal(size=t.size)),
            nuisance_signatures={"selection function": sel,
                                 "seeing": t, "extinction": t ** 0.5}))

    # New mechanism 3: NON-MONOTONE response -- the statistic is sensitive at
    # the amplitude that happened to be tested and flat at the PREDICTED one.
    def nonmono(a):
        return float(np.sin(3.0 * a))          # flat near a = pi/6*... turning pts
    tested = np.linspace(0.0, 0.5, 9)          # looks responsive here
    cases["CERT.NONMONOTONE_RESPONSE.001"] = dict(
        _must_fail=True, _mech="responsive where tested, flat where predicted",
        C1_responsive=C.c1_responsive(nonmono, tested),
        # at the PREDICTED amplitude the local slope is ~0
        C4_powered=C.c4_powered(
            responsiveness=abs(3.0 * np.cos(3.0 * (np.pi / 6))),
            predicted_effect=0.30, noise_sd=0.05))

    # New mechanism 4: partly sensitive but badly underpowered at the PREDICTED
    # effect, while well powered at a convenient larger one.
    cases["CERT.UNDERPOWERED_AT_PREDICTION.001"] = dict(
        _must_fail=True, _mech="powered at a convenient amplitude, not the predicted",
        C1_responsive=C.c1_responsive(lambda a: 0.6 * a, np.linspace(0, 1, 9)),
        C4_powered=C.c4_powered(responsiveness=0.6, predicted_effect=0.04,
                                noise_sd=0.05))

    # New mechanism 5: a permutation arm that is subtly NON-exchangeable --
    # the null is built by shuffling a label that is itself a function of the
    # measured quantity, so the null mean is displaced.
    cases["CERT.NULL.NONEXCHANGEABLE.001"] = dict(
        _must_fail=True, _mech="permuted label is a function of the measurement",
        C3_exchangeable=C.c3_exchangeable(
            observed=-0.31, null_draws=RNG.normal(-0.28, 0.05, 4000)))

    # Positive controls that MUST pass, so the suite is two-sided.
    cases["CERT.VALID.CLEAN_EFFECT.001"] = dict(
        _must_fail=False, _mech="a clean, well-supported, well-powered effect",
        C1_responsive=C.c1_responsive(lambda a: 1.1 * a, np.linspace(0, 1, 9)),
        C2_not_a_restatement=c2_typed(target_resp=1.1, control_resp=0.08),
        C3_exchangeable=C.c3_exchangeable(0.55, RNG.normal(0.0, 0.09, 4000)),
        C4_powered=C.c4_powered(1.1, 0.45, 0.09),
        C5_support=C.c5_support((0.3, 0.8), (0.2, 0.95)),
        C6_out_of_grammar=C.c6_out_of_grammar(0.82),
        C7_nuisance_distinct=C.c7_nuisance_distinct(
            np.sin(4 * t), {"seeing": t, "extinction": t ** 0.5,
                            "miscentring": np.log(t + 1)}))

    cases["CERT.VALID.MODEST_BUT_HONEST.001"] = dict(
        _must_fail=False, _mech="a smaller effect, honestly powered",
        C1_responsive=C.c1_responsive(lambda a: 0.4 * a, np.linspace(0, 1, 9)),
        C2_not_a_restatement=c2_typed(target_resp=0.4, control_resp=0.05),
        C3_exchangeable=C.c3_exchangeable(0.2, RNG.normal(0.0, 0.03, 4000)),
        C4_powered=C.c4_powered(0.4, 0.30, 0.03),
        C5_support=C.c5_support((0.4, 0.7), (0.3, 0.9)),
        C6_out_of_grammar=C.c6_out_of_grammar(0.61),
        C7_nuisance_distinct=C.c7_nuisance_distinct(
            np.cos(6 * t), {"seeing": t, "PSF size": t ** 2}))
    return cases


def main():
    cases = suite()
    caught = missed = false_alarm = 0
    rows = []
    for cid, spec in cases.items():
        must_fail = spec["_must_fail"]
        mech = spec["_mech"]
        checks = {k: v for k, v in spec.items() if not k.startswith("_")}
        issued = C.certify(f"{cid}   [{mech}]", checks)
        if must_fail and not issued:
            caught += 1
            verdict = "CAUGHT"
        elif must_fail and issued:
            missed += 1
            verdict = "*** MISSED -- a coverage gap ***"
        elif (not must_fail) and issued:
            verdict = "correctly passed"
        else:
            false_alarm += 1
            verdict = "*** FALSE ALARM ***"
        rows.append(dict(id=cid, mechanism=mech, must_fail=must_fail,
                         issued=issued, verdict=verdict, checks=checks))

    n_fail = sum(1 for s in cases.values() if s["_must_fail"])
    n_pass = len(cases) - n_fail
    print("\n" + "=" * 74)
    print("PROSPECTIVE VALIDATION -- mechanisms the gate was NOT designed against")
    print("=" * 74)
    for r in rows:
        print(f"  {r['verdict']:<32} {r['id']}")
    print()
    print(f"  new failure mechanisms caught : {caught}/{n_fail}")
    print(f"  coverage gaps (missed)        : {missed}/{n_fail}")
    print(f"  false alarms on valid effects : {false_alarm}/{n_pass}")
    if missed:
        print()
        print("  A MISS IS THE POINT OF THIS EXERCISE, not a failure of it:")
        print("  it names a failure class the seven checks cannot see.")

    doc = dict(caught=caught, missed=missed, false_alarms=false_alarm,
               n_must_fail=n_fail, n_must_pass=n_pass, cases=rows)
    p = os.path.join(HERE, "prospective.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(doc, indent=1, default=float))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
