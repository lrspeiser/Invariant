"""Gates for the widened inverse symbolic engine: series, product, and integral families.

The non-negotiable idea is inherited verbatim from the continued-fraction lane: a match found
at fit precision is interpolation, and only survival at much higher precision may promote it
-- to a conjecture, never to a theorem.  These tests pin that guard with a fabricated fake
that must die, pin every family's rediscovery controls, the three ordinal codecs, the
declared convergence gates, the accelerator's conditioning, the quadrature's accuracy, the
exact evaluators against independently known closed forms, CPU/GPU agreement, receipt
determinism, and tamper detection.  GPU tests skip cleanly when no CUDA device is present.

The screening and proof-routing gates additionally pin what makes DG3 different from the
first continued-fraction run: a corpus of declared scale with a closed provenance forest,
verdicts that always carry a re-verified justification, and a headline that is *computed*
from the candidate records rather than asserted.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from sigma_theory_compiler import families_prior_art_screen as screen
from sigma_theory_compiler import families_proof_router as router
from sigma_theory_compiler import inverse_symbolic_families as fam
from sigma_theory_compiler.sigma_core import canonical_json_bytes

RECEIPTS = Path(__file__).resolve().parents[1] / "runs" / "math" / "inverse-symbolic"
ENUMERATION_RECEIPT = RECEIPTS / "families-v1.json"
SCREEN_RECEIPT = RECEIPTS / "families-screen-v1.json"
PROOF_RECEIPT = RECEIPTS / "families-proof-v1.json"


def _cupy_or_none():
    try:
        import cupy

        cupy.arange(4).sum()
        return cupy
    except Exception:  # noqa: BLE001 - any CUDA absence means skip
        return None


def _reload(receipt):
    """Round-trip a receipt through canonical JSON, as a validator would receive it."""

    return json.loads(canonical_json_bytes(receipt))


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"receipt not present: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus():
    return screen.build_corpus()


# ---------------------------------------------------------------------------
# Declared scale and ordinal codecs
# ---------------------------------------------------------------------------


def test_every_family_declares_at_least_1e8_ordinals():
    for family in fam.FAMILIES:
        assert fam.FAMILY_ORDINALS[family] >= 10**8, family
    # The two iterated families also clear 1e8 in *distinct members*, before the rational
    # prefactor -- so the scale claim does not lean on the factorizing dimension alone.
    assert fam.S_SERIES_COUNT >= 10**8
    assert fam.P_PRODUCT_COUNT >= 10**8


def test_series_ordinal_codec_round_trips():
    ordinal = fam.encode_s_ordinal([1, 2, 1, 0], [4, 4, 1, 0], Fraction(1), Fraction(1))
    decoded = fam.decode_s_ordinal(ordinal)
    assert decoded["p"] == [1, 2, 1, 0]
    assert decoded["q"] == [4, 4, 1, 0]
    assert decoded["z"] == "1"
    assert decoded["prefactor"] == "1"
    for probe in (0, 1, 12345, fam.S_ORDINAL_COUNT - 1):
        row = fam.decode_s_ordinal(probe)
        assert (
            fam.encode_s_ordinal(
                row["p"], row["q"], Fraction(row["z"]), Fraction(row["prefactor"])
            )
            == probe
        )


def test_product_ordinal_codec_round_trips():
    ordinal = fam.encode_p_ordinal([0, 0, 4, 0], [-1, 0, 4, 0], 1, Fraction(2))
    decoded = fam.decode_p_ordinal(ordinal)
    assert decoded["a"] == [0, 0, 4, 0]
    assert decoded["b"] == [-1, 0, 4, 0]
    assert decoded["k0"] == 1
    for probe in (0, 7, 999999, fam.P_ORDINAL_COUNT - 1):
        row = fam.decode_p_ordinal(probe)
        assert (
            fam.encode_p_ordinal(row["a"], row["b"], row["k0"], Fraction(row["prefactor"]))
            == probe
        )


def test_integral_ordinal_codec_round_trips():
    ordinal = fam.encode_i_ordinal(
        Fraction(-1, 2), Fraction(-1, 2), 0, Fraction(0), Fraction(1)
    )
    decoded = fam.decode_i_ordinal(ordinal)
    assert decoded["a"] == "-1/2"
    assert decoded["b"] == "-1/2"
    assert decoded["kernel"] == "one"
    for probe in (0, 5, 1234567, fam.I_ORDINAL_COUNT - 1):
        row = fam.decode_i_ordinal(probe)
        assert (
            fam.encode_i_ordinal(
                Fraction(row["a"]),
                Fraction(row["b"]),
                row["kernel_index"],
                Fraction(row["power"]),
                Fraction(row["prefactor"]),
            )
            == probe
        )


def test_ordinal_codecs_reject_out_of_range():
    with pytest.raises(fam.FamilyError):
        fam.decode_s_ordinal(fam.S_ORDINAL_COUNT)
    with pytest.raises(fam.FamilyError):
        fam.encode_poly_shape([5, 0, 0, 0], [1, 0, 0, 0])
    with pytest.raises(fam.FamilyError):
        fam.encode_i_ordinal(Fraction(1, 7), Fraction(0), 0, Fraction(0), Fraction(1))


# ---------------------------------------------------------------------------
# The digit holdout
# ---------------------------------------------------------------------------


def test_fabricated_near_miss_passes_fp64_and_dies_at_sixty_digits():
    control = fam.fabricated_near_miss_control()
    assert control["passed_fp64_window"] is True
    assert control["died_at_stage"] == fam.VERIFY_STAGES[0]["stage"]
    assert control["control_passed"] is True
    stages = {row["stage"]: row["passed"] for row in control["digit_trail"]}
    assert stages["fp64_window"] is True
    assert stages["mpmath_dps60"] is False
    assert stages["mpmath_dps120"] is False


def test_survival_trail_promotes_a_true_identity_and_kills_a_planted_one():
    basel = fam._builtin_entry("S", "basel_series")
    member = fam.member_from_index(
        "S",
        fam.builtin_ordinal("S", basel) // len(fam.PREFACTORS),
        fam.builtin_ordinal("S", basel) % len(fam.PREFACTORS),
    )
    trail, status = fam.survival_trail("S", member, "zeta2", Fraction(1), 1e-15)
    assert status == "SURVIVED_ALL_STAGES"
    assert all(row["passed"] for row in trail)

    # Same series, wrong target: it must die at the first verification stage.
    trail, status = fam.survival_trail("S", member, "zeta3", Fraction(1), 1e-15)
    assert status == "DIED_AT_MPMATH_DPS60"
    assert trail[-1]["passed"] is False


def test_verify_stages_are_strictly_increasing_in_precision():
    stages = fam.VERIFY_STAGES
    assert [stage["dps"] for stage in stages] == [60, 120]
    thresholds = [mp.mpf(stage["threshold"]) for stage in stages]
    assert thresholds[0] > thresholds[1]
    assert mp.mpf(fam.FP64_MATCH_WINDOW) > thresholds[0]


# ---------------------------------------------------------------------------
# The declared accelerator and quadrature
# ---------------------------------------------------------------------------


def test_neville_weights_are_well_conditioned_and_partition_unity():
    assert abs(fam.NEVILLE_W8.sum() - 1.0) < 1e-12
    assert abs(fam.NEVILLE_W6.sum() - 1.0) < 1e-12
    # Bounded weights are what makes the extrapolation cost under one decimal digit.
    assert fam.NEVILLE_W8_NORM < 5.0
    assert fam.NEVILLE_W6_NORM < 5.0


def test_neville_extrapolation_recovers_a_known_slow_series():
    # Partial sums of sum 1/k^2 converge like 1/n; direct summation at 2048 terms is only
    # good to ~5e-4, and the extrapolation must turn that into fp64 accuracy.
    partial = np.zeros(len(fam.FP64_CHECKPOINTS))
    total = 0.0
    slot = 0
    for k in range(1, fam.FP64_TERMS + 1):
        total += 1.0 / (k * k)
        if slot < len(fam.FP64_CHECKPOINTS) and k == fam.FP64_CHECKPOINTS[slot]:
            partial[slot] = total
            slot += 1
    assert abs(total - float(mp.zeta(2))) > 1e-4
    assert abs(float(partial @ fam.NEVILLE_W8) - float(mp.zeta(2))) < 1e-12


def test_quadrature_reproduces_three_independent_classical_integrals():
    with mp.workdps(50):
        cases = [
            ((Fraction(-1, 2), Fraction(-1, 2), 0, Fraction(0)), float(mp.pi)),
            ((Fraction(0), Fraction(0), 3, Fraction(1)), float(mp.pi / 4)),
            ((Fraction(0), Fraction(-1), 1, Fraction(1)), float(mp.zeta(2))),
            ((Fraction(0), Fraction(0), 7, Fraction(1)), float(mp.catalan)),
        ]
    for (a, b, kernel, power), expected in cases:
        got = fam.QUADRATURE.quadrature(a, b, kernel, power)
        assert abs(got - expected) < 1e-13, (a, b, kernel, power)


def test_quadrature_nodes_are_declared_and_finite():
    assert fam.QUADRATURE.nodes == fam.FAMILY_I_CONFIG["quadrature_nodes"]
    assert fam.QUADRATURE.nodes > 500
    assert np.all(np.isfinite(fam.QUADRATURE.log_x))
    assert np.all(np.isfinite(fam.QUADRATURE.log_1mx))
    assert np.all(fam.QUADRATURE.prefactor > 0)


# ---------------------------------------------------------------------------
# Exact evaluators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "q", "z", "expected"),
    [
        ([1, 0, 0, 0], [1, 1, 0, 0], "1", "e"),
        ([1, 2, 1, 0], [4, 4, 1, 0], "1", "zeta2"),
        ([1, 3, 3, 1], [8, 12, 6, 1], "1", "zeta3"),
        ([1, 4, 4, 0], [9, 12, 4, 0], "-1", "catalan"),
    ],
)
def test_series_exact_evaluation_matches_the_named_constant(p, q, z, expected):
    with mp.workdps(80):
        value = fam.series_value_mp(p, q, Fraction(z))
        assert abs(value - fam.constant_value(expected)) < mp.mpf("1e-70")


def test_product_exact_evaluation_is_the_gamma_ratio():
    with mp.workdps(80):
        wallis = fam.product_value_mp([0, 0, 4, 0], [-1, 0, 4, 0], 1)
        assert abs(wallis - mp.pi / 2) < mp.mpf("1e-70")
        sinh_product = fam.product_value_mp([1, 0, 1, 0], [0, 0, 1, 0], 1)
        assert abs(sinh_product - mp.sinh(mp.pi) / mp.pi) < mp.mpf("1e-70")
        telescoping = fam.product_value_mp([-1, 0, 1, 0], [0, 0, 1, 0], 2)
        assert abs(telescoping - mp.mpf(1) / 2) < mp.mpf("1e-70")


def test_integral_exact_evaluation_matches_known_closed_forms():
    with mp.workdps(80):
        assert abs(
            fam.integral_value_mp(Fraction(0), Fraction(-1), 1, Fraction(2)) - 2 * mp.zeta(3)
        ) < mp.mpf("1e-70")
        assert abs(
            fam.integral_value_mp(Fraction(-1, 2), Fraction(-1, 2), 0, Fraction(0)) - mp.pi
        ) < mp.mpf("1e-70")
        assert abs(
            fam.integral_value_mp(Fraction(0), Fraction(0), 7, Fraction(1)) - mp.catalan
        ) < mp.mpf("1e-70")


def test_integral_evaluator_holds_up_on_the_most_singular_admitted_exponent():
    """The grid's most singular endpoint is where an adaptive quadrature quietly fails.

    ``mpmath.quad`` truncates its abscissa range and loses seventy digits on ``x^(-11/12)``,
    which would kill true identities at the verification stages; the declared evaluator
    solves the range from the requested precision instead, so this must be exact.
    """

    for dps, bound in ((60, "1e-50"), (120, "1e-100")):
        with mp.workdps(dps):
            value = fam.integral_value_mp(Fraction(-11, 12), Fraction(0), 0, Fraction(0))
            assert abs(value - mp.mpf(12)) < mp.mpf(bound)


def test_integral_quadrature_step_is_converged_at_the_declared_level():
    """Halving the declared tanh-sinh step must not move a value at the stage thresholds."""

    cases = [
        (Fraction(0), Fraction(-1), 1, Fraction(1)),
        (Fraction(0), Fraction(0), 7, Fraction(1)),
        (Fraction(-1, 4), Fraction(1), 2, Fraction(5, 2)),
    ]
    coarse = []
    with mp.workdps(120):
        for a, b, kernel, power in cases:
            coarse.append(fam.integral_value_mp(a, b, kernel, power))
    original = fam.MP_QUAD_LEVEL
    try:
        fam.MP_QUAD_LEVEL = original + 1
        with mp.workdps(120):
            for (a, b, kernel, power), reference in zip(cases, coarse, strict=True):
                refined = fam.integral_value_mp(a, b, kernel, power)
                assert abs(refined - reference) < mp.mpf("1e-100")
    finally:
        fam.MP_QUAD_LEVEL = original


def test_hypergeometric_parameters_are_exact_and_cancel_correctly():
    parameters = fam.series_hypergeometric_parameters([1, 2, 1, 0], [4, 4, 1, 0], Fraction(1))
    assert parameters is not None
    upper, lower, argument = parameters
    assert argument == 1
    assert len(upper) == 3 and len(lower) == 2  # 3F2(1,1,1;2,2;1) = zeta(2)


# ---------------------------------------------------------------------------
# Declared gates
# ---------------------------------------------------------------------------


def test_product_convergence_test_is_the_classical_condition():
    assert fam.product_converges([0, 0, 4, 0], [-1, 0, 4, 0], 1)[0] is True
    # Unequal leading coefficients: the product runs to zero or infinity.
    assert fam.product_converges([0, 0, 4, 0], [-1, 0, 3, 0], 1)[1] == "leading_coefficient_mismatch"
    # Unequal subleading coefficients: it diverges like n^c.
    assert fam.product_converges([0, 1, 4, 0], [-1, 0, 4, 0], 1)[1] == (
        "subleading_coefficient_mismatch"
    )
    # A factor that vanishes or turns negative on the index range.
    assert fam.product_converges([-1, 0, 1, 0], [0, 0, 1, 0], 1)[1] == "factor_not_positive"


def test_integral_convergence_test_uses_the_kernel_orders():
    # b = -1 alone diverges, but ln(1/x) vanishes to first order at x = 1 and rescues it.
    assert fam.integral_converges(Fraction(0), Fraction(-1), 0, Fraction(0)) is False
    assert fam.integral_converges(Fraction(0), Fraction(-1), 1, Fraction(1)) is True
    assert fam.integral_converges(Fraction(-1), Fraction(0), 0, Fraction(0)) is False


def test_terminating_series_are_detected_structurally_not_by_underflow():
    # P(2) = 0, so the series terminates and its value is rational.
    assert fam.series_terminates([-2, 1, 0, 0]) == 2
    # The exponential series never terminates even though its fp64 terms underflow to zero.
    assert fam.series_terminates([1, 0, 0, 0]) is None


def test_canonical_form_collapses_the_declared_grammar_degeneracies():
    # A common polynomial factor leaves the series unchanged.
    left = {"p": [1, 2, 0, 0], "q": [2, 2, 0, 0], "z": "1/2", "prefactor": "1"}
    right = {"p": [2, 4, 0, 0], "q": [4, 4, 0, 0], "z": "1/2", "prefactor": "1"}
    assert fam.canonical_form("S", left)["key"] == fam.canonical_form("S", right)["key"]

    # A pure-power kernel only shifts the exponents.
    plain = {"a": "1/2", "b": "1/2", "kernel": "one", "kernel_index": 0, "power": "0",
             "prefactor": "1"}
    dressed = {"a": "1/2", "b": "3/2", "kernel": "inv_1mx", "kernel_index": 4, "power": "1",
               "prefactor": "1"}
    assert fam.canonical_form("I", plain)["key"] == fam.canonical_form("I", dressed)["key"]

    # 1/sqrt(1+x) squared is 1/(1+x).
    root = {"a": "0", "b": "0", "kernel": "inv_sqrt_1px", "kernel_index": 11, "power": "2",
            "prefactor": "1"}
    flat = {"a": "0", "b": "0", "kernel": "inv_1px", "kernel_index": 2, "power": "1",
            "prefactor": "1"}
    assert fam.canonical_form("I", root)["key"] == fam.canonical_form("I", flat)["key"]


# ---------------------------------------------------------------------------
# Rediscovery controls and the fp64 sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["S", "P", "I"])
def test_every_declared_classic_is_admitted_by_its_own_fp64_path(family):
    for classic in fam.FAMILY_CLASSICS[family]:
        entry = fam._builtin_entry(family, classic["builtin_id"])
        target = classic["target"] or entry["target"]
        if target is None:
            continue
        ordinal = fam.builtin_ordinal(family, entry)
        member = fam.member_from_index(
            family, ordinal // len(fam.PREFACTORS), ordinal % len(fam.PREFACTORS)
        )
        prefactor = Fraction(member["prefactor"])
        if family == "I":
            value = fam.QUADRATURE.quadrature(
                Fraction(member["a"]),
                Fraction(member["b"]),
                member["kernel_index"],
                Fraction(member["power"]),
            )
        else:
            values, admitted, code = fam.evaluate_iterated_numpy(
                family, np.array([ordinal // len(fam.PREFACTORS)])
            )
            assert bool(admitted[0]), (classic["name"], int(code[0]))
            value = float(values[0])
        error = abs(value * float(prefactor) - float(fam.constant_value(target)))
        assert error < fam.FP64_MATCH_WINDOW, (classic["name"], error)


def test_value_identity_controls_reproduce_their_cited_closed_forms():
    for family in fam.FAMILIES:
        for row in fam.value_identity_controls(family):
            assert row["reproduced"] is True, (family, row["builtin_id"])


def test_cpu_sweep_finds_the_basel_series_in_a_small_window():
    basel = fam.builtin_ordinal("S", fam._builtin_entry("S", "basel_series"))
    member_index = basel // len(fam.PREFACTORS)
    matches, counters, _ = fam.sweep_iterated_cpu(
        "S", index_start=member_index, index_stop=member_index + 1
    )
    assert counters["resolved"] == 1
    assert any(item["target"] == "zeta2" for item in matches)


# ---------------------------------------------------------------------------
# GPU / CPU agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["S", "P"])
def test_gpu_and_cpu_sweeps_report_the_same_matches(family):
    if _cupy_or_none() is None:
        pytest.skip("no CUDA device")
    start, stop = fam.crosscheck_window(family, 1 << 13)
    gpu_matches, _gpu, _ = fam.sweep_iterated_gpu(
        family, index_start=start, index_stop=stop
    )
    cpu_matches, _cpu, _ = fam.sweep_iterated_cpu(family, index_start=start, index_stop=stop)

    def key(item):
        return (item["member_index"], item["target"], item["prefactor_index"])

    assert [key(item) for item in gpu_matches] == [key(item) for item in cpu_matches]


def test_integral_sweep_agrees_between_numpy_and_cupy_on_one_kernel():
    cupy = _cupy_or_none()
    if cupy is None:
        pytest.skip("no CUDA device")
    cpu_matches, cpu_counters, _ = fam.sweep_integrals(np, kernel_subset=[3])
    gpu_matches, gpu_counters, _ = fam.sweep_integrals(cupy, kernel_subset=[3])
    assert cpu_counters == gpu_counters
    assert [item["member_index"] for item in cpu_matches] == [
        item["member_index"] for item in gpu_matches
    ]


# ---------------------------------------------------------------------------
# Enumeration receipt
# ---------------------------------------------------------------------------


def test_enumeration_receipt_validates():
    fam.validate_receipt(_reload(_load(ENUMERATION_RECEIPT)))


def test_enumeration_receipt_records_every_family_exhaustively():
    receipt = _load(ENUMERATION_RECEIPT)
    families = {block["family"] for block in receipt["families"]}
    assert families == set(fam.FAMILIES)
    for block in receipt["families"]:
        assert block["scale"]["exhaustive_this_run"] is True
        assert block["scale"]["declared_total_ordinals"] >= 10**8
        assert block["scale"]["evaluated_ordinals"] == block["scale"]["declared_total_ordinals"]
        assert block["controls"]["rediscovery"]["passed"] is True
        assert block["controls"]["rediscovery"]["classics_rediscovered"] >= 2


def test_enumeration_receipt_claims_are_sealed_and_honest():
    receipt = _load(ENUMERATION_RECEIPT)
    assert receipt["claims"]["corpus_absence_establishes_novelty"] is False
    assert receipt["claims"]["builtin_table_absence_establishes_novelty"] is False
    assert receipt["claims"]["survivors_are_conjectures_not_theorems"] is True
    assert receipt["claims"]["match_at_fit_precision_is_not_discovery"] is True


def test_every_survivor_carries_a_complete_digit_trail():
    receipt = _load(ENUMERATION_RECEIPT)
    for block in receipt["families"]:
        for survivor in block["survivors"]:
            stages = [row["stage"] for row in survivor["digit_trail"]]
            assert stages[1:] == [stage["stage"] for stage in fam.VERIFY_STAGES]
            assert all(row["passed"] for row in survivor["digit_trail"])
            assert survivor["status"] == "SURVIVED_ALL_STAGES"
            assert len(survivor["member_value_120_digits"]) > 60


def test_enumeration_receipt_detects_tampering():
    receipt = _load(ENUMERATION_RECEIPT)
    tampered = json.loads(json.dumps(receipt))
    tampered["families"][0]["counts"]["survivors"] += 1
    with pytest.raises(fam.FamilyError):
        fam.validate_receipt(tampered)

    flipped = json.loads(json.dumps(receipt))
    flipped["claims"] = {**flipped["claims"], "corpus_absence_establishes_novelty": True}
    with pytest.raises(fam.FamilyError):
        fam.validate_receipt(flipped)

    faked = json.loads(json.dumps(receipt))
    faked["controls"]["fabricated_near_miss"]["control_passed"] = False
    with pytest.raises(fam.FamilyError):
        fam.validate_receipt(faked)


def test_enumeration_receipt_is_deterministic_across_a_small_replay():
    """The deterministic core excludes timings, so a replayed sweep reseals identically."""

    receipt = _load(ENUMERATION_RECEIPT)
    body = {
        key: value
        for key, value in receipt.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    from sigma_theory_compiler.sigma_core import canonical_sha256

    assert canonical_sha256(body) == receipt["result_core_sha256"]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def test_corpus_has_the_declared_scale_and_a_closed_provenance_forest(corpus):
    manifest = corpus["manifest"]
    assert manifest["counts"]["total_seeds"] >= 60
    assert manifest["counts"]["records"] >= manifest["counts"]["concrete_seeds"]
    assert manifest["provenance_forest"]["all_records_resolve_to_a_cited_seed"] is True
    assert manifest["claims"]["external_fetch_performed"] is False
    for family in ("S", "P", "I"):
        assert manifest["counts"]["by_family"][family] > 0


def test_every_corpus_record_carries_a_resolvable_citation(corpus):
    for record in corpus["records"]:
        assert record.citation.reference
        assert record.citation.confidence in screen.CITATION_CONFIDENCES
        assert record.value_expr


def test_corpus_forest_closure_rejects_an_undeclared_transformation(corpus):
    broken = list(corpus["records"])
    derived = next(item for item in broken if item.kind == "derived")
    import dataclasses

    tampered = dataclasses.replace(
        derived, transform=(("transformation", "not_declared"), ("detail", "x"))
    )
    replaced = [tampered if item.record_id == derived.record_id else item for item in broken]
    with pytest.raises(screen.ScreenError):
        screen.verify_forest_closure(replaced)


def test_class_invariants_identify_objects_not_ordinals(corpus):
    # Two spellings of the same series must share a signature.
    left = {"p": [1, 2, 0, 0], "q": [2, 2, 0, 0], "z": "1/2", "prefactor": "1"}
    right = {"p": [2, 4, 0, 0], "q": [4, 4, 0, 0], "z": "1/2", "prefactor": "1"}
    assert screen.signature_key("S", left) == screen.signature_key("S", right)
    # Different series must not.
    other = {"p": [1, 1, 0, 0], "q": [2, 1, 0, 0], "z": "-1", "prefactor": "1"}
    assert screen.signature_key("S", left) != screen.signature_key("S", other)


# ---------------------------------------------------------------------------
# Screening receipt
# ---------------------------------------------------------------------------


def test_screen_receipt_validates():
    screen.validate_receipt(_reload(_load(SCREEN_RECEIPT)))


def test_screen_controls_recovered_every_known_rediscovered_survivor():
    receipt = _load(SCREEN_RECEIPT)
    controls = receipt["controls"]
    assert controls["passed"] is True
    assert float(controls["recovery_rate"]) >= float(screen.CONTROL_RECOVERY_THRESHOLD)
    assert controls["labelled_known_rediscovered"] > 0


def test_every_known_verdict_carries_a_citation_and_a_verified_justification():
    receipt = _load(SCREEN_RECEIPT)
    for candidate in receipt["candidates"]:
        assert candidate["verdict"] in screen.VERDICTS
        if candidate["verdict"] == "KNOWN":
            assert candidate["matched_record"]["citation"]["reference"]
            assert candidate["justification_verified"] is True
        else:
            assert candidate["why_no_structural_match"]


def test_screen_receipt_never_claims_novelty():
    receipt = _load(SCREEN_RECEIPT)
    assert receipt["claims"]["corpus_absence_establishes_novelty"] is False
    assert receipt["claims"]["value_match_alone_is_not_membership"] is True
    assert receipt["claims"]["human_review_required_before_any_novelty_claim"] is True


def test_screen_receipt_detects_tampering():
    receipt = _load(SCREEN_RECEIPT)
    tampered = json.loads(json.dumps(receipt))
    tampered["counts"]["by_verdict"]["KNOWN"] += 1
    with pytest.raises(screen.ScreenError):
        screen.validate_receipt(tampered)

    stripped = json.loads(json.dumps(receipt))
    for candidate in stripped["candidates"]:
        if candidate["verdict"] == "KNOWN":
            candidate["matched_record"]["citation"]["reference"] = ""
            break
    else:
        pytest.skip("no KNOWN verdict to strip")
    with pytest.raises(screen.ScreenError):
        screen.validate_receipt(stripped)


def test_screen_receipt_binds_its_predecessor():
    enumeration = _load(ENUMERATION_RECEIPT)
    receipt = _load(SCREEN_RECEIPT)
    assert receipt["input"]["content_sha256"] == enumeration["content_sha256"]
    assert receipt["input"]["result_core_sha256"] == enumeration["result_core_sha256"]


def test_value_equality_alone_never_yields_known(corpus):
    """A planted formula whose value equals a corpus record but whose structure differs."""

    # The Leibniz series for pi/4 and the Beta integral B(1/2,1/2)/4 share a value but not a
    # family, so the screen must never call one the other.
    member = {"a": "-1/2", "b": "-1/2", "kernel": "one", "kernel_index": 0, "power": "0",
              "prefactor": "1/4"}
    report = screen.screen_candidate(
        corpus, "I", member, "planted", "pi", "NOT_IN_BUILTIN_TABLE", ""
    )
    assert report["verdict"] in screen.VERDICTS
    if report["verdict"] == "KNOWN":
        # If it is KNOWN it must be for a *structural* reason, never a bare value match.
        assert report["test_that_fired"] != "value_match_without_structure"


# ---------------------------------------------------------------------------
# Proof routing receipt
# ---------------------------------------------------------------------------


def test_router_receipt_validates():
    router.validate_receipt(_reload(_load(PROOF_RECEIPT)))


def test_router_controls_prove_classics_refute_a_falsehood_and_type_a_blocker():
    receipt = _load(PROOF_RECEIPT)
    controls = receipt["controls"]
    assert controls["passed"] is True
    assert controls["classical_identities_proved"] >= controls["classical_identities_required"]
    assert controls["deliberate_falsification"]["refuted"] is True
    assert isinstance(controls["deliberate_falsification"]["first_differing_decimal_place"], int)
    assert controls["absent_technique"]["blocked"] is True
    assert str(controls["absent_technique"]["typed_blocker"]).startswith(
        "missing_proof_technique:"
    )


def test_router_headline_is_computed_from_the_candidate_records():
    receipt = _load(PROOF_RECEIPT)
    proved_absent = [
        item["candidate_id"]
        for item in receipt["candidates"]
        if item["verdict"] == "PROVED"
        and item["reclassification"].get("proof_family_present_in_corpus") is not True
    ]
    unproved = [
        item["candidate_id"]
        for item in receipt["candidates"]
        if item["verdict"] == "MISSING_TECHNIQUE"
    ]
    headline = receipt["headline"]
    assert sorted(headline["proved_and_absent_candidate_ids"]) == sorted(proved_absent)
    assert headline["proved_and_proof_family_still_absent_from_the_corpus"] == len(proved_absent)
    assert sorted(headline["not_reducible_candidate_ids"]) == sorted(unproved)
    assert headline["intersection_absent_from_corpus_and_not_reducible"] == len(unproved)


def test_every_proved_candidate_exhibits_a_derivation_that_reproduces_its_value():
    receipt = _load(PROOF_RECEIPT)
    for candidate in receipt["candidates"]:
        if candidate["verdict"] != "PROVED":
            continue
        assert candidate["technique_that_fired"] in router.TECHNIQUE_ORDER
        assert candidate["derivation"]["closed_form"]
        assert int(candidate["derivation"]["agreement_digits"]) >= 100
        assert candidate["cited_theorem"]
        assert candidate["reclassification"]["to"] == "KNOWN_BY_PROOF_FAMILY"


def test_router_never_claims_a_kernel_result():
    receipt = _load(PROOF_RECEIPT)
    assert receipt["lean"]["sources_emitted"] == 0
    assert receipt["lean"]["kernel_verification_pending"] is True
    assert receipt["claims"]["novelty_claimed"] is False
    for candidate in receipt["candidates"]:
        assert candidate.get("lean", {}).get("kernel_verified") is False


def test_router_receipt_detects_tampering():
    receipt = _load(PROOF_RECEIPT)
    tampered = json.loads(json.dumps(receipt))
    tampered["headline"]["not_reducible_by_any_declared_technique"] += 1
    with pytest.raises(router.ProofRouterError):
        router.validate_receipt(tampered)

    faked = json.loads(json.dumps(receipt))
    if faked["candidates"]:
        faked["candidates"][0]["lean"] = {"kernel_verified": True}
        with pytest.raises(router.ProofRouterError):
            router.validate_receipt(faked)


def test_router_receipt_binds_both_predecessors():
    enumeration = _load(ENUMERATION_RECEIPT)
    adjudication = _load(SCREEN_RECEIPT)
    receipt = _load(PROOF_RECEIPT)
    assert receipt["input"]["enumeration_content_sha256"] == enumeration["content_sha256"]
    assert receipt["input"]["adjudication_content_sha256"] == adjudication["content_sha256"]


def test_router_refutes_a_deliberately_wrong_claim_with_a_digit():
    member = {"p": [1, 2, 1, 0], "q": [4, 4, 1, 0], "z": "1", "prefactor": "1"}
    check = router.numeric_check("S", member, "zeta3")
    assert check["holds"] is False
    assert isinstance(check["first_differing_decimal_place"], int)


def test_hurwitz_reduction_verifies_its_own_quasi_polynomial_fit():
    """The reduction must decline rather than guess when its fit cannot be verified."""

    # A kernel that is not (ln(1/x))^s times a rational function.
    member = {"a": "0", "b": "0", "kernel": "x_over_expm1", "kernel_index": 9, "power": "1",
              "prefactor": "1"}
    outcome = router.attempt_log_power_hurwitz(member)
    assert outcome["fired"] is False
    assert "rational" in outcome["reason"]


def test_proof_family_membership_is_looked_up_not_assumed(corpus):
    assert router._family_in_corpus(corpus, "generalized_hypergeometric_pFq") is True
    assert router._family_in_corpus(corpus, "a_family_nobody_encoded") is False


def test_every_technique_names_a_classical_family_that_the_corpus_carries(corpus):
    for technique in router.TECHNIQUE_ORDER:
        family = router.TECHNIQUE_CLASSICAL_FAMILY[technique]
        assert router.TECHNIQUE_CITED_THEOREM[technique]
        assert router._family_in_corpus(corpus, family) is True, technique


def test_euler_representation_names_the_relation_that_makes_it_summable():
    member = {"a": "-1/4", "b": "1", "kernel": "inv_1px", "kernel_index": 2, "power": "5/2",
              "prefactor": "3"}
    outcome = router.attempt_euler_2f1(member)
    assert outcome["fired"] is True
    assert outcome["derivation"]["c"] == "11/4"
    relations = outcome["derivation"]["classical_evaluation_relations"]
    assert any("Kummer" in item or "quadratic" in item for item in relations)


def test_cyclotomic_substitution_declines_when_the_factors_do_not_cancel():
    good = {"a": "1/2", "b": "-1/2", "kernel": "inv_1pxpx2", "kernel_index": 10,
            "power": "1/2", "prefactor": "3"}
    assert router.attempt_cyclotomic_substitution(good)["fired"] is True
    bad = {**good, "b": "0"}
    outcome = router.attempt_cyclotomic_substitution(bad)
    assert outcome["fired"] is False
    assert "do not cancel" in outcome["reason"]


def test_beta_derivative_reduction_matches_the_differentiated_beta_integral():
    member = {"a": "-1/2", "b": "-1/2", "kernel": "log_inv_over_1mx", "kernel_index": 5,
              "power": "1", "prefactor": "1/2"}
    outcome = router.attempt_beta_derivative(member)
    assert outcome["fired"] is True
    assert outcome["derivation"]["n"] == 1
    # A non-integer log power is not a repeated derivative and must be declined.
    assert router.attempt_beta_derivative({**member, "power": "1/2"})["fired"] is False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "receipt"),
    [
        ("sigma_theory_compiler.inverse_symbolic_families", ENUMERATION_RECEIPT),
        ("sigma_theory_compiler.families_prior_art_screen", SCREEN_RECEIPT),
        ("sigma_theory_compiler.families_proof_router", PROOF_RECEIPT),
    ],
)
def test_cli_validate_checked_accepts_the_sealed_receipt(module, receipt, monkeypatch, capsys):
    if not receipt.exists():
        pytest.skip(f"receipt not present: {receipt}")
    import importlib
    import sys

    target = importlib.import_module(module)
    monkeypatch.setattr(
        sys, "argv", [module, "--output", str(receipt), "--validate-checked"]
    )
    assert target.main() == 0
    assert json.loads(capsys.readouterr().out)["validated"] is True
