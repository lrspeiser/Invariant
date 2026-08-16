"""Gates for the inverse symbolic engine.

The engine's one non-negotiable idea is the digit holdout: a match found at fit precision is
interpolation, and only survival at much higher precision may promote it — to a conjecture,
never to a theorem.  These tests pin that guard with a fabricated fake that must die, pin the
PSLQ and continued-fraction rediscovery controls, the ordinal codec, the deduplicated Moebius
family, CPU/GPU agreement, receipt determinism, and tamper detection.  GPU tests skip cleanly
when no CUDA device is present.
"""

from __future__ import annotations

import json
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from sigma_theory_compiler.inverse_symbolic_engine import (
    BUILTIN_KNOWN_TABLE,
    CF_CLAIMS,
    CF_TARGET_NAMES,
    MOBIUS_COUNT,
    MOBIUS_INDEX,
    MOBIUS_TABLE,
    PSLQ_CLAIMS,
    PSLQ_CONFIG,
    SHAPE_COUNT,
    TOTAL_ORDINALS,
    InverseSymbolicError,
    _canonical_mobius,
    cf_value_mp,
    classify_prior_art,
    constant_value,
    decide_ordinals,
    decode_ordinal,
    encode_ordinal,
    relation_normalized_residual,
    render_cf_conjecture,
    run_cf_lane,
    run_pslq_lane,
    survival_trail,
    validate_cf_receipt,
    validate_pslq_receipt,
)


def _cupy_or_none():
    try:
        import cupy

        cupy.arange(4).sum()
        return cupy
    except Exception:  # noqa: BLE001 - any CUDA absence means skip
        return None


def _reload(receipt):
    """Round-trip a receipt through canonical JSON, as a validator would receive it."""

    from sigma_theory_compiler.sigma_core import canonical_json_bytes

    return json.loads(canonical_json_bytes(receipt))


PHI_ORDINAL = encode_ordinal((1, 0, 0), (1, 0, 0), (1, 0, 0, 1))
SILVER_SQRT2_ORDINAL = encode_ordinal((2, 0, 0), (1, 0, 0), (1, -1, 0, 1))
EULER_E_TAIL_ORDINAL = encode_ordinal((1, 1, 0), (0, 1, 0), (2, 1, 1, 0))
EULER_E_ALT_ORDINAL = encode_ordinal((3, 1, 0), (0, -1, 0), (1, 0, 0, 1))
LAMBERT_COTH_HALF_ORDINAL = encode_ordinal((2, 4, 0), (1, 0, 0), (1, 1, 1, -1))


@pytest.fixture(scope="module")
def small_cf_receipt():
    """CPU run over a shape prefix that contains real matches (the 4-2/x sqrt2 family)."""

    return run_cf_lane(use_gpu=False, shape_limit=140000)


@pytest.fixture(scope="module")
def pslq_receipt():
    return run_pslq_lane()


# ---------------------------------------------------------------------------
# Family declaration and ordinal codec
# ---------------------------------------------------------------------------


def test_family_size_is_the_declared_product():
    assert SHAPE_COUNT == 9**6 == 531_441
    assert MOBIUS_COUNT == 224
    assert TOTAL_ORDINALS == SHAPE_COUNT * MOBIUS_COUNT == 119_042_784


def test_mobius_table_is_deduplicated_and_includes_identity():
    assert (1, 0, 0, 1) in MOBIUS_INDEX
    assert len(set(MOBIUS_TABLE)) == MOBIUS_COUNT
    for p, q, r, s in MOBIUS_TABLE:
        assert p * s - q * r != 0
        assert _canonical_mobius(p, q, r, s) == (p, q, r, s)
    # Scalar multiples collapse onto one class representative.
    assert _canonical_mobius(2, 0, 0, 2) == (1, 0, 0, 1)
    assert _canonical_mobius(-1, 0, 0, -1) == (1, 0, 0, 1)
    assert _canonical_mobius(-2, 4, 2, -2) == (1, -2, -1, 1)


@pytest.mark.parametrize(
    "ordinal",
    [0, 1, MOBIUS_COUNT - 1, MOBIUS_COUNT, TOTAL_ORDINALS - 1, 59_684_886],
)
def test_ordinal_codec_roundtrip(ordinal):
    decoded = decode_ordinal(ordinal)
    assert decoded["ordinal"] == ordinal
    assert 0 <= decoded["shape_index"] < SHAPE_COUNT
    assert 0 <= decoded["mobius_index"] < MOBIUS_COUNT
    assert all(-4 <= value <= 4 for value in decoded["alpha"] + decoded["beta"])
    assert encode_ordinal(decoded["alpha"], decoded["beta"], decoded["mobius"]) == ordinal


def test_out_of_range_ordinals_and_coefficients_are_refused():
    with pytest.raises(InverseSymbolicError):
        decode_ordinal(-1)
    with pytest.raises(InverseSymbolicError):
        decode_ordinal(TOTAL_ORDINALS)
    with pytest.raises(InverseSymbolicError):
        encode_ordinal((5, 0, 0), (1, 0, 0), (1, 0, 0, 1))
    with pytest.raises(InverseSymbolicError):
        encode_ordinal((1, 0, 0), (1, 0, 0), (1, 1, 1, 1))  # determinant zero


# ---------------------------------------------------------------------------
# fp64 matcher: classical rediscoveries and structural refusals
# ---------------------------------------------------------------------------


def test_classical_cfs_match_their_targets_in_fp64():
    ordinals = np.array(
        [
            PHI_ORDINAL,
            SILVER_SQRT2_ORDINAL,
            EULER_E_TAIL_ORDINAL,
            EULER_E_ALT_ORDINAL,
            LAMBERT_COTH_HALF_ORDINAL,
        ],
        dtype=np.int64,
    )
    decisions = decide_ordinals(np, ordinals)
    expected = ["phi", "sqrt2", "e", "e", "e"]
    for row, target in zip(decisions, expected, strict=True):
        assert row[CF_TARGET_NAMES.index(target)]
        assert int(row.sum()) == 1


def test_rational_limit_shape_matches_nothing():
    # x = 2 - 1/x converges to the rational 1 (double root), and only linearly.
    ordinal = encode_ordinal((2, 0, 0), (-1, 0, 0), (1, 0, 0, 1))
    decisions = decide_ordinals(np, np.array([ordinal], dtype=np.int64))
    assert not decisions.any()


def test_oscillating_shape_is_rejected_by_the_convergence_gate():
    # a_n = 0: x -> b/x never settles; the depth-40 vs depth-39 gate must refuse it.
    ordinal = encode_ordinal((0, 0, 0), (1, 0, 0), (1, 0, 0, 1))
    decisions = decide_ordinals(np, np.array([ordinal], dtype=np.int64))
    assert not decisions.any()


def test_gpu_and_cpu_decisions_agree_on_a_sample():
    cupy = _cupy_or_none()
    if cupy is None:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(7)
    sample = rng.choice(TOTAL_ORDINALS, size=1024, replace=False).astype(np.int64)
    sample = np.unique(np.concatenate([sample, [PHI_ORDINAL, EULER_E_TAIL_ORDINAL]]))
    cpu = decide_ordinals(np, sample)
    gpu = decide_ordinals(cupy, cupy.asarray(sample)).get()
    assert int((cpu != gpu).sum()) == 0
    assert bool(cpu[np.where(sample == PHI_ORDINAL)[0][0], CF_TARGET_NAMES.index("phi")])


# ---------------------------------------------------------------------------
# The digit holdout: fabricated fakes must die, true structures must survive
# ---------------------------------------------------------------------------


def test_fabricated_near_match_dies_at_60_digits():
    """A rational forged to sit 1e-13 from Catalan passes the fp64 gate and must die."""

    with mp.workdps(40):
        catalan = constant_value("catalan")
        fake = Fraction(float(catalan)).limit_denominator(3_000_000)
        fp64_error = abs(float(fake) - float(catalan))
    assert fp64_error < 1e-12  # it *would* be reported by a fit-precision-only matcher
    assert fp64_error > 1e-16  # and it is genuinely wrong, not merely rounded

    def fake_value_at_depth(depth: int) -> mp.mpf:
        return mp.mpf(fake.numerator) / fake.denominator

    trail, status = survival_trail(fake_value_at_depth, "catalan", fp64_error)
    assert status == "DIED_AT_MPMATH_DEPTH2000_DPS60"
    assert trail[0]["passed"] is True
    assert trail[1]["passed"] is False
    assert len(trail) == 2  # the 120-digit stage must never run for a stage-2 corpse


def test_true_structure_survives_every_stage():
    trail, status = survival_trail(
        lambda depth: cf_value_mp((1, 0, 0), (1, 0, 0), depth), "phi", 0.0
    )
    assert status == "SURVIVED_ALL_STAGES"
    assert [entry["passed"] for entry in trail] == [True, True, True]


def test_builtin_table_reachable_entries_survive_their_own_wraps():
    for entry in BUILTIN_KNOWN_TABLE:
        if entry["target"] is None:
            continue
        alpha, beta, mobius = entry["alpha"], entry["beta"], entry["mobius"]

        def value_at_depth(depth: int, alpha=alpha, beta=beta, mobius=mobius) -> mp.mpf:
            p, q, r, s = mobius
            value = cf_value_mp(alpha, beta, depth)
            return (p * value + q) / (r * value + s)

        _, status = survival_trail(value_at_depth, entry["target"], 0.0)
        assert status == "SURVIVED_ALL_STAGES", entry["id"]


def test_builtin_table_documented_pi_family_values_are_what_they_claim():
    for entry in BUILTIN_KNOWN_TABLE:
        if entry["target"] is not None:
            continue
        depth = 20_000 if entry["convergence"] == "slow" else 2_000
        tolerance = 1e-3 if entry["convergence"] == "slow" else 1e-6
        with mp.workdps(30):
            value = cf_value_mp(entry["alpha"], entry["beta"], depth)
            error = abs(value - mp.mpf(entry["value_decimal_approx"]))
        assert error < tolerance, entry["id"]


# ---------------------------------------------------------------------------
# Prior-art labeling: rediscovered vs not-in-table, never "novel"
# ---------------------------------------------------------------------------


def test_prior_art_exact_builtin_shape():
    verdict = classify_prior_art((1, 0, 0), (1, 0, 0), "phi")
    assert verdict["label"] == "KNOWN_REDISCOVERED"
    assert verdict["basis"] == "exact_builtin_shape"
    assert verdict["builtin_id"] == "phi_simple_cf"


def test_prior_art_constant_equivalence_scaling():
    # CF(c*a, c^2*b) = c * CF(a, b): a=2, b=4 is the golden-ratio CF scaled by c=2.
    verdict = classify_prior_art((2, 0, 0), (4, 0, 0), "phi")
    assert verdict["label"] == "KNOWN_REDISCOVERED"
    assert verdict["basis"] == "constant_equivalence_scaling_c=2"
    assert verdict["builtin_id"] == "phi_simple_cf"
    negated = classify_prior_art((-3, -1, 0), (0, -1, 0), "e")
    assert negated["basis"] == "constant_equivalence_scaling_c=-1"
    assert negated["builtin_id"] == "euler_e_alternating_cf"


def test_prior_art_quadratic_surd_rule_catches_disguised_periodic_cfs():
    # CF(2n+1, 4n^2-1) is the golden-ratio CF under a variable equivalence transform.
    verdict = classify_prior_art((1, 2, 0), (-1, 0, 4), "phi")
    assert verdict["label"] == "KNOWN_REDISCOVERED"
    assert verdict["basis"] == "quadratic_surd_classical_theory"


def test_prior_art_pi_shape_is_not_in_table_and_never_novel():
    verdict = classify_prior_art((1, 3, 0), (0, 1, -2), "pi")
    assert verdict["label"] == "NOT_IN_BUILTIN_TABLE"
    assert verdict["basis"] == "no_structural_match_in_builtin_table"
    assert CF_CLAIMS["corpus_absence_establishes_novelty"] is False
    assert CF_CLAIMS["builtin_table_absence_establishes_novelty"] is False


def test_render_produces_text_and_mathml_ready_latex():
    rendered = render_cf_conjecture((1, 0, 0), (1, 0, 0), (1, 0, 0, 1), "phi")
    assert rendered["text"].startswith("phi =? x where x = 1 + 1/(1 + 1/(1 + ")
    assert "a_n = 1, b_n = 1" in rendered["text"]
    assert r"\varphi" in rendered["latex"] and r"\cfrac" in rendered["latex"]
    pi_form = render_cf_conjecture((1, 3, 0), (0, 1, -2), (0, 2, 1, 0), "pi")
    assert "pi =? (2)/(x)" in pi_form["text"]
    assert "1 - 1/(4 - 6/(7 - " in pi_form["text"]


# ---------------------------------------------------------------------------
# PSLQ lane: knowns rediscovered, negative control clean, fakes die
# ---------------------------------------------------------------------------


def test_pslq_rediscovers_atan1_as_pi_over_four(pslq_receipt):
    report = next(t for t in pslq_receipt["targets"] if t["name"] == "atan_one")
    assert report["status"] == "RELATION_FOUND_AND_VERIFIED"
    assert report["relation"]["solved_form"] == "atan(1) = (1/4)*pi"
    assert set(report["relation"]["coefficients_on_basis"]) == {"pi"}


def test_pslq_rediscovers_log6_as_ln2_plus_ln3(pslq_receipt):
    report = next(t for t in pslq_receipt["targets"] if t["name"] == "log_six")
    assert report["status"] == "RELATION_FOUND_AND_VERIFIED"
    assert report["relation"]["solved_form"] == "log(6) = ln(2) + ln(3)"


def test_pslq_rediscovers_the_constructed_controls(pslq_receipt):
    mixed = next(
        t for t in pslq_receipt["targets"] if t["name"] == "three_plus_two_pi_over_seven"
    )
    assert mixed["relation"]["solved_form"] == "(3 + 2*pi)/7 = 3/7 + (2/7)*pi"
    tail = next(t for t in pslq_receipt["targets"] if t["name"] == "gamma_plus_two_catalan")
    assert tail["relation"]["solved_form"] == "EulerGamma + 2*Catalan = EulerGamma + 2*Catalan"


def test_pslq_random_target_yields_no_relation_and_records_the_bound(pslq_receipt):
    report = next(t for t in pslq_receipt["targets"] if t["name"] == "sha256_pseudorandom")
    assert report["status"] == "NO_RELATION_FOUND_UNDER_BOUND"
    assert report["relation"] is None
    assert report["searched_bound"]["max_coefficient"] == 1_000_000
    assert "not a proof" in report["searched_bound"]["note"]
    counts = pslq_receipt["counts"]
    assert counts == {
        "targets": 5,
        "relations_found": 4,
        "relations_verified_at_200dps": 4,
        "negative_controls_clean": 1,
    }


def test_pslq_receipt_validates_and_detects_tamper(pslq_receipt):
    loaded = _reload(pslq_receipt)
    validate_pslq_receipt(loaded)
    tampered = _reload(pslq_receipt)
    tampered["counts"]["relations_found"] = 5
    with pytest.raises(InverseSymbolicError):
        validate_pslq_receipt(tampered)
    flipped = _reload(pslq_receipt)
    flipped["claims"]["match_at_fit_precision_is_not_discovery"] = False
    with pytest.raises(InverseSymbolicError):
        validate_pslq_receipt(flipped)


def test_pslq_fake_relation_fits_at_double_precision_but_dies_at_200_digits():
    """1360120*pi - 4272943 = 0 'fits' to 3e-15 normalized; the holdout must kill it."""

    coefficients = [1_360_120, -4_272_943] + [0] * 9
    with mp.workdps(200):
        _, normalized = relation_normalized_residual(+mp.pi, coefficients)
        assert normalized < mp.mpf("1e-12")  # a fit-precision matcher would report it
        assert normalized > mp.mpf(PSLQ_CONFIG["verify_normalized_residual_threshold"])


def test_mandated_honesty_claims_are_pinned_in_both_lanes():
    for claims in (PSLQ_CLAIMS, CF_CLAIMS):
        assert claims["match_at_fit_precision_is_not_discovery"] is True
        assert claims["survival_at_verify_precision_is_conjecture_not_proof"] is True
        assert claims["corpus_absence_establishes_novelty"] is False


# ---------------------------------------------------------------------------
# CF receipts: determinism, validation, tamper, and float-free canonical form
# ---------------------------------------------------------------------------


def test_small_cf_receipt_has_real_survivors_with_full_trails(small_cf_receipt):
    assert small_cf_receipt["family"]["evaluated_ordinals"] == 140_000 * MOBIUS_COUNT
    assert small_cf_receipt["family"]["exhaustive_this_run"] is False
    survivors = small_cf_receipt["survivors"]
    assert survivors, "the 140k-shape prefix contains the 4 - 2/x sqrt2 family"
    for survivor in survivors:
        assert survivor["status"] == "SURVIVED_ALL_STAGES"
        stages = [entry["stage"] for entry in survivor["digit_trail"]]
        assert stages == ["fp64_depth40", "mpmath_depth2000_dps60", "mpmath_depth8000_dps120"]
        assert survivor["prior_art"]["label"] in {"KNOWN_REDISCOVERED", "NOT_IN_BUILTIN_TABLE"}
    per_target = small_cf_receipt["counts"]["per_target"]
    assert set(per_target) == set(CF_TARGET_NAMES)


def test_cf_receipt_is_canonical_and_float_free(small_cf_receipt):
    from sigma_theory_compiler.sigma_core import canonical_json_bytes

    canonical_json_bytes(small_cf_receipt)  # raises SchemaViolation on any float


def test_cf_receipt_validates_and_detects_tamper(small_cf_receipt):
    validate_cf_receipt(_reload(small_cf_receipt))
    tampered = _reload(small_cf_receipt)
    tampered["counts"]["stage3_survivors"] += 1
    with pytest.raises(InverseSymbolicError):
        validate_cf_receipt(tampered)
    # A forger who flips a prior-art label and re-seals the receipt must still be caught
    # by the classification replay, not merely by the hash.
    from sigma_theory_compiler.sigma_core import canonical_sha256

    relabeled = _reload(small_cf_receipt)
    first = relabeled["survivors"][0]
    flipped_label = (
        "KNOWN_REDISCOVERED"
        if first["prior_art"]["label"] == "NOT_IN_BUILTIN_TABLE"
        else "NOT_IN_BUILTIN_TABLE"
    )
    first["prior_art"] = {**first["prior_art"], "label": flipped_label}
    core_body = {
        key: value
        for key, value in relabeled.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    relabeled["result_core_sha256"] = canonical_sha256(core_body)
    body = {key: value for key, value in relabeled.items() if key != "content_sha256"}
    relabeled["content_sha256"] = canonical_sha256(body)
    with pytest.raises(InverseSymbolicError):
        validate_cf_receipt(relabeled)


def test_cf_lane_is_deterministic_over_its_core(small_cf_receipt):
    replay = run_cf_lane(use_gpu=False, shape_limit=140_000)
    assert replay["result_core_sha256"] == small_cf_receipt["result_core_sha256"]
    assert replay["survivors"] == small_cf_receipt["survivors"]
