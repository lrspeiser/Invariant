"""Gates for the compositional expression search.

The module's whole value rests on three things being true: the grammar is generative and
reaches real mathematics, the enumeration is what it says it is, and the chance-match gate is
calibrated rather than asserted.  These tests pin all three.  The rediscovery controls check
that classical identities live inside the declared space and come back out of the evaluator;
the chance-gate tests check that an exact-match spike cannot inflate its own chance model and
that a denser neighbourhood demands more digits; the decoy tests check that a real number with
no closed form can never survive.  GPU tests skip cleanly when no CUDA device is present.
"""

from __future__ import annotations

import functools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from sigma_theory_compiler.compositional_expression_search import (
    BINARY_TOKENS,
    BUILTIN_KNOWN_IDENTITIES,
    CLAIMS,
    CORE_WINDOW,
    DECOY_SEED,
    HLL_REGISTERS,
    INTEGER_TOKENS,
    MATCH_WINDOW,
    MODE_CONFIG,
    MODE_F_NEAR_WINDOW,
    RATIONAL_TOKENS,
    RESULT_SCHEMA,
    SLOT_COUNT,
    STACK_DEPTH,
    STRESS_WINDOW,
    TARGET_SLOTS,
    TOKEN_ARITY,
    TOKEN_COUNT,
    TOKEN_NAMES,
    UNARY_TOKENS,
    WINDOW_EXPONENTS,
    WINDOWS,
    CompositionalSearchError,
    SweepParameters,
    _tokens_from_rpn,
    build_decoy_targets,
    chance_accounting,
    classical_reduction,
    count_valid_programs,
    cuda_source,
    decode_ordinal,
    derive_retention,
    encode_program,
    evaluate_batch_cpu,
    evaluate_mp,
    evaluate_series_batch_cpu,
    evaluate_series_mp,
    hll_estimate,
    hll_registers_from_hashes,
    is_valid_program,
    program_status,
    reachability_controls,
    render_infix,
    render_rpn,
    screen_prior_art,
    target_value_mp,
    to_sympy,
    validate_receipt,
    value_hash_keys,
    verify_candidate,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "runs" / "math" / "compositional" / "search-v1.json"


def _cupy_or_none():
    try:
        import cupy

        cupy.arange(4).sum()
        return cupy
    except Exception:  # noqa: BLE001 - any CUDA absence means skip
        return None


CUPY = _cupy_or_none()
requires_gpu = pytest.mark.skipif(CUPY is None, reason="no CUDA device available")


# ---------------------------------------------------------------------------
# Alphabet and spaces
# ---------------------------------------------------------------------------


def test_alphabet_is_the_declared_twenty_four_tokens():
    assert TOKEN_COUNT == 24
    assert len(INTEGER_TOKENS) == 5
    assert len(RATIONAL_TOKENS) == 4
    assert len(UNARY_TOKENS) == 9
    assert len(BINARY_TOKENS) == 5
    assert TOKEN_NAMES[9] == "k"
    assert sum(1 for arity in TOKEN_ARITY if arity == 0) == 10
    assert sum(1 for arity in TOKEN_ARITY if arity == 1) == 9
    assert sum(1 for arity in TOKEN_ARITY if arity == 2) == 5


def test_declared_space_sizes():
    assert MODE_CONFIG["C"]["alphabet_size"] == 23
    assert MODE_CONFIG["C"]["program_length"] == 9
    assert MODE_CONFIG["C"]["space_size"] == 23**9 == 1_801_152_661_463
    assert MODE_CONFIG["F"]["alphabet_size"] == 24
    assert MODE_CONFIG["F"]["program_length"] == 8
    assert MODE_CONFIG["F"]["space_size"] == 24**8 == 110_075_314_176
    total = MODE_CONFIG["C"]["space_size"] + MODE_CONFIG["F"]["space_size"]
    assert total > 10**12


def test_mode_c_alphabet_cannot_express_the_index_variable():
    assert 9 not in MODE_CONFIG["C"]["digit_to_token"]
    tokens = decode_ordinal(MODE_CONFIG["C"]["space_size"] - 1, "C")
    assert 9 not in tokens


# ---------------------------------------------------------------------------
# Ordinal codec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["C", "F"])
@pytest.mark.parametrize("fraction", [0.0, 1e-9, 0.25, 0.5, 0.75, 0.999999])
def test_codec_roundtrip(mode, fraction):
    ordinal = int(fraction * (MODE_CONFIG[mode]["space_size"] - 1))
    tokens = decode_ordinal(ordinal, mode)
    assert len(tokens) == MODE_CONFIG[mode]["program_length"]
    assert encode_program(tokens, mode) == ordinal


def test_codec_is_little_endian_over_the_mode_alphabet():
    digits = MODE_CONFIG["C"]["digit_to_token"]
    tokens = decode_ordinal(0, "C")
    assert tokens == tuple([digits[0]] * 9)
    tokens = decode_ordinal(1, "C")
    assert tokens[0] == digits[1]
    assert tokens[1:] == tuple([digits[0]] * 8)


def test_codec_rejects_out_of_range_and_foreign_tokens():
    with pytest.raises(CompositionalSearchError):
        decode_ordinal(MODE_CONFIG["C"]["space_size"], "C")
    with pytest.raises(CompositionalSearchError):
        decode_ordinal(-1, "C")
    with pytest.raises(CompositionalSearchError):
        encode_program((9,) * 9, "C")  # k is outside the mode C alphabet
    with pytest.raises(CompositionalSearchError):
        decode_ordinal(0, "Z")


# ---------------------------------------------------------------------------
# Validity pass
# ---------------------------------------------------------------------------


def test_validity_pass_names_the_failure():
    add = TOKEN_NAMES.index("add")
    one = TOKEN_NAMES.index("1")
    neg = TOKEN_NAMES.index("neg")
    assert program_status((one, one, add)) == "ok"
    assert program_status((one, add)) == "underflow"
    assert program_status((add,)) == "underflow"
    assert program_status((one, one)) == "residue"
    assert program_status(tuple([one] * (STACK_DEPTH + 1))) == "overflow"
    assert program_status((one, neg)) == "ok"
    assert is_valid_program((one, one, add))


def test_closed_form_valid_count_matches_brute_force(monkeypatch):
    monkeypatch.setitem(MODE_CONFIG["F"], "program_length", 3)
    monkeypatch.setitem(MODE_CONFIG["F"], "space_size", 24**3)
    counted = count_valid_programs("F")
    valid = 0
    with_variable = 0
    for ordinal in range(24**3):
        tokens = decode_ordinal(ordinal, "F")
        if program_status(tokens) == "ok":
            valid += 1
            if 9 in tokens:
                with_variable += 1
    assert counted["structurally_valid"] == valid
    assert counted["valid_with_variable"] == with_variable


def test_measured_valid_fraction_is_small_but_real():
    counted = count_valid_programs("C")
    fraction = counted["structurally_valid"] / counted["space_size"]
    assert 0.01 < fraction < 0.05
    assert counted["structurally_valid"] == 32_971_249_179


# ---------------------------------------------------------------------------
# Evaluation semantics
# ---------------------------------------------------------------------------


def test_rendering_round_trips_through_a_human_reading():
    tokens = _tokens_from_rpn("1/2 atan 1/3 atan add 4 mul 1 mul")
    assert render_rpn(tokens) == "1/2 atan 1/3 atan add 4 mul 1 mul"
    assert render_infix(tokens) == "(((atan(1/2) + atan(1/3)) * 4) * 1)"


@pytest.mark.parametrize(
    "rpn",
    [
        "1 1 sub recip",  # 1/0
        "1 neg sqrt",  # sqrt of a negative
        "1 neg ln",  # ln of a negative
        "1 1 1 sub div",  # division by zero
        "1 neg 1/2 pow",  # fractional power of a negative base
        "1 1 sub 1 neg pow",  # 0 to a negative power
    ],
)
def test_domain_violations_return_none(rpn):
    with mp.workdps(30):
        assert evaluate_mp(_tokens_from_rpn(rpn)) is None


def test_value_cap_rejects_an_overflowing_intermediate():
    with mp.workdps(30):
        assert evaluate_mp(_tokens_from_rpn("5 exp exp sqr")) is None
        assert evaluate_mp(_tokens_from_rpn("5 exp exp")) is not None


def test_trig_argument_bound_is_enforced_identically_everywhere():
    """Past the bound a double no longer determines sin to the match tolerance."""

    with mp.workdps(30):
        assert evaluate_mp(_tokens_from_rpn("5 exp sqr sqr sin")) is None
        assert evaluate_mp(_tokens_from_rpn("5 exp sin")) is not None
        assert evaluate_mp(_tokens_from_rpn("5 exp sqr sqr cos")) is None
    ordinals = np.array(
        [
            encode_program(_tokens_from_rpn("5 exp sqr sqr sin 1 mul 1 mul"), "C"),
            encode_program(_tokens_from_rpn("5 exp sin 1 mul 1 mul 1 mul"), "C"),
        ],
        dtype=np.int64,
    )
    _, usable = evaluate_batch_cpu(ordinals, "C")
    assert not usable[0]
    assert usable[1]


def test_evaluate_needs_a_variable_value_for_index_programs():
    tokens = _tokens_from_rpn("k sqr recip")
    with mp.workdps(30):
        assert evaluate_mp(tokens) is None
        assert evaluate_mp(tokens, 4) == mp.mpf(1) / 16


def test_sympy_translation_matches_the_numeric_evaluator():
    tokens = _tokens_from_rpn("1/2 atan 1/3 atan add 4 mul 1 mul")
    expression = to_sympy(tokens)
    with mp.workdps(40):
        assert abs(mp.mpf(str(expression.evalf(35))) - evaluate_mp(tokens)) < mp.mpf("1e-30")


# ---------------------------------------------------------------------------
# THE REDISCOVERY CONTROL
# ---------------------------------------------------------------------------


def test_grammar_rediscovers_classical_mathematics_in_its_own_space():
    """Machin-like arctan for pi, Basel and Apery series, phi via sqrt, Bernoulli's e."""

    rows = reachability_controls()
    assert len(rows) == len(BUILTIN_KNOWN_IDENTITIES) >= 11
    for row in rows:
        assert row["structural_status"] == "ok", row["id"]
        assert row["codec_roundtrip"], row["id"]
        assert row["program_length"] == MODE_CONFIG[row["mode"]]["program_length"], row["id"]
        assert row["reproduces_target"], row["id"]
    identifiers = {row["id"] for row in rows}
    assert {
        "pi_euler_machin_like",
        "zeta2_basel_series",
        "zeta3_apery_series",
        "phi_from_sqrt5",
        "e_bernoulli_limit",
    } <= identifiers


def test_basel_series_control_sums_to_zeta_two():
    tokens = _tokens_from_rpn("k sqr recip 2 sqr 4 div mul")
    with mp.workdps(40):
        total = evaluate_series_mp(tokens)
        assert total is not None
        assert abs(total - mp.pi**2 / 6) < mp.mpf("1e-35")


def test_divergent_series_is_refused_rather_than_accelerated():
    with mp.workdps(30):
        assert evaluate_series_mp(_tokens_from_rpn("k recip 1 mul 1 mul 1 mul")) is None
        assert evaluate_series_mp(_tokens_from_rpn("1 1 mul 1 mul 1 mul k 1 mul")) is None


# ---------------------------------------------------------------------------
# CPU reference machine
# ---------------------------------------------------------------------------


def test_cpu_batch_evaluator_matches_mpmath_on_the_controls():
    ordinals = np.array(
        [
            encode_program(_tokens_from_rpn(str(item["rpn"])), "C")
            for item in BUILTIN_KNOWN_IDENTITIES
            if item["mode"] == "C"
        ],
        dtype=np.int64,
    )
    values, usable = evaluate_batch_cpu(ordinals, "C")
    assert usable.all()
    with mp.workdps(30):
        for ordinal, value in zip(ordinals, values, strict=True):
            exact = evaluate_mp(decode_ordinal(int(ordinal), "C"))
            assert abs(float(exact) - value) < 1e-12 * max(1.0, abs(value))


def test_cpu_batch_evaluator_marks_invalid_programs_unusable():
    rng = np.random.default_rng(11)
    ordinals = rng.integers(0, MODE_CONFIG["C"]["space_size"], size=512, dtype=np.int64)
    _, usable = evaluate_batch_cpu(ordinals, "C")
    for ordinal, flag in zip(ordinals, usable, strict=True):
        if not is_valid_program(decode_ordinal(int(ordinal), "C")):
            assert not flag


def test_cpu_series_machine_recovers_the_basel_sum():
    ordinal = encode_program(_tokens_from_rpn("k sqr recip 2 sqr 4 div mul"), "F")
    series, _, flags = evaluate_series_batch_cpu(
        np.array([ordinal], dtype=np.int64), "F", 64, 4
    )
    assert flags[0, 0]
    assert abs(series[0] - math.pi**2 / 6) < 1e-5


# ---------------------------------------------------------------------------
# Deduplication measurement
# ---------------------------------------------------------------------------


def test_value_hash_collides_at_the_declared_precision_and_not_before():
    base = np.array([1.234567890123456], dtype=np.float64)
    near = base * (1 + 1e-14)
    far = base * (1 + 1e-9)
    assert value_hash_keys(base)[0] == value_hash_keys(near)[0]
    assert value_hash_keys(base)[0] != value_hash_keys(far)[0]


@pytest.mark.parametrize("truth", [1000, 250_000, 4_000_000])
def test_hyperloglog_estimates_a_known_cardinality(truth):
    rng = np.random.default_rng(7)
    hashes = rng.integers(0, 1 << 62, size=truth, dtype=np.int64).astype(np.uint64) * np.uint64(4)
    registers = hll_registers_from_hashes(hashes)
    estimate = hll_estimate(registers)
    assert abs(estimate - truth) / truth < 6 * 1.04 / math.sqrt(HLL_REGISTERS) + 0.01


# ---------------------------------------------------------------------------
# Decoys
# ---------------------------------------------------------------------------


def test_decoys_are_seeded_deterministic_and_paired():
    first = build_decoy_targets()
    second = build_decoy_targets()
    assert [item["value"] for item in first] == [item["value"] for item in second]
    paired = [item for item in first if item["kind"] == "paired"]
    assert len(paired) == 12
    with mp.workdps(50):
        for item in paired:
            real = float(target_value_mp(str(item["paired_with"])))
            offset = abs(item["value"] - real)
            assert 1e-3 <= offset <= 1e-2
    assert len([item for item in first if item["kind"] == "uniform"]) == 4


def test_target_slots_carry_both_roles():
    assert SLOT_COUNT == 28
    assert sum(1 for slot in TARGET_SLOTS if slot["role"] == "real") == 12
    assert sum(1 for slot in TARGET_SLOTS if slot["role"] == "decoy") == 16


def test_decoy_seed_is_declared_in_the_module():
    assert DECOY_SEED.startswith("invariant-compositional-expression-search-decoys")


# ---------------------------------------------------------------------------
# THE CHANCE-MATCH GATE
# ---------------------------------------------------------------------------


def _counts_with(profile: dict[int, int]) -> np.ndarray:
    """Cumulative window counts from a per-exponent specification."""

    counts = np.zeros((1, SLOT_COUNT, len(WINDOWS)), dtype=np.int64)
    for index, exponent in enumerate(WINDOW_EXPONENTS):
        counts[0, :, index] = profile.get(exponent, 0)
    return counts


def test_expected_chance_matches_follows_the_measured_density():
    # A linear-density population: C(W) = 1e6 * W.
    profile = {exponent: int(1e6 * 10.0**-exponent) for exponent in WINDOW_EXPONENTS}
    gate = chance_accounting(_counts_with(profile), ("constant",))
    row = gate["per_target"][0]
    density = float(row["local_density_per_unit"])
    assert 4e5 < density < 6e5
    expected = float(row["expected_chance_matches"])
    assert expected == pytest.approx(2.0 * MATCH_WINDOW * density, rel=1e-9)
    tolerance = float(row["tolerance_for_chance_below_threshold"])
    assert tolerance == pytest.approx(1e-6 / (2.0 * density), rel=1e-9)
    assert row["required_agreement_digits"] == max(1, math.ceil(-math.log10(tolerance)))
    assert row["count_scaling_exponent"]["supports_linear_extrapolation"]


def test_a_denser_neighbourhood_demands_more_digits():
    sparse = chance_accounting(
        _counts_with({exponent: int(1e6 * 10.0**-exponent) for exponent in WINDOW_EXPONENTS}),
        ("constant",),
    )
    dense = chance_accounting(
        _counts_with({exponent: int(1e12 * 10.0**-exponent) for exponent in WINDOW_EXPONENTS}),
        ("constant",),
    )
    assert (
        dense["per_target"][0]["required_agreement_digits"]
        > sparse["per_target"][0]["required_agreement_digits"]
    )
    assert dense["verification_dps"] >= sparse["verification_dps"]


def test_exact_match_spike_cannot_inflate_its_own_chance_model():
    """The annulus rule is the whole reason the gate is not self-defeating."""

    smooth = {exponent: int(1e6 * 10.0**-exponent) for exponent in WINDOW_EXPONENTS}
    spiked = dict(smooth)
    for exponent in WINDOW_EXPONENTS:
        spiked[exponent] = smooth[exponent] + 50_000  # an atom sitting exactly on the target
    plain = chance_accounting(_counts_with(smooth), ("constant",))["per_target"][0]
    spike = chance_accounting(_counts_with(spiked), ("constant",))["per_target"][0]
    assert spike["required_agreement_digits"] == plain["required_agreement_digits"]
    assert float(spike["expected_chance_matches"]) == pytest.approx(
        float(plain["expected_chance_matches"]), rel=1e-9
    )
    assert int(spike["fp64_matches_at_1e-12"]) > int(plain["fp64_matches_at_1e-12"])
    assert float(spike["excess_over_chance"]) > 49_000


def test_empty_neighbourhood_falls_back_to_a_conservative_upper_bound():
    gate = chance_accounting(_counts_with({}), ("constant",))
    row = gate["per_target"][0]
    assert row["density_source"]["count"] == 0
    assert "conservative" in row["density_source"]["rule"]
    assert row["required_agreement_digits"] >= 1


def test_verification_precision_is_derived_not_hardcoded():
    gate = chance_accounting(
        _counts_with(
            {exponent: min(2**62, int(1e26 * 10.0**-exponent)) for exponent in WINDOW_EXPONENTS}
        ),
        ("constant",),
    )
    assert "2 * eps * local_density" in gate["derivation"]
    assert gate["verification_dps"] == max(
        60, gate["max_required_agreement_digits"] + 30
    )
    assert gate["verification_dps"] > 60


# ---------------------------------------------------------------------------
# Retention derivation
# ---------------------------------------------------------------------------


def test_retention_rate_is_one_when_the_slot_fits_the_budget():
    profile = {exponent: 4 for exponent in WINDOW_EXPONENTS}
    parameters = derive_retention({"window_counts": _counts_with(profile)}, ("constant",), "C")
    assert parameters.window_core[0] == CORE_WINDOW
    assert parameters.window_band[0] == STRESS_WINDOW
    assert parameters.threshold_core[0] == (1 << 64) - 1
    assert parameters.threshold_band[0] == (1 << 64) - 1


def test_retention_rate_shrinks_when_a_slot_floods():
    profile = {exponent: 10**7 for exponent in WINDOW_EXPONENTS}
    parameters = derive_retention({"window_counts": _counts_with(profile)}, ("constant",), "C")
    rate = parameters.threshold_core[0] / float(1 << 64)
    assert 0 < rate < 1e-3


def test_rate_to_threshold_is_monotone():
    assert SweepParameters.rate_to_threshold(0.0) == 0
    assert SweepParameters.rate_to_threshold(1.0) == (1 << 64) - 1
    assert 0 < SweepParameters.rate_to_threshold(0.5) < (1 << 64) - 1


# ---------------------------------------------------------------------------
# Verification and the decoy gate
# ---------------------------------------------------------------------------


def _slot(name: str) -> dict:
    for slot in TARGET_SLOTS:
        if slot["name"] == name:
            return slot
    raise AssertionError(name)


def test_an_exact_identity_survives_the_gate():
    ordinal = encode_program(_tokens_from_rpn("1/2 atan 1/3 atan add 4 mul 1 mul"), "C")
    report = verify_candidate(ordinal, "C", "constant", _slot("pi"), 30, 80)
    assert report["survived_chance_gate"]
    assert len(report["verification_stages"]) == 2
    assert all(stage["status"] == "AGREES" for stage in report["verification_stages"])


def test_a_near_miss_dies_at_the_derived_precision():
    ordinal = encode_program(_tokens_from_rpn("1/2 atan 1/3 atan add 4 mul 1 mul"), "C")
    report = verify_candidate(ordinal, "C", "constant", _slot("e"), 30, 80)
    assert not report["survived_chance_gate"]
    assert report["verification_stages"][0]["status"] == "DIVERGES"


def test_asymptotic_near_identity_dies_on_the_holdout_bar():
    """An atan saturating at pi/2 beats any chance model; only the holdout bar sees it."""

    near = encode_program(_tokens_from_rpn("3/2 exp exp exp atan 1/2 neg div neg"), "C")
    report = verify_candidate(near, "C", "constant", _slot("pi"), 17, 60)
    assert not report["survived_chance_gate"]
    stage = report["verification_stages"][0]
    assert stage["agreement_digits"] >= 17  # it beats the chance bar comfortably
    assert stage["failed_bar"] == "HOLDOUT_BAR_AGREEMENT_DID_NOT_TRACK_PRECISION"

    true_identity = encode_program(_tokens_from_rpn("1/2 atan 1/3 atan add 4 mul 1 mul"), "C")
    good = verify_candidate(true_identity, "C", "constant", _slot("pi"), 17, 60)
    assert good["survived_chance_gate"]
    assert good["verification_stages"][0]["agreement_digits"] >= 45
    assert good["verification_stages"][1]["agreement_digits"] >= 105


def test_a_decoy_can_never_be_reproduced_by_an_expression():
    """The single most important test: a real with no closed form must not survive."""

    decoy = next(slot for slot in TARGET_SLOTS if slot["role"] == "decoy")
    for item in BUILTIN_KNOWN_IDENTITIES:
        if item["mode"] != "C":
            continue
        ordinal = encode_program(_tokens_from_rpn(str(item["rpn"])), "C")
        report = verify_candidate(ordinal, "C", "constant", decoy, 20, 80)
        assert not report["survived_chance_gate"], item["id"]


def test_verification_reports_the_agreement_digits_it_measured():
    ordinal = encode_program(_tokens_from_rpn("4 sqrt sqrt 1 mul 1 mul 1 mul"), "C")
    report = verify_candidate(ordinal, "C", "constant", _slot("sqrt2"), 40, 70)
    assert report["survived_chance_gate"]
    assert report["verification_stages"][0]["agreement_digits"] >= 40
    assert report["rpn"] == "4 sqrt sqrt 1 mul 1 mul 1 mul"
    assert "sqrt(sqrt(4))" in report["infix"]


# ---------------------------------------------------------------------------
# Downstream: prior art and classical reduction
# ---------------------------------------------------------------------------


def test_machin_like_pi_reduces_to_a_classical_family():
    tokens = _tokens_from_rpn("1/2 atan 1/3 atan add 4 mul 1 mul")
    reduction = classical_reduction(tokens, "constant", "pi")
    assert reduction["verdict"] == "KNOWN_BY_PROOF_FAMILY"
    assert reduction["technique_that_fired"] in {
        "symbolic_identity",
        "pslq_rational_linear_in_classical_basis",
    }


def test_basel_series_reduces_through_the_cf_proof_router():
    tokens = _tokens_from_rpn("k sqr recip 2 sqr 4 div mul")
    reduction = classical_reduction(tokens, "series", "zeta2")
    assert reduction["verdict"] == "KNOWN_BY_PROOF_FAMILY"
    assert reduction["technique_that_fired"] == (
        "hypergeometric_closed_form_via_cf_proof_router"
    )
    assert "term_ratio" in reduction["derivation"]


def test_a_non_identity_is_not_reduced_and_names_what_declined():
    tokens = _tokens_from_rpn("2 ln 3 ln mul 1 mul 1 mul")
    reduction = classical_reduction(tokens, "constant", "catalan")
    assert reduction["verdict"] == "NOT_REDUCED"
    assert reduction["techniques_attempted"]
    assert any(item.get("blocker") for item in reduction["techniques_attempted"])


def test_prior_art_value_hit_is_about_the_constant_not_the_expression():
    with mp.workdps(60):
        value = mp.nstr(mp.pi, 50)
    rows = [
        {
            "record_id": "r1",
            "family": "inverse_trigonometric",
            "value": value,
            "value_expr": "pi",
            "citation": {"author": "Euler"},
        }
    ]
    verdict = screen_prior_art(value, rows)
    assert verdict["verdict"] == "CONSTANT_ATTESTED_IN_CORPUS"
    assert verdict["matches"][0]["record_id"] == "r1"
    assert screen_prior_art(value, [])["verdict"] == "NOT_FOUND_IN_CORPUS"
    assert screen_prior_art(value, None)["verdict"] == "CORPUS_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Receipt sealing, claims, and tamper
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _null_reachability_block() -> str:
    """The C1 block a receipt reporting a null must now carry.  Built once; it is not cheap."""

    from sigma_theory_compiler.certified_null_search import campaign_reachability_block

    return json.dumps(campaign_reachability_block({"C": {}, "F": {}}))


def _minimal_receipt() -> dict:
    body = {
        "schema_version": RESULT_SCHEMA,
        "claims": dict(CLAIMS),
        "decision": "SEARCHED",
        "codec_validity": [{"mode": "C", "agrees_with_closed_form": True}],
        "controls": {
            "reachability": [{"id": "pi_euler_machin_like", "reproduces_target": True}],
            "determinism": [
                {"identical_accumulators": True, "identical_retained_ordinals": True}
            ],
            "cpu_gpu_crosscheck": [{"passes": True}],
        },
        "decoy_calibration": {
            "totals": {
                "real_fp64_matches": 10,
                "decoy_fp64_matches": 3,
                "real_post_gate_survivors": 2,
                "decoy_post_gate_survivors": 0,
            }
        },
        # A receipt reporting zero headline candidates is a negative result, and under C1 a
        # negative result without a reachability certificate is not publishable at all.
        "reachability": json.loads(_null_reachability_block()),
        "headline": {"count": 0, "entries": []},
    }
    core = canonical_sha256(body)
    body["result_core_sha256"] = core
    body["measurement"] = {"elapsed_seconds": "1.000"}
    return {**body, "content_sha256": canonical_sha256(body)}


def test_a_null_receipt_without_a_reachability_certificate_is_refused():
    """The C1 falsifier, at the receipt level: an uninformative null is not a result."""

    receipt = _minimal_receipt()
    del receipt["reachability"]
    _reseal(receipt)
    with pytest.raises(CompositionalSearchError, match="C1"):
        validate_receipt(receipt)


def _reseal(receipt: dict) -> dict:
    """Recompute both seals so a tamper test exercises the check it is aiming at."""

    body = {
        key: value
        for key, value in receipt.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    receipt["result_core_sha256"] = canonical_sha256(body)
    receipt["content_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    return receipt


def test_tamper_control_rejects_every_declared_probe():
    from sigma_theory_compiler.compositional_expression_search import tamper_control

    report = tamper_control()
    assert report["untampered_receipt_validates"]
    assert report["all_probes_rejected"]
    assert {item["probe"] for item in report["probes"]} == {
        "silent_edit_without_resealing",
        "flipped_claim_with_a_fresh_seal",
        "surviving_decoy_with_a_fresh_seal",
        "reduced_candidate_promoted_to_the_headline",
        "null_published_after_stripping_the_reachability_block",
        "unresolved_reachability_upgraded_to_a_real_negative",
    }


def test_minimal_receipt_validates():
    validate_receipt(_minimal_receipt())


def test_tampered_seal_is_rejected():
    receipt = _minimal_receipt()
    receipt["decision"] = "TAMPERED"
    with pytest.raises(CompositionalSearchError, match="seal"):
        validate_receipt(receipt)


def test_flipped_claim_is_rejected():
    receipt = _minimal_receipt()
    receipt["claims"]["corpus_absence_establishes_novelty"] = True
    _reseal(receipt)
    with pytest.raises(CompositionalSearchError, match="claim"):
        validate_receipt(receipt)


def test_surviving_decoy_invalidates_the_receipt():
    receipt = _minimal_receipt()
    receipt["decoy_calibration"]["totals"]["decoy_post_gate_survivors"] = 1
    _reseal(receipt)
    with pytest.raises(CompositionalSearchError, match="decoy"):
        validate_receipt(receipt)


def test_failed_rediscovery_control_invalidates_the_receipt():
    receipt = _minimal_receipt()
    receipt["controls"]["reachability"][0]["reproduces_target"] = False
    _reseal(receipt)
    with pytest.raises(CompositionalSearchError, match="rediscovery"):
        validate_receipt(receipt)


def test_reduced_candidate_cannot_sit_in_the_headline():
    receipt = _minimal_receipt()
    receipt["headline"] = {
        "count": 1,
        "entries": [
            {
                "survived_chance_gate": True,
                "classical_reduction": {"verdict": "KNOWN_BY_PROOF_FAMILY"},
            }
        ],
    }
    _reseal(receipt)
    with pytest.raises(CompositionalSearchError, match="reduced"):
        validate_receipt(receipt)


def test_core_binding_is_independent_of_the_timing_measurement():
    receipt = _minimal_receipt()
    receipt["measurement"]["elapsed_seconds"] = "2.000"
    receipt["content_sha256"] = canonical_sha256(
        {k: v for k, v in receipt.items() if k != "content_sha256"}
    )
    validate_receipt(receipt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(*arguments: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    source = str(REPO_ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{source}{os.pathsep}{existing}" if existing else source
    return subprocess.run(
        [sys.executable, "-m", "sigma_theory_compiler.compositional_expression_search", *arguments],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )


def test_cli_prints_the_declared_alphabet():
    result = _cli("--print-alphabet")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["token_count"] == 24
    assert payload["modes"]["C"]["space_size"] == 23**9
    assert payload["modes"]["F"]["space_size"] == 24**8


def test_cli_validates_a_receipt_file(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_minimal_receipt()), encoding="utf-8")
    result = _cli("--validate-checked", "--output", str(path))
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("mode", ["C", "F"])
def test_cuda_source_compiles(mode):
    import cupy as cp

    module = cp.RawModule(code=cuda_source(mode), options=("-std=c++14",), backend="nvrtc")
    name = "eval_mode_c" if mode == "C" else "eval_mode_f_stage1"
    assert module.get_function(name) is not None


@requires_gpu
def test_gpu_and_cpu_stack_machines_agree():
    from sigma_theory_compiler.compositional_expression_search import _Engine

    engine = _Engine("C")
    rng = np.random.default_rng(3)
    ordinals = np.sort(
        rng.integers(0, MODE_CONFIG["C"]["space_size"], size=20000, dtype=np.int64)
    )
    gpu_values, gpu_flags = engine.evaluate_list(ordinals)
    cpu_values, cpu_usable = evaluate_batch_cpu(ordinals, "C")
    assert (gpu_flags.astype(bool) == cpu_usable).all()
    both = gpu_flags.astype(bool)
    difference = np.abs(gpu_values[both] - cpu_values[both]) / np.maximum(
        1.0, np.abs(cpu_values[both])
    )
    assert difference.max() < 1e-12


@requires_gpu
def test_gpu_finds_every_rediscovery_control():
    from sigma_theory_compiler.compositional_expression_search import _Engine

    for mode in ("C", "F"):
        engine = _Engine(mode)
        items = [item for item in BUILTIN_KNOWN_IDENTITIES if item["mode"] == mode]
        ordinals = np.array(
            [encode_program(_tokens_from_rpn(str(item["rpn"])), mode) for item in items],
            dtype=np.int64,
        )
        values, flags = engine.evaluate_list(ordinals)
        series, limit, flags2 = (
            engine.stage2(ordinals) if mode == "F" else (None, None, None)
        )
        for index, item in enumerate(items):
            with mp.workdps(30):
                target = float(target_value_mp(str(item["target"])))
            scale = max(1.0, abs(target))
            if mode == "C":
                assert flags[index] & 1, item["id"]
                assert abs(float(values[index]) - target) / scale < MATCH_WINDOW, item["id"]
                continue
            bit = 0 if item["submode"] == "series" else 1
            # Stage 1 only has to reach the near-target set; stage 2 has to land the match.
            assert int(flags[index]) & (1 << bit), item["id"]
            stage1 = abs(float(values[index, bit]) - target) / scale
            assert stage1 < MODE_F_NEAR_WINDOW, (item["id"], stage1)
            assert int(flags2[index]) & (1 << bit), item["id"]
            stage2_value = float((series if bit == 0 else limit)[index])
            assert abs(stage2_value - target) / scale < MATCH_WINDOW, item["id"]


@requires_gpu
def test_gpu_structural_count_matches_the_closed_form_on_a_slice():
    from sigma_theory_compiler.compositional_expression_search import (
        SweepParameters,
        _Engine,
    )

    engine = _Engine("F")
    engine.reset()
    start = MODE_CONFIG["F"]["space_size"] - (1 << 20)
    engine.launch(start, 1 << 20, SweepParameters(engine.n_sub))
    status = engine.status_counts.get()
    ordinals = np.arange(start, start + (1 << 20), dtype=np.int64)
    expected = sum(
        1 for ordinal in ordinals if is_valid_program(decode_ordinal(int(ordinal), "F"))
    )
    assert int(status[0]) == (1 << 20)
    assert int(status[1]) == expected


@requires_gpu
def test_gpu_sweep_is_deterministic():
    from sigma_theory_compiler.compositional_expression_search import _Engine, determinism_control

    engine = _Engine("C")
    report = determinism_control(engine, MODE_CONFIG["C"]["space_size"] - (1 << 20), 1 << 20)
    assert report["identical_accumulators"]
    assert report["identical_retained_ordinals"]


# ---------------------------------------------------------------------------
# The campaign receipt, when one has been produced
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RECEIPT_PATH.exists(), reason="no campaign receipt in the tree")
def test_campaign_receipt_validates_and_reports_the_decoy_column():
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    validate_receipt(receipt)
    totals = receipt["decoy_calibration"]["totals"]
    assert totals["decoy_post_gate_survivors"] == 0
    assert receipt["campaign"]["enumerated_total"] >= 10**12
    assert receipt["campaign"]["exhaustive_over_declared_space"]["C"]
    assert receipt["campaign"]["exhaustive_over_declared_space"]["F"]
    assert receipt["claims"]["grammar_is_generative_not_curated"]
    assert receipt["claims"]["chance_match_gate_calibrated_by_decoys"]
    assert receipt["claims"]["unreduced_is_not_novel_it_is_unreviewed"]
    assert receipt["claims"]["enumerated_count_is_measured_not_extrapolated"]
    assert not receipt["claims"]["corpus_absence_establishes_novelty"]
