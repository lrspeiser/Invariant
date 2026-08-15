"""M7 counterexample-sweep gates.

A sweep is trivially "successful" if it can quietly skip the lanes it could not
decide, trust the screen layer's word for a witness, or claim a record it did not set.
These tests pin the three-bucket discipline (pass / fail / undecided-at-step-cap), the
exact-CPU witness verification, the literature-bound honesty rule, and the receipt
seal.  GPU-specific tests skip cleanly when no CUDA device is present.
"""

from __future__ import annotations

import json

import pytest

from sigma_theory_compiler.gpu_counterexample_sweep import (
    DECISION_COUNTEREXAMPLE,
    DECISION_NO_COUNTEREXAMPLE,
    DECISION_UNDECIDED,
    SYSTEM_CAPS,
    CounterexampleSweepError,
    main,
    sweep,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256


def _cupy_or_none():
    try:
        import cupy

        cupy.arange(4).sum()
        return cupy
    except Exception:  # noqa: BLE001 - any CUDA absence means skip
        return None


def _affine_divisibility(divisor: int) -> dict:
    return {
        "kind": "divisibility",
        "sequence": "affine",
        "sequence_params": {"a": 6, "b": 12},
        "divisor": divisor,
    }


COLLATZ_HALVING = {
    "kind": "index_scaling_relation",
    "sequence": "collatz_total_stopping_time",
    "scale": 2,
    "alpha": 1,
    "beta": 1,
}

_VOLATILE = {"elapsed_seconds", "throughput_per_second", "content_sha256"}


def _stable(receipt: dict, *, drop_device: bool = False) -> dict:
    dropped = _VOLATILE | ({"device"} if drop_device else set())
    return {key: value for key, value in receipt.items() if key not in dropped}


# ---------------------------------------------------------------------------
# Controls: true and false statements over declared ranges
# ---------------------------------------------------------------------------


def test_divisibility_holds_over_a_million():
    receipt = sweep(_affine_divisibility(6), 0, 10**6, use_gpu=False)
    assert receipt["decision"] == DECISION_NO_COUNTEREXAMPLE
    assert receipt["witness"] is None
    assert receipt["undecided"] == {"count": 0, "sample": []}
    assert receipt["counts"]["checked"] == 10**6
    assert receipt["counts"]["scanned_up_to"] == 10**6
    validate_receipt(receipt)


def test_divisibility_counterexample_reports_the_smallest_witness():
    receipt = sweep(_affine_divisibility(7), 0, 10**6, use_gpu=False)
    assert receipt["decision"] == DECISION_COUNTEREXAMPLE
    witness = receipt["witness"]
    assert witness["n"] == 0
    assert witness["exact_check"]["sequence_value"] == 12
    assert witness["exact_check"]["remainder"] == 12 % 7
    validate_receipt(receipt)


def test_collatz_halving_relation_has_no_counterexample_below_1e5():
    receipt = sweep(COLLATZ_HALVING, 1, 10**5, use_gpu=False)
    assert receipt["decision"] == DECISION_NO_COUNTEREXAMPLE
    assert receipt["undecided"]["count"] == 0
    assert receipt["counts"]["checked"] == 10**5 - 1
    assert receipt["statement"]["step_cap"] == SYSTEM_CAPS["collatz_step_cap_default"]
    validate_receipt(receipt)


def test_false_collatz_scaling_is_refuted_with_an_exact_witness():
    statement = {**COLLATZ_HALVING, "beta": 2}
    receipt = sweep(statement, 1, 4096, use_gpu=False)
    assert receipt["decision"] == DECISION_COUNTEREXAMPLE
    witness = receipt["witness"]
    assert witness["n"] == 1
    assert witness["exact_check"]["value_at_n"] == 0
    assert witness["exact_check"]["value_at_scaled_index"] == 1
    assert witness["exact_check"]["predicted"] == {"numerator": 2, "denominator": 1}
    validate_receipt(receipt)


def test_goldbach_below_the_literature_bound_is_a_mechanism_receipt():
    receipt = sweep({"kind": "goldbach_even_sum_of_two_primes"}, 4, 10**6, use_gpu=False)
    assert receipt["decision"] == DECISION_NO_COUNTEREXAMPLE
    assert receipt["counts"]["checked"] == (10**6 - 4) // 2
    assert receipt["claims"]["exceeds_literature_bound"] is False
    assert receipt["claims"]["mechanism_receipt_below_literature_bound"] is True
    assert receipt["claims"]["statement_has_declared_literature_bound"] is True
    assert receipt["literature"]["verified_below"] == 4 * 10**18
    validate_receipt(receipt)


def test_monotonicity_counterexample_for_the_digit_sum():
    statement = {"kind": "monotonicity", "sequence": "digit_sum_base10", "direction": "increasing"}
    receipt = sweep(statement, 0, 100, use_gpu=False)
    assert receipt["decision"] == DECISION_COUNTEREXAMPLE
    assert receipt["witness"]["n"] == 9
    assert receipt["witness"]["exact_check"] == {"value_at_n": 9, "value_at_next": 1}
    validate_receipt(receipt)


def test_monotonicity_holds_for_triangular_numbers():
    statement = {"kind": "monotonicity", "sequence": "triangular_number", "direction": "increasing"}
    receipt = sweep(statement, 0, 10**5, use_gpu=False)
    assert receipt["decision"] == DECISION_NO_COUNTEREXAMPLE
    validate_receipt(receipt)


def test_congruence_control_true_then_false():
    base = {"kind": "congruence", "sequence": "affine", "sequence_params": {"a": 6, "b": 12}}
    held = sweep({**base, "modulus": 3, "residue": 0}, 0, 10**4, use_gpu=False)
    assert held["decision"] == DECISION_NO_COUNTEREXAMPLE
    broken = sweep({**base, "modulus": 5, "residue": 2}, 0, 10**4, use_gpu=False)
    assert broken["decision"] == DECISION_COUNTEREXAMPLE
    assert broken["witness"]["n"] == 1
    assert broken["witness"]["exact_check"]["observed_residue"] == 3
    validate_receipt(broken)


def test_polynomial_positivity_true_and_false():
    held = sweep(
        {"kind": "polynomial_positivity", "coefficients": [1, 0, 1]}, -1000, 1000, use_gpu=False
    )
    assert held["decision"] == DECISION_NO_COUNTEREXAMPLE
    broken = sweep(
        {"kind": "polynomial_positivity", "coefficients": [-10, 1]}, 0, 100, use_gpu=False
    )
    assert broken["decision"] == DECISION_COUNTEREXAMPLE
    assert broken["witness"]["n"] == 0
    assert broken["witness"]["exact_check"]["value"] == -10
    validate_receipt(broken)


def test_int64_unsound_configurations_fall_back_to_exact_python():
    statement = {"kind": "polynomial_positivity", "coefficients": [1, 10**30]}
    receipt = sweep(statement, 0, 512, use_gpu=False)
    assert receipt["decision"] == DECISION_NO_COUNTEREXAMPLE
    assert receipt["arithmetic_path"] == "python-bigint"
    assert receipt["device"] == "cpu-python"
    validate_receipt(receipt)


# ---------------------------------------------------------------------------
# The fail-closed third bucket
# ---------------------------------------------------------------------------


def test_step_cap_hits_land_in_the_undecided_bucket_never_pass_or_fail():
    statement = {**COLLATZ_HALVING, "step_cap": 5}
    receipt = sweep(statement, 1, 100, use_gpu=False)
    assert receipt["decision"] == DECISION_UNDECIDED
    assert receipt["witness"] is None
    assert receipt["counts"]["screen_violation_lanes"] == 0
    assert receipt["undecided"]["count"] > 0
    assert 27 in receipt["undecided"]["sample"]  # sigma(27) = 111 >> 5
    assert 16 not in receipt["undecided"]["sample"]  # sigma(16) = 4, sigma(32) = 5: decided
    validate_receipt(receipt)


def test_undecided_sample_is_capped_at_100_with_a_full_count():
    statement = {**COLLATZ_HALVING, "step_cap": 1}
    receipt = sweep(statement, 1, 1000, chunk=64, use_gpu=False)
    assert receipt["decision"] == DECISION_UNDECIDED
    assert receipt["undecided"]["count"] > SYSTEM_CAPS["undecided_sample_cap"]
    sample = receipt["undecided"]["sample"]
    assert len(sample) == SYSTEM_CAPS["undecided_sample_cap"]
    assert sample == sorted(sample)
    assert receipt["chunks"] == (999 + 63) // 64
    validate_receipt(receipt)


# ---------------------------------------------------------------------------
# Receipt integrity
# ---------------------------------------------------------------------------


def test_receipt_is_deterministic_up_to_timing_and_tamper_fails_closed():
    first = sweep(_affine_divisibility(6), 0, 10**4, use_gpu=False)
    second = sweep(_affine_divisibility(6), 0, 10**4, use_gpu=False)
    assert _stable(first) == _stable(second)
    validate_receipt(first)
    body = {key: value for key, value in first.items() if key != "content_sha256"}
    body["decision"] = DECISION_COUNTEREXAMPLE
    with pytest.raises(CounterexampleSweepError):
        validate_receipt({**body, "content_sha256": canonical_sha256(body)})
    with pytest.raises(CounterexampleSweepError):
        validate_receipt({**first, "content_sha256": "0" * 64})


def test_witness_tamper_is_caught_by_exact_reverification():
    receipt = sweep(_affine_divisibility(7), 0, 10**6, use_gpu=False)
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    body["witness"] = {**body["witness"], "n": 5}  # a(5) = 42 is divisible by 7: not a witness
    with pytest.raises(CounterexampleSweepError):
        validate_receipt({**body, "content_sha256": canonical_sha256(body)})


def test_claims_and_scope_bind_the_honesty_rules():
    receipt = sweep(_affine_divisibility(6), 0, 1000, use_gpu=False)
    claims = receipt["claims"]
    assert claims["decision_is_proof_of_universal_statement"] is False
    assert claims["exceeds_literature_bound"] is False
    assert claims["statement_has_declared_literature_bound"] is False
    assert claims["scalar_truth_or_probability_score"] is False
    assert claims["step_cap_lanes_counted_as_pass_or_fail"] is False
    assert receipt["literature"] is None
    assert "mechanism receipt" in receipt["scope"]
    assert "UNDECIDED_STEP_CAP_HIT" in receipt["scope"]


# ---------------------------------------------------------------------------
# Fail-closed statement validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("statement", "lo", "hi"),
    [
        ({"kind": "nonsense"}, 0, 10),
        ({"kind": "divisibility", "sequence": "nope", "divisor": 3}, 0, 10),
        (_affine_divisibility(6), 5, 5),
        (_affine_divisibility(1), 0, 10),
        (COLLATZ_HALVING, 0, 10),
        ({"kind": "goldbach_even_sum_of_two_primes"}, 4, 10**9 + 2),
        ({**COLLATZ_HALVING, "alpha": 1.5}, 1, 10),
        ({**_affine_divisibility(6), "step_cap": 10}, 0, 10),
        ({**_affine_divisibility(6), "surprise": 1}, 0, 10),
        ({**_affine_divisibility(6), "text": "something else"}, 0, 10),
        ({"kind": "polynomial_positivity", "coefficients": []}, 0, 10),
        ({"kind": "monotonicity", "sequence": "affine", "sequence_params": {"a": 1, "b": 0}}, 0, 10),
    ],
)
def test_malformed_or_unsound_statements_are_refused(statement, lo, hi):
    with pytest.raises(CounterexampleSweepError):
        sweep(statement, lo, hi, use_gpu=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_a_sealed_receipt_and_validates_it(tmp_path, monkeypatch):
    statement_path = tmp_path / "statement.json"
    output_path = tmp_path / "receipt.json"
    statement_path.write_text(json.dumps(_affine_divisibility(6)), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "gpu-counterexample-sweep",
            "--statement",
            str(statement_path),
            "--lo",
            "0",
            "--hi",
            "20000",
            "--cpu",
            "--output",
            str(output_path),
        ],
    )
    assert main() == 0
    validate_receipt(json.loads(output_path.read_text(encoding="utf-8")))
    monkeypatch.setattr(
        "sys.argv",
        ["gpu-counterexample-sweep", "--validate-checked", "--output", str(output_path)],
    )
    assert main() == 0


# ---------------------------------------------------------------------------
# GPU/CPU agreement (skips without a CUDA device)
# ---------------------------------------------------------------------------


def test_gpu_and_cpu_decisions_agree_on_the_collatz_relation():
    if _cupy_or_none() is None:
        pytest.skip("no CUDA device")
    gpu = sweep(COLLATZ_HALVING, 1, 20000, use_gpu=True)
    cpu = sweep(COLLATZ_HALVING, 1, 20000, use_gpu=False)
    assert _stable(gpu, drop_device=True) == _stable(cpu, drop_device=True)
    validate_receipt(gpu)


def test_gpu_and_cpu_decisions_agree_on_goldbach():
    if _cupy_or_none() is None:
        pytest.skip("no CUDA device")
    statement = {"kind": "goldbach_even_sum_of_two_primes"}
    gpu = sweep(statement, 4, 10**5, use_gpu=True)
    cpu = sweep(statement, 4, 10**5, use_gpu=False)
    assert _stable(gpu, drop_device=True) == _stable(cpu, drop_device=True)
    assert gpu["decision"] == DECISION_NO_COUNTEREXAMPLE
