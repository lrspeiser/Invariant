"""Exponent-Diophantine sweeper gates.

The sweeper is only as honest as its funnel: the fp64 screen must flag every true
solution (capture), the exact layer must kill every fp64 mirage (rejection), and the
receipt must refuse to seal or validate around a fabricated witness.  The load-bearing
tests are the controls — the gcd > 1 Beal family, the known Fermat-Catalan solutions,
the exactly verified Erdos-Straus witnesses — and the fail-closed negatives.  GPU
tests skip cleanly when no CUDA device is present.
"""

from __future__ import annotations

import json
from math import log
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.exponent_diophantine_sweeper import (
    BEAL_CONTROL,
    DECISION_KNOWNS_ONLY,
    DECISION_NO_COUNTEREXAMPLE,
    DECISION_NO_UNSOLVABLE,
    LABEL_CANDIDATE,
    LABEL_FAMILY,
    LABEL_KNOWN,
    SYSTEM_CAPS,
    ExponentDiophantineError,
    classify_beal_solution,
    es_complete_search,
    es_parametric_witness,
    es_symbolic_identity_checks,
    es_witness_is_exact,
    fc_exponent_condition,
    independent_beal_recheck,
    integer_root,
    known_fermat_catalan_table,
    main,
    run_beal_sweep,
    run_erdos_straus_sweep,
    run_fermat_catalan_sweep,
    screen_log_power,
    validate_receipt,
)
from sigma_theory_compiler.problem_queue import load_queue
from sigma_theory_compiler.sigma_core import canonical_sha256

QUEUE_V3_PATH = Path(__file__).resolve().parents[1] / "configs" / "problem_queue_v3.json"

_VOLATILE = {"elapsed_seconds", "throughput_per_second", "content_sha256"}


def _cupy_or_none():
    try:
        import cupy

        cupy.arange(4).sum()
        return cupy
    except Exception:  # noqa: BLE001 - any CUDA absence means skip
        return None


def _stable(receipt: dict, *, drop_device: bool = False) -> dict:
    dropped = _VOLATILE | ({"device"} if drop_device else set())
    return {key: value for key, value in receipt.items() if key not in dropped}


def _reseal(receipt: dict) -> dict:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    return {**body, "content_sha256": canonical_sha256(body)}


@pytest.fixture(scope="module")
def beal_small() -> dict:
    return run_beal_sweep(6, 5, use_gpu=False)


@pytest.fixture(scope="module")
def fc_small() -> dict:
    return run_fermat_catalan_sweep(20, 7, 12, use_gpu=False)


@pytest.fixture(scope="module")
def es_small() -> dict:
    return run_erdos_straus_sweep(1000, use_gpu=False)


# ---------------------------------------------------------------------------
# Exact helpers
# ---------------------------------------------------------------------------


def test_integer_root_is_exact_at_boundaries():
    for value, degree in [(0, 3), (1, 3), (7, 3), (8, 3), (9, 3), (10**30, 5), (2**52, 2)]:
        root = integer_root(value, degree)
        assert root**degree <= value < (root + 1) ** degree
    assert integer_root(3**5, 5) == 3
    assert integer_root(3**5 - 1, 5) == 2


# ---------------------------------------------------------------------------
# The fp64 -> exact funnel
# ---------------------------------------------------------------------------


def test_true_solution_always_passes_the_screen():
    """Capture: 3^3 + 6^3 = 3^5 must be flagged by the fp64 layer, with the
    resolution-proof-backed candidate base C0 = 3."""

    pairs = [(3, 3), (6, 3)]
    logs = np.array([e * log(b) for b, e in pairs])
    near, lanes = screen_log_power(np, logs, [5], chunk=8, near_cap=100)
    assert [0, 1, 5, 3] in near.tolist()
    assert lanes == 3


def test_synthetic_near_collision_is_flagged_then_exactly_rejected():
    """100000^3 + 1^3 = 10^15 + 1 sits ~3e-11 from a perfect cube in fp64 —
    inside the screen tolerance, but the exact layer must reject it (here at
    the congruence stage, which is exact integer arithmetic)."""

    from sigma_theory_compiler.exponent_diophantine_sweeper import _confirm_beal

    pairs = [(1, 3), (100000, 3)]
    logs = np.array([e * log(b) for b, e in pairs])
    near, _ = screen_log_power(np, logs, [3], chunk=8, near_cap=100)
    assert [0, 1, 3, 100000] in near.tolist()  # fp64 cannot see the +1
    solutions, congruence_rejected, rejected = _confirm_beal(pairs, near)
    assert solutions == [] and congruence_rejected >= 1 and rejected == 0
    total = 1**3 + 100000**3
    root = integer_root(total, 3)
    assert root**3 != total  # full big-integer arithmetic agrees: not a power


def test_near_hit_cap_fails_closed():
    pairs = [(2, 3), (2, 3)]
    logs = np.array([e * log(b) for b, e in pairs])
    with pytest.raises(ExponentDiophantineError):
        screen_log_power(np, logs, [4], chunk=8, near_cap=0)


def test_beal_receipt_records_a_balanced_funnel(beal_small):
    prefilter = beal_small["prefilter"]
    assert prefilter["near_hits"] == (
        prefilter["congruence_rejected"]
        + prefilter["exact_confirmed"]
        + prefilter["exact_rejected"]
    )
    assert prefilter["exact_confirmed"] == beal_small["results"]["solution_count"]
    assert prefilter["eps_rel"] == "1e-13"
    assert prefilter["delta_floor"] == "1e-9"


# ---------------------------------------------------------------------------
# Beal mode: the gcd > 1 control and the candidate discipline
# ---------------------------------------------------------------------------


def test_beal_small_box_rediscovers_common_factor_families(beal_small):
    assert beal_small["decision"] == DECISION_NO_COUNTEREXAMPLE
    results = beal_small["results"]
    keys = [(s["a"], s["x"], s["b"], s["y"], s["c"], s["z"]) for s in results["solutions_sample"]]
    assert (3, 3, 6, 3, 3, 5) in keys  # the named validation control
    assert (2, 3, 2, 3, 2, 4) in keys
    assert all(s["label"] == LABEL_FAMILY and s["gcd"] > 1 for s in results["solutions_sample"])
    assert results["control"] == {"found": True, "reachable": True, "witness": BEAL_CONTROL}
    assert results["counterexample_candidate_count"] == 0
    validate_receipt(beal_small)


def test_beal_claims_are_honest(beal_small):
    claims = beal_small["claims"]
    assert claims["box_decides_conjecture"] is False
    assert claims["exceeds_documented_search_landscape"] is False
    assert claims["mechanism_receipt"] is True
    assert claims["counterexample_candidate_present"] is False
    assert claims["known_common_factor_control_found"] is True
    assert claims["screen_trusted_without_exact_confirmation"] is False
    assert "Norvig" in beal_small["literature"]["citation"]


def test_coprime_classification_is_the_loud_path():
    label, common = classify_beal_solution(3, 3, 6, 3, 3, 5)
    assert (label, common) == (LABEL_FAMILY, 3)
    # A hypothetical coprime confirmation classifies as a candidate, never a family.
    label, common = classify_beal_solution(3, 3, 4, 3, 5, 3)
    assert (label, common) == (LABEL_CANDIDATE, 1)
    recheck = independent_beal_recheck(BEAL_CONTROL)
    assert recheck["equation_holds_by_iterated_multiplication"] is True
    assert recheck["gcd_by_euclid"] == recheck["gcd_by_math_gcd"] == 3


def test_fabricated_beal_counterexample_fails_closed(beal_small):
    """A resealed receipt carrying an invented coprime witness must die on the
    exact re-verification, not slip through as a discovery."""

    tampered = json.loads(json.dumps(beal_small))
    fake = {"a": 3, "x": 3, "b": 4, "y": 3, "c": 5, "z": 3, "gcd": 1, "label": LABEL_CANDIDATE}
    fake["independent_recheck"] = {
        "equation_holds_by_iterated_multiplication": True,
        "gcd_by_euclid": 1,
        "gcd_by_math_gcd": 1,
    }
    tampered["results"]["counterexample_candidates"] = [fake]
    tampered["results"]["counterexample_candidate_count"] = 1
    tampered["results"]["solution_count"] += 1
    tampered["prefilter"]["exact_confirmed"] += 1
    tampered["prefilter"]["near_hits"] += 1
    tampered["decision"] = "COUNTEREXAMPLE_CANDIDATE"
    tampered["claims"]["counterexample_candidate_present"] = True
    with pytest.raises(ExponentDiophantineError):
        validate_receipt(_reseal(tampered))


# ---------------------------------------------------------------------------
# Fermat-Catalan mode: known-solution rediscovery
# ---------------------------------------------------------------------------


def test_builtin_table_verifies_and_condition_is_exact():
    table = known_fermat_catalan_table()
    assert len(table) == 10
    assert fc_exponent_condition(2, 3, 7) and not fc_exponent_condition(2, 3, 6)
    assert not fc_exponent_condition(3, 3, 3)


def test_fc_small_box_rediscovers_the_reachable_knowns(fc_small):
    assert fc_small["decision"] == DECISION_KNOWNS_ONLY
    results = fc_small["results"]
    assert results["required_known_indices"] == [0, 1, 2, 3, 4]
    assert results["found_known_indices"] == [0, 1, 2, 3, 4]
    found = {
        (s["x"], s["p"], s["y"], s["q"], s["z"], s["r"]): s["label"]
        for s in results["solutions"]
    }
    for rediscovery in [
        (1, 7, 2, 3, 3, 2),
        (2, 5, 7, 2, 3, 4),
        (13, 2, 7, 3, 2, 9),
        (2, 7, 17, 3, 71, 2),
        (3, 5, 11, 4, 122, 2),
    ]:
        assert found[rediscovery] == LABEL_KNOWN
    assert results["new_to_table_count"] == 0
    validate_receipt(fc_small)


def test_fc_noncoprime_families_are_excluded_not_solutions(fc_small):
    prefilter = fc_small["prefilter"]
    assert prefilter["exact_confirmed_noncoprime_excluded"] > 0  # e.g. 2^3 + 2^3 = 2^4
    assert prefilter["near_hits"] == (
        prefilter["congruence_rejected"]
        + prefilter["exact_confirmed_coprime"]
        + prefilter["exact_confirmed_noncoprime_excluded"]
        + prefilter["exact_rejected"]
    )
    assert fc_small["claims"]["novelty_claimed_for_new_hits"] is False
    assert fc_small["claims"]["finiteness_decided"] is False


def test_fc_receipt_missing_a_required_known_fails_closed(fc_small):
    tampered = json.loads(json.dumps(fc_small))
    kept = [s for s in tampered["results"]["solutions"] if s["known_index"] != 2]
    tampered["results"]["solutions"] = kept
    tampered["results"]["solution_count"] = len(kept)
    tampered["results"]["known_rediscovered_count"] -= 1
    tampered["results"]["found_known_indices"] = [0, 1, 3, 4]
    tampered["prefilter"]["exact_confirmed_coprime"] -= 1
    tampered["prefilter"]["near_hits"] -= 1
    with pytest.raises(ExponentDiophantineError):
        validate_receipt(_reseal(tampered))


# ---------------------------------------------------------------------------
# Erdos-Straus mode: exact witnesses, symbolic identities, storage honesty
# ---------------------------------------------------------------------------


def test_es_symbolic_identities_all_hold():
    assert es_symbolic_identity_checks() == {
        "even": True,
        "mod4_3": True,
        "mod3_0": True,
        "mod3_2": True,
    }


def test_es_small_range_is_fully_witnessed(es_small):
    assert es_small["decision"] == DECISION_NO_UNSOLVABLE
    results = es_small["results"]
    assert results["coverage"] == {"class_total": 999, "expected_total": 999}
    assert results["unsolvable_candidates"] == []
    counts = {name: block["count"] for name, block in results["classes"].items()}
    assert counts == {"even": 500, "mod4_3": 250, "mod3_0": 83, "mod3_2": 83,
                      "hard_1_mod_12": 83}
    for name, block in results["classes"].items():
        for witness in block["sample"]:
            assert es_witness_is_exact(witness["n"], witness["x"], witness["y"], witness["z"])
            if name != "hard_1_mod_12":
                expected = es_parametric_witness(name, witness["n"])
                assert (witness["x"], witness["y"], witness["z"]) == expected
    assert es_small["claims"]["witnesses_exact_verified"] is True
    assert es_small["claims"]["exceeds_literature_bound"] is False
    assert es_small["claims"]["full_witness_table_stored"] is False
    validate_receipt(es_small)


def test_es_cpu_divisor_search_completes_starved_schedules():
    """With a deliberately starved GPU schedule the complete divisor search must
    pick up the leftovers and the receipt still closes the range."""

    receipt = run_erdos_straus_sweep(500, use_gpu=False, x_rounds=1, t_rounds=1)
    assert receipt["decision"] == DECISION_NO_UNSOLVABLE
    prefilter = receipt["prefilter"]
    assert prefilter["cpu_divisor_completed"] > 0
    assert (
        prefilter["gpu_resolved"] + prefilter["cpu_divisor_completed"]
        == prefilter["hard_class_count"]
    )
    assert es_complete_search(1201) is not None  # a classically stubborn n = 1 (mod 24)


def test_es_corrupted_witness_fails_closed(es_small):
    tampered = json.loads(json.dumps(es_small))
    tampered["results"]["classes"]["hard_1_mod_12"]["sample"][0]["z"] += 1
    with pytest.raises(ExponentDiophantineError):
        validate_receipt(_reseal(tampered))
    tampered = json.loads(json.dumps(es_small))
    tampered["results"]["classes"]["even"]["sample"][0]["x"] += 1
    with pytest.raises(ExponentDiophantineError):
        validate_receipt(_reseal(tampered))


# ---------------------------------------------------------------------------
# Receipt integrity: determinism, tamper, queue binding, box refusal
# ---------------------------------------------------------------------------


def test_receipts_are_deterministic_up_to_timing(beal_small, es_small):
    assert _stable(run_beal_sweep(6, 5, use_gpu=False)) == _stable(beal_small)
    assert _stable(run_erdos_straus_sweep(1000, use_gpu=False)) == _stable(es_small)


def test_tampered_decision_or_seal_fails_closed(beal_small):
    tampered = json.loads(json.dumps(beal_small))
    tampered["decision"] = "COUNTEREXAMPLE_CANDIDATE"
    with pytest.raises(ExponentDiophantineError):
        validate_receipt(_reseal(tampered))
    with pytest.raises(ExponentDiophantineError):
        validate_receipt({**beal_small, "content_sha256": "0" * 64})
    with pytest.raises(ExponentDiophantineError):
        validate_receipt({**beal_small, "extra": 1})


def test_receipts_bind_the_sealed_v3_queue(beal_small, fc_small, es_small):
    queue = load_queue(QUEUE_V3_PATH)
    for receipt in (beal_small, fc_small, es_small):
        assert receipt["queue_content_sha256"] == queue["content_sha256"]
    assert beal_small["problem_id"] == "beal_conjecture"
    assert fc_small["problem_id"] == "fermat_catalan"
    assert es_small["problem_id"] == "erdos_straus_sweeper_target"


def test_unsound_boxes_are_refused_a_priori():
    with pytest.raises(ExponentDiophantineError):
        run_beal_sweep(100000, 40, use_gpu=False)  # fp64 cannot resolve C_hat
    with pytest.raises(ExponentDiophantineError):
        run_fermat_catalan_sweep(100000, 40, 12, use_gpu=False)
    with pytest.raises(ExponentDiophantineError):
        run_erdos_straus_sweep(12, use_gpu=False)  # below the minimum declared range
    with pytest.raises(ExponentDiophantineError):
        run_beal_sweep(6, 5, use_gpu=False, queue_path="does/not/exist.json")


def test_cli_writes_validates_and_summarizes(tmp_path, capsys):
    output = tmp_path / "beal.json"
    assert main(
        ["--mode", "beal", "--base-max", "6", "--exp-max", "5", "--cpu",
         "--output", str(output)]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["decision"] == DECISION_NO_COUNTEREXAMPLE
    assert summary["prefilter"]["near_hits"] >= summary["prefilter"]["exact_confirmed"]
    assert main(["--validate-checked", "--output", str(output)]) == 0
    validate_receipt(json.loads(output.read_text(encoding="utf-8")))


def test_system_caps_are_declared_in_receipts(beal_small):
    assert beal_small["system_caps"] == SYSTEM_CAPS
    assert beal_small["arithmetic"] == {
        "confirm": "python-bigint-exact",
        "screen": "fp64-log-space",
    }


# ---------------------------------------------------------------------------
# The shipped run directory
# ---------------------------------------------------------------------------

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs" / "math" / "exponent-diophantine"


def test_shipped_receipts_exist_validate_and_bind_the_v3_queue():
    queue = load_queue(QUEUE_V3_PATH)
    expected = {
        "beal": DECISION_NO_COUNTEREXAMPLE,
        "fermat_catalan": DECISION_KNOWNS_ONLY,
        "erdos_straus": DECISION_NO_UNSOLVABLE,
    }
    for mode, decision in expected.items():
        receipt = json.loads((RUNS_DIR / f"{mode}.json").read_text(encoding="utf-8"))
        validate_receipt(receipt)
        assert receipt["mode"] == mode
        assert receipt["decision"] == decision
        assert receipt["queue_content_sha256"] == queue["content_sha256"]
        assert receipt["device"].startswith("NVIDIA")
    beal = json.loads((RUNS_DIR / "beal.json").read_text(encoding="utf-8"))
    assert beal["results"]["control"]["found"] is True
    assert beal["results"]["counterexample_candidate_count"] == 0
    fc = json.loads((RUNS_DIR / "fermat_catalan.json").read_text(encoding="utf-8"))
    assert fc["results"]["found_known_indices"] == [0, 1, 2, 3, 4]
    es = json.loads((RUNS_DIR / "erdos_straus.json").read_text(encoding="utf-8"))
    assert es["results"]["unsolvable_candidates"] == []
    assert es["box"]["n_max"] == 10**7


# ---------------------------------------------------------------------------
# GPU/CPU agreement (skips without a CUDA device)
# ---------------------------------------------------------------------------


def test_gpu_and_cpu_agree_on_beal_and_fc_solutions(beal_small, fc_small):
    if _cupy_or_none() is None:
        pytest.skip("no CUDA device")
    gpu_beal = run_beal_sweep(6, 5, use_gpu=True)
    assert gpu_beal["decision"] == beal_small["decision"]
    assert gpu_beal["results"]["solutions_sample"] == beal_small["results"]["solutions_sample"]
    gpu_fc = run_fermat_catalan_sweep(20, 7, 12, use_gpu=True)
    assert gpu_fc["results"]["solutions"] == fc_small["results"]["solutions"]
    validate_receipt(gpu_beal)
    validate_receipt(gpu_fc)


def test_gpu_and_cpu_agree_exactly_on_erdos_straus(es_small):
    """The ES screen is integer-exact, so GPU and CPU must agree byte for byte."""

    if _cupy_or_none() is None:
        pytest.skip("no CUDA device")
    gpu = run_erdos_straus_sweep(1000, use_gpu=True)
    assert _stable(gpu, drop_device=True) == _stable(es_small, drop_device=True)
    validate_receipt(gpu)
