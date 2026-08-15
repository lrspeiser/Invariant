from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.constraint_recovered_recurrence_lean_kernel_bridge import (
    CLAIMS,
    CONFIG_PATH,
    OUTPUT_PATH,
    TARGET,
    THEOREM_PATH,
    WINNER_CONTRACT,
    _content_sha,
    build_live_receipt,
    validate_checked_receipt,
    validate_live_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str] | None:
    environment = dict(os.environ)
    if environment.get("INVARIANT_LEAN_EXECUTABLE") or shutil.which("lean"):
        return environment
    candidate = (
        Path.home()
        / ".cache"
        / "invariant"
        / "lean"
        / "v4.33.0"
        / "lean-4.33.0-windows"
        / "bin"
        / "lean.exe"
    )
    if not candidate.is_file():
        return None
    environment["INVARIANT_LEAN_EXECUTABLE"] = str(candidate)
    return environment


@pytest.fixture(scope="module")
def checked() -> dict:
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_checked_receipt(value, root=ROOT)
    return value


def test_receipt_binds_exact_recovery_candidate_and_certificate(checked: dict) -> None:
    evidence = checked["recovery_evidence"]
    for key, expected in WINNER_CONTRACT.items():
        assert evidence[key] == expected
    assert evidence["candidate_family"] == "llm"
    assert evidence["certificate_kind"] == "exact_first_order_induction"
    assert evidence["certificate_decision"] == "proved_by_base_and_symbolic_successor_identity"


def test_theorem_is_nontrivial_inductive_recurrence_proof(checked: dict) -> None:
    source = (ROOT / THEOREM_PATH).read_text(encoding="utf-8")
    assert "constraintRecoveredSequence n + (6 * n ^ 2 + 10 * n + 5)" in source
    assert "2 * n ^ 3 + 2 * n ^ 2 + n + 7" in source
    assert "induction n with" in source
    assert "recoveredPolynomialSuccessor" in source
    assert "target=Invariant.constraintRecoveredSequenceClosedForm" in source
    assert "sorry" not in source.lower()
    assert "axiom " not in source.lower()
    assert checked["theorem_contract"]["target"] == TARGET
    assert checked["theorem_contract"]["sorry_or_axiom_used"] is False


def test_real_kernel_dependency_closure_and_false_control(checked: dict) -> None:
    adapter = checked["adapter_receipt"]
    assert adapter["decision"] == "pass_lean_checked_closed_premise"
    assert adapter["execution"]["exit_code"] == 0
    assert adapter["dependency_audit"]["closure_valid"] is True
    assert checked["dependency_closure"]["out_of_manifest_dependency_allowed"] is False
    assert checked["dependency_closure"]["recovery_target_used_as_Lean_premise"] is False
    assert checked["false_control"] == {
        "adapter_decision": "block_lean_process_failure",
        "adapter_status": "block",
        "alteration": "base value 7 falsely claimed equal to 8",
        "kernel_attempted": True,
        "nonzero_exit_code": True,
        "rejected_before_receipt_promotion": True,
        "source_sha256": checked["false_control"]["source_sha256"],
        "target": "Invariant.constraintRecoveredSequenceFalseControl",
        "timed_out": False,
    }
    assert len(checked["false_control"]["source_sha256"]) == 64


def test_counts_and_claims_remain_bounded(checked: dict) -> None:
    assert checked["counts"] == {
        "blocked": 0,
        "false_controls_rejected": 1,
        "kernel_checked_theorems": 1,
        "kernel_executions": 2,
        "recovered_candidates_bound": 1,
        "rejected": 0,
        "symbolic_certificates_bound": 1,
    }
    assert checked["claims"] == CLAIMS
    assert CLAIMS["recurrence_closed_form_kernel_checked"] is True
    assert CLAIMS["general_recovery_established"] is False
    assert CLAIMS["novelty_established"] is False
    assert CLAIMS["promotion_authorized"] is False
    assert CLAIMS["scientific_or_physics_truth_inferred"] is False


def test_checked_receipt_is_host_path_free(checked: dict) -> None:
    encoded = json.dumps(checked, sort_keys=True)
    assert "C:\\Users\\" not in encoded
    assert "/home/" not in encoded
    assert checked["toolchain_receipt"]["executable_path_persisted"] is False


def test_optional_real_live_replay_is_exact() -> None:
    environment = _environment()
    if environment is None:
        pytest.skip("registered Lean unavailable; portable checked receipt remains valid")
    first = build_live_receipt(ROOT / CONFIG_PATH, environment=environment)
    second = build_live_receipt(ROOT / CONFIG_PATH, environment=environment)
    assert first == second
    validate_live_receipt(first, ROOT / CONFIG_PATH, environment=environment)
    assert first["receipt_role"] == "live_replay"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["recovery_evidence"].__setitem__(
            "candidate_artifact_id", "sig-tampered"
        ),
        lambda value: value["recovery_evidence"].__setitem__(
            "certificate_content_sha256", "0" * 64
        ),
        lambda value: value["recovery_evidence"].__setitem__("solver_result_sha256", "1" * 64),
        lambda value: value["adapter_receipt"]["dependency_audit"].__setitem__(
            "dependencies", ["False.elim"]
        ),
        lambda value: value["false_control"].__setitem__("nonzero_exit_code", False),
        lambda value: value["theorem_contract"].__setitem__("sorry_or_axiom_used", True),
        lambda value: value["claims"].__setitem__("general_recovery_established", True),
        lambda value: value["toolchain_receipt"].__setitem__("version", "4.32.0"),
        lambda value: value.__setitem__("unknown_top_level_key", True),
        lambda value: value["source_bindings"]["recovery"]["artifact"].__setitem__(
            "file_sha256", "2" * 64
        ),
    ],
)
def test_resealed_semantic_tampers_fail_closed(checked: dict, mutator) -> None:
    tampered = copy.deepcopy(checked)
    mutator(tampered)
    tampered["adapter_receipt"]["content_sha256"] = _content_sha(tampered["adapter_receipt"])
    tampered["content_sha256"] = _content_sha(tampered)
    with pytest.raises(ValueError):
        validate_checked_receipt(tampered, root=ROOT)


def test_resealed_absolute_path_tamper_fails_closed(checked: dict) -> None:
    tampered = copy.deepcopy(checked)
    tampered["scope"] = r"C:\Users\attacker\proof.lean"
    tampered["content_sha256"] = _content_sha(tampered)
    with pytest.raises(ValueError, match="host path"):
        validate_checked_receipt(tampered, root=ROOT)
