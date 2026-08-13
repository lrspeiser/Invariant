from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.constraint_recovered_identity_breadth_lean_bridge import (
    ALLOWED_PREMISES,
    CLAIMS,
    CONFIG_PATH,
    OUTPUT_PATH,
    TARGET,
    THEOREM_PATH,
    WORLD_CONTRACTS,
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


def test_receipt_binds_two_exact_recovery_worlds(checked: dict) -> None:
    evidence = checked["recovery_evidence"]
    assert evidence["target_fields_read_before_unseal"] == 0
    assert len(evidence["worlds"]) == 2
    for world_id, contract in WORLD_CONTRACTS.items():
        world = next(item for item in evidence["worlds"] if item["world_id"] == world_id)
        for key, expected in contract.items():
            assert world[key] == expected
        assert world["candidate_count"] == 7
        assert world["candidate_families"] == [
            "bayesian",
            "cross_domain",
            "egraph",
            "evolutionary",
            "grammar",
            "llm",
            "symbolic",
        ]
        assert world["certificate_kind"] == "exact_rational_identity"
        assert world["certificate_decision"] == "proved_exact_rational_identity_on_regular_domain"


def test_closed_integer_polynomial_replays_are_exact_and_sealed(checked: dict) -> None:
    quartic, partial = checked["integer_polynomial_replays"]
    assert quartic["world_id"] == "constraint.hidden_quartic"
    assert quartic["factor_coefficients_constant_first"] == [[-2, 1], [3, 1], [5, 1, 1]]
    assert quartic["computed_coefficients_constant_first"] == [-30, -1, 0, 2, 1]
    assert (
        quartic["computed_coefficients_constant_first"]
        == quartic["recovered_coefficients_constant_first"]
    )
    assert partial["world_id"] == "constraint.hidden_partial_fraction"
    assert partial["partial_fraction_weights"] == [3, -2, 5]
    assert partial["computed_numerator_coefficients_constant_first"] == [127, 53, 6]
    assert partial["computed_denominator_coefficients_constant_first"] == [70, 59, 14, 1]
    assert partial["regular_domain_exclusions"] == [-7, -5, -2]
    for replay in (quartic, partial):
        assert replay["exact_equality"] is True
        assert replay["floating_point_operations"] == 0
        assert replay["content_sha256"] == _content_sha(replay)


def test_real_lean_quartic_proof_is_executable_coefficient_arithmetic(checked: dict) -> None:
    source = (ROOT / THEOREM_PATH).read_text(encoding="utf-8")
    assert "def recoveredPolyAdd" in source
    assert "def recoveredPolyScale" in source
    assert "def recoveredPolyMul" in source
    assert "recoveredPolyMul (recoveredPolyMul [-2, 1] [3, 1]) [5, 1, 1]" in source
    assert "[-30, -1, 0, 2, 1]" in source
    assert "rfl" in source
    assert f"target={TARGET}" in source
    assert "sorry" not in source.lower()
    assert "axiom " not in source.lower()
    assert checked["theorem_contract"]["sorry_or_axiom_used"] is False


def test_kernel_dependency_closure_and_false_control(checked: dict) -> None:
    adapter = checked["adapter_receipt"]
    assert adapter["decision"] == "pass_lean_checked_closed_premise"
    assert adapter["execution"]["exit_code"] == 0
    assert adapter["dependency_audit"]["closure_valid"] is True
    assert adapter["dependency_audit"]["dependencies"] == sorted(ALLOWED_PREMISES)
    assert checked["dependency_closure"]["recovery_targets_used_as_Lean_premises"] is False
    assert checked["dependency_closure"]["out_of_manifest_dependency_allowed"] is False
    false_control = checked["false_control"]
    assert false_control["alteration"] == "constant coefficient -30 changed to -29"
    assert false_control["adapter_status"] == "block"
    assert false_control["adapter_decision"] == "block_lean_process_failure"
    assert false_control["kernel_attempted"] is True
    assert false_control["nonzero_exit_code"] is True
    assert false_control["rejected_before_receipt_promotion"] is True


def test_counts_and_claims_are_bounded(checked: dict) -> None:
    assert checked["counts"] == {
        "recovered_worlds_bound": 2,
        "recovered_candidates_bound": 14,
        "symbolic_certificates_bound": 2,
        "integer_polynomial_replays": 2,
        "kernel_executions": 2,
        "kernel_checked_theorems": 1,
        "false_controls_rejected": 1,
        "blocked": 0,
        "rejected": 0,
    }
    assert checked["claims"] == CLAIMS
    assert CLAIMS["quartic_identity_kernel_checked"] is True
    assert CLAIMS["partial_fraction_integer_polynomial_identity_replayed"] is True
    assert CLAIMS["general_recovery_established"] is False
    assert CLAIMS["novelty_established"] is False
    assert CLAIMS["promotion_authorized"] is False


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
        lambda value: value["recovery_evidence"]["worlds"][0].__setitem__(
            "candidate_artifact_id", "sig-tampered"
        ),
        lambda value: value["recovery_evidence"].__setitem__("target_fields_read_before_unseal", 1),
        lambda value: value["integer_polynomial_replays"][0].__setitem__("exact_equality", False),
        lambda value: value["integer_polynomial_replays"][1][
            "computed_numerator_coefficients_constant_first"
        ].__setitem__(0, 128),
        lambda value: value["adapter_receipt"]["dependency_audit"].__setitem__(
            "dependencies", ["False.elim"]
        ),
        lambda value: value["false_control"].__setitem__("nonzero_exit_code", False),
        lambda value: value["theorem_contract"].__setitem__("sorry_or_axiom_used", True),
        lambda value: value["claims"].__setitem__("general_recovery_established", True),
        lambda value: value["toolchain_receipt"].__setitem__("version", "4.32.0"),
        lambda value: value["source_bindings"]["recovery"]["artifact"].__setitem__(
            "file_sha256", "2" * 64
        ),
        lambda value: value.__setitem__("unknown_top_level_key", True),
    ],
)
def test_resealed_semantic_tampers_fail_closed(checked: dict, mutator) -> None:
    tampered = copy.deepcopy(checked)
    mutator(tampered)
    for replay in tampered.get("integer_polynomial_replays", []):
        replay["content_sha256"] = _content_sha(replay)
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
