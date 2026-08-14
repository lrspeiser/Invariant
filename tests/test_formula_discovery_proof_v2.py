from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.formula_discovery_cli import EXIT_OPERATIONAL_ERROR, EXIT_PASS, main
from sigma_theory_compiler.formula_discovery_proof_v2 import (
    CASE_SPECS,
    FormulaDiscoveryProofV2Error,
    build_proof_v2_receipt,
    generate_proof_v2_source,
    load_proof_v2_receipt,
    validate_live_proof_v2_receipt,
    validate_proof_v2_receipt,
    write_proof_v2_receipt,
)


def _lean() -> Path:
    configured = os.environ.get("INVARIANT_LEAN_EXECUTABLE") or shutil.which("lean")
    if configured:
        return Path(configured)
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
    assert candidate.is_file(), "registered portable Lean 4.33 is required for Proof v2"
    return candidate


def _seal(value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    value["content_sha256"] = hashlib.sha256(payload).hexdigest()


@pytest.fixture(scope="module")
def receipt() -> dict:
    value = build_proof_v2_receipt(executable=_lean())
    validate_proof_v2_receipt(value)
    return value


def test_real_portable_lean_433_checks_three_strategies_and_rejects_controls(
    receipt: dict,
) -> None:
    assert receipt["decision"] == "PASS"
    assert receipt["toolchain_receipt"]["version"] == "4.33.0"
    assert receipt["counts"] == {
        "generated_theorems": 3,
        "distinct_proof_strategies": 3,
        "explicit_nonzero_denominator_premises": 1,
        "kernel_executions": 6,
        "kernel_passes": 3,
        "false_controls_rejected": 3,
        "blocked": 0,
    }
    assert len({case["strategy"] for case in receipt["cases"]}) == 3
    for case in receipt["cases"]:
        assert case["adapter_receipt"]["decision"] == "pass_lean_checked_closed_premise"
        rejected = case["false_control"]["adapter_receipt"]
        assert rejected["decision"] == "block_lean_process_failure"
        assert rejected["execution"]["nonzero_exit_code"] is True
        assert rejected["diagnostic_bytes_persisted"] is False


def test_rational_case_has_explicit_nonzero_denominator_premise(receipt: dict) -> None:
    case = receipt["cases"][0]
    assert "denominator_nonzero : x - 2 ≠ 0" in case["source"]
    assert "Rat.mul_div_cancel denominator_nonzero" in case["source"]
    assert case["allowed_premise_manifest"]["allowed_premises"] == ["Rat.mul_div_cancel"]


def test_second_order_recurrence_uses_two_prior_terms(receipt: dict) -> None:
    source = receipt["cases"][1]["source"]
    assert "| n + 2 =>" in source
    assert "(n + 1)" in source
    assert "formulaDiscoverySecondOrderSequenceV2 n" in source
    assert "rfl" in source


def test_quantified_non_identity_uses_distinct_order_strategy(receipt: dict) -> None:
    case = receipt["cases"][2]
    assert "(n : Nat)" in case["source"]
    assert "≠" in case["source"]
    assert "omega" in case["source"]
    assert case["strategy"] == "strict_order_contradiction_via_presburger_arithmetic"


def test_each_false_source_changes_exactly_one_token_and_no_other_byte_region() -> None:
    expected_changes = [("3", "4"), ("1", "2"), ("3", "2")]
    for spec, expected in zip(CASE_SPECS, expected_changes, strict=True):
        positive = generate_proof_v2_source(spec)[0].split()
        negative = generate_proof_v2_source(spec, false_control=True)[0].split()
        assert len(positive) == len(negative)
        assert [(left, right) for left, right in zip(positive, negative) if left != right] == [
            expected
        ]


def test_sources_have_no_unsafe_or_unregistered_proof_escape(receipt: dict) -> None:
    for case in receipt["cases"]:
        for source in (case["source"], case["false_control"]["source"]):
            lowered = source.lower()
            assert "sorry" not in lowered
            assert "axiom" not in lowered
            assert "admit" not in lowered
            assert "classical.choice" not in lowered
            assert "false.elim" not in lowered


def test_receipt_is_sealed_and_host_path_free(receipt: dict) -> None:
    encoded = json.dumps(receipt, sort_keys=True)
    assert "C:\\Users\\" not in encoded
    assert "/home/" not in encoded
    assert receipt["toolchain_receipt"]["executable_path_persisted"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["cases"][0].__setitem__("theorem_statement", "False"),
        lambda value: value["cases"][1]["false_control"].__setitem__("outcome", "PASS"),
        lambda value: value["claims"].__setitem__("general_formula_discovery_established", True),
        lambda value: value["toolchain_receipt"].__setitem__("version", "4.34.0"),
        lambda value: value.__setitem__("unknown", True),
    ],
)
def test_resealed_semantic_tampers_fail_closed(receipt: dict, mutator) -> None:
    tampered = copy.deepcopy(receipt)
    mutator(tampered)
    _seal(tampered)
    with pytest.raises(FormulaDiscoveryProofV2Error):
        validate_proof_v2_receipt(tampered)


def test_resealed_nested_adapter_tamper_fails_closed(receipt: dict) -> None:
    tampered = copy.deepcopy(receipt)
    adapter = tampered["cases"][2]["adapter_receipt"]
    adapter["dependency_audit"]["dependencies"] = ["False.elim"]
    _seal(adapter)
    _seal(tampered)
    with pytest.raises(FormulaDiscoveryProofV2Error):
        validate_proof_v2_receipt(tampered)


def test_resealed_host_path_tamper_fails_closed(receipt: dict) -> None:
    tampered = copy.deepcopy(receipt)
    tampered["scope"] = r"C:\Users\attacker\proof.lean"
    _seal(tampered)
    with pytest.raises(FormulaDiscoveryProofV2Error, match="host path"):
        validate_proof_v2_receipt(tampered)


def test_api_write_load_and_live_replay_are_exact(receipt: dict, tmp_path: Path) -> None:
    path = tmp_path / "proof-v2.json"
    assert write_proof_v2_receipt(receipt, path) == path
    assert load_proof_v2_receipt(path) == receipt
    with pytest.raises(FormulaDiscoveryProofV2Error, match="immutably"):
        write_proof_v2_receipt(receipt, path)
    validate_live_proof_v2_receipt(receipt, executable=_lean())


def test_public_cli_emits_pass_and_visible_reject_controls(tmp_path: Path, capsys) -> None:
    path = tmp_path / "proof-v2-cli.json"
    lean = _lean()
    assert main(["proof-v2-run", "--result", str(path), "--lean", str(lean)]) == EXIT_PASS
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["decision"] == "PASS"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert [case["false_control"]["outcome"] for case in stored["cases"]] == [
        "REJECT",
        "REJECT",
        "REJECT",
    ]
    assert main(["proof-v2-validate", "--result", str(path), "--lean", str(lean)]) == EXIT_PASS
    assert json.loads(capsys.readouterr().out)["decision"] == "PASS"
    assert (
        main(["proof-v2-run", "--result", str(path), "--lean", str(lean)]) == EXIT_OPERATIONAL_ERROR
    )
