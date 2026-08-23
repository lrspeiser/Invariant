from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sigma_theory_compiler import external_creativity_lean_bridge as L
from sigma_theory_compiler.math_lean_adapter import ChildProcessResult
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_formula_specific_source_and_closed_premise_manifest() -> None:
    source = (ROOT / L.SOURCE_PATH).read_text(encoding="utf-8")
    assert "theorem recoveredKineticNormalForm" in source
    assert "theorem recoveredSumSquaresNormalForm" in source
    assert "theorem externalKnownFormulaControls" in source
    assert "sorry" not in source.lower()
    assert "axiom " not in source.lower()
    config = L.adapter_config(ROOT / "missing-lean")
    assert config.target == L.TARGET
    assert "Invariant.recoveredSumSquaresNormalForm" in config.allowed_premises
    assert "Classical.choice" in config.forbidden_premises
    for control in L.NEGATIVE_CONTROLS:
        negative = (ROOT / control["source_path"]).read_text(encoding="utf-8")
        assert control["target"].rsplit(".", 1)[-1] in negative
        assert "+ 1" in negative
        assert "rfl" in negative
        assert "sorry" not in negative.lower()
        assert "axiom " not in negative.lower()


def test_missing_lean_fails_closed_without_a_formal_claim() -> None:
    receipt = L.run_bridge(ROOT, executable=ROOT / "missing-lean", environment={})
    L.validate_receipt(receipt)
    assert receipt["status"] == "BLOCKED_LEAN_UNAVAILABLE_OR_REJECTED"
    assert not receipt["claims"]["known_formula_normal_forms_kernel_checked"]
    assert not receipt["claims"]["candidate_specific_unit_offset_mutations_kernel_rejected"]
    assert not receipt["claims"]["novel_formula_established"]
    assert not receipt["claims"]["physical_law_proved"]
    assert receipt["counts"]["wrong_formula_controls_rejected"] == 0
    assert all(
        item["outcome"] == "NOT_RUN_POSITIVE_CONTROL_BLOCKED"
        for item in receipt["wrong_formula_kernel_controls"]
    )


def _fake_lean_runner(command: tuple[str, ...], _: Path, __: float) -> ChildProcessResult:
    source = Path(command[-1]).name
    if source == Path(L.SOURCE_PATH).name:
        lines = [
            "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN",
            f"target={L.TARGET}",
            *(f"dependency={item}" for item in L.ALLOWED_PREMISES),
            "result=checked",
            "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_END",
        ]
        return ChildProcessResult(0, stdout=("\n".join(lines) + "\n").encode())
    return ChildProcessResult(1, stderr=b"error: tactic 'rfl' failed\n")


def _ci_environment() -> dict[str, str]:
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": L.CI_REPOSITORY,
        "GITHUB_WORKFLOW": L.CI_WORKFLOW,
        "INVARIANT_EVIDENCE_JOB": L.CI_JOB,
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_RUN_ATTEMPT": "1",
        "INVARIANT_EVIDENCE_HEAD_SHA": "a" * 40,
        "GITHUB_EVENT_NAME": "pull_request",
        "RUNNER_OS": "Linux",
        "RUNNER_ARCH": "X64",
    }


def _passing_receipt() -> dict[str, object]:
    return L.run_bridge(
        ROOT,
        executable=Path(sys.executable),
        environment={},
        ci_environment=_ci_environment(),
        runner=_fake_lean_runner,
    )


def test_two_candidate_specific_wrong_formulas_are_kernel_rejected() -> None:
    receipt = _passing_receipt()
    L.validate_receipt(receipt, ROOT, require_ci_provenance=True)
    assert receipt["status"] == "PASS"
    assert receipt["counts"] == {
        "kernel_executions_attempted": 3,
        "kernel_positive_passes": 1,
        "wrong_formula_controls_required": 2,
        "wrong_formula_controls_rejected": 2,
    }
    assert receipt["claims"]["candidate_specific_unit_offset_mutations_kernel_rejected"]
    assert receipt["ci_provenance"]["complete"] is True
    for item, expected in zip(
        receipt["wrong_formula_kernel_controls"], L.NEGATIVE_CONTROLS, strict=True
    ):
        assert item["benchmark_id"] == expected["benchmark_id"]
        assert item["candidate_id"] == expected["candidate_id"]
        assert item["mutation_operator"] == "add_exact_unit"
        assert item["expected_residual"] == "1/1"
        assert item["outcome"] == "REJECTED_BY_LEAN_KERNEL"
        assert item["rejection_receipt"]["execution"]["nonzero_exit_code"] is True


def test_resealed_mutation_substitution_is_rejected() -> None:
    receipt = _passing_receipt()
    changed = dict(receipt)
    changed["wrong_formula_kernel_controls"] = [
        dict(item) for item in receipt["wrong_formula_kernel_controls"]
    ]
    changed["wrong_formula_kernel_controls"][0]["candidate_id"] = "candidate.substituted"
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(ValueError, match="mutation binding"):
        L.validate_receipt(changed, ROOT, require_ci_provenance=True)
