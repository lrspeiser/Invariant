from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_matter_lensing_kinetic_gate_cone_straddling as cone

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / cone.CONFIG_PATH).read_text(encoding="utf-8"))


def test_config_and_admission_policy_are_hard_bound() -> None:
    config = cone.load_config(ROOT)
    gate = config["source_or_paper_gate"]
    assert gate["real_observational_source_required_before_observational_scoring"] is True
    assert gate["observational_scoring_authorized"] is False
    assert gate["missing_paper_action"] == "SOURCE_BLOCKED"
    assert gate["failed_analytic_benchmark_action"] == "BENCHMARK_FAILED"
    assert cone._sha256_file(ROOT / cone.POLICY_PATH) == gate["project_admission_policy_sha256"]


def test_symbolic_theorem_is_exact() -> None:
    checks = cone.symbolic_checks()
    assert all(checks.values())
    assert checks["matrix_map"]
    assert checks["det_k_minus_g"]
    assert checks["characteristic_at_zero"]
    assert checks["characteristic_at_one"]
    assert checks["positive_leading_coefficient"]
    assert checks["p0_independent_straddling_identity"]


def test_all_numeric_cases_are_healthy_and_straddle() -> None:
    records = cone.numeric_suite(_config())
    assert len(records) == 3
    for record in records:
        speeds = [float(value) for value in record["generalized_speed_squared"]]
        assert 0.0 < speeds[0] < 1.0 < speeds[1]
        assert min(float(value) for value in record["K_eigenvalues"]) > 0.0
        assert min(float(value) for value in record["G_eigenvalues"]) > 0.0
        assert record["passed"] is True


def test_linear_field_redefinition_preserves_generalized_speeds() -> None:
    case = _config()["numeric_cases"][0]
    x = float(case["X"])
    y = float(case["Y"])
    z, zx, zxx = cone._gate_derivatives(case)
    c = 1.0 + y * zx
    k = np.array(
        [
            [c + 2.0 * x * y * zxx, 2.0 * zx * np.sqrt(x * y)],
            [2.0 * zx * np.sqrt(x * y), z],
        ]
    )
    g = np.diag([c, z])
    transform = np.array([[0.9, -0.4], [0.25, 1.2]])
    original = cone._generalized_speeds(k, g)
    changed = cone._generalized_speeds(transform.T @ k @ transform, transform.T @ g @ transform)
    assert np.allclose(original, changed, rtol=2.0e-12, atol=2.0e-12)


def test_escape_cases_are_boundaries_not_active_counterexamples() -> None:
    checks = cone.symbolic_checks()
    assert checks["escape_x_zero"]
    assert checks["escape_y_zero"]
    assert checks["escape_zx_zero"]
    assert _config()["theorem"]["escape_cases"][:3] == ["X=0", "Y=0", "Z_X=0"]


def test_predecessor_commit_and_worktree_bytes_match() -> None:
    config = cone.load_config(ROOT)
    binding = cone._validate_predecessor(ROOT, config)
    assert binding["commit"] == "35868a7e37fe371f2b02d076c8ff3c554ae0c79b"
    assert binding["receipt_content_sha256"] == (
        "69a5c135e00c2b636c1a189968728aec186bfcb0cfc1c4dedd7623bb8705ac5a"
    )


def test_receipt_is_deterministic_and_strictly_scoped() -> None:
    first = cone.build_receipt(ROOT)
    second = cone.build_receipt(ROOT)
    assert first == second
    assert first["checks_passed"] == first["checks_total"] == 13
    assert first["decision"] == cone.DECISION
    assert first["claim_boundary"]["exact_metric_cone_straddling_theorem_established"] is True
    assert first["claim_boundary"]["unconditional_causality_violation_established"] is False
    assert first["claim_boundary"]["historical_novelty_established"] is False
    assert first["claim_boundary"]["publication_ready"] is False
    assert all(value == 0 for value in first["access_ledger"].values())
    assert first["content_sha256"] == cone._self_hash(first)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("source_or_paper_gate", "observational_scoring_authorized", True),
        ("source_or_paper_gate", "missing_paper_action", "SCORE_ANYWAY"),
        ("theorem", "exact_identity", "det(K-G)>0"),
        ("claim_boundary", "unconditional_causality_violation_established", True),
        ("claim_boundary", "historical_novelty_established", True),
        ("access_ledger", "observational_rows_read", 1),
    ],
)
def test_semantic_mutations_fail_closed(section: str, key: str, value: object) -> None:
    config = copy.deepcopy(_config())
    config[section][key] = value
    with pytest.raises(cone.KineticGateConeStraddlingError):
        cone.validate_config(config)


def test_paper_inventory_cannot_claim_historical_search_completion() -> None:
    config = copy.deepcopy(_config())
    config["paper_anchors"][0]["exact_straddling_theorem_found"] = True
    with pytest.raises(cone.KineticGateConeStraddlingError, match="paper inventory"):
        cone.validate_config(config)


def test_receipt_claim_forgery_fails_even_when_rehashed() -> None:
    config = cone.load_config(ROOT)
    receipt = cone.build_receipt(ROOT)
    receipt["claim_boundary"]["healthy_modified_gravity_model_established"] = True
    receipt["content_sha256"] = cone._self_hash(receipt)
    with pytest.raises(cone.KineticGateConeStraddlingError, match="claims changed"):
        cone.validate_receipt(receipt, config)


def test_atomic_receipt_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = {"content_sha256": "example"}
    assert cone._atomic_no_clobber(path, value) == "CREATED"
    assert cone._atomic_no_clobber(path, value) == "EXISTING_IDENTICAL"
    with pytest.raises(cone.KineticGateConeStraddlingError, match="refusing to replace"):
        cone._atomic_no_clobber(path, {"content_sha256": "different"})
