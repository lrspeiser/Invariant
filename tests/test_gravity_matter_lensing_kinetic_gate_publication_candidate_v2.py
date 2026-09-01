from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_kinetic_gate_publication_candidate_v2 as candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / candidate.CONFIG_PATH).read_text(encoding="utf-8"))


def test_independent_symbolic_checks_prove_both_theorems() -> None:
    checks = candidate.symbolic_checks()
    assert all(checks.values())
    assert checks["cone_determinant"]
    assert checks["cone_polynomial_at_zero"]
    assert checks["cone_polynomial_at_one"]
    assert checks["cone_positive_leading_coefficient"]
    assert checks["dynamic_range_log_slope_identity"]
    assert checks["riccati_equality_solution"]
    assert checks["riccati_finite_pole"]


def test_builder_solver_admission_policy_is_bound() -> None:
    config = candidate.load_config(ROOT)
    policy = config["admission_policy"]
    assert candidate._sha256_file(ROOT / policy["path"]) == policy["sha256"]
    assert "REAL_PUBLIC_SOURCE_DATASET" in policy["future_observational_gate"]
    assert "PRIMARY_DATASET_PAPER" in policy["future_observational_gate"]
    assert "INDEPENDENT_SOLVER_BENCHMARK" in policy["future_observational_gate"]


def test_bound_artifacts_and_v1_commit_are_exact() -> None:
    config = candidate.load_config(ROOT)
    receipts = candidate._validate_bindings(ROOT, config)
    assert set(receipts) == {
        "PUBLICATION_CANDIDATE_V1",
        "NOVELTY_BENCHMARK_V1",
        "CONE_STRADDLING_V1",
    }
    v1 = config["bindings"][0]
    assert v1["state"] == "COMMITTED"
    for role in ("config", "module", "test", "receipt"):
        relative = v1[f"{role}_path"]
        assert (
            candidate._sha256_bytes(candidate._git_show(ROOT, v1["commit"], relative))
            == v1[f"{role}_sha256"]
        )


def test_draft_contains_both_proofs_and_caveats() -> None:
    checks = candidate._validate_draft(ROOT)
    assert all(checks.values())
    text = (ROOT / candidate.DRAFT_PATH).read_text(encoding="utf-8")
    assert "not a claim of a successful theory of gravity" in text
    assert "does not establish historical novelty" in text
    assert "real public source dataset" in text


def test_receipt_reproduces_witness_as_cone_theorem_case() -> None:
    receipt = candidate.build_receipt(ROOT)
    assert receipt["bounded_witness_match"]["matched"] is True
    witness = float(receipt["bounded_witness_match"]["v1_max_speed_squared"])
    theorem = float(receipt["bounded_witness_match"]["cone_theorem_case_max_speed_squared"])
    assert witness == pytest.approx(theorem, rel=2.0e-15, abs=2.0e-15)
    assert theorem > 1.0


def test_publication_adjudication_is_positive_but_not_overclaimed() -> None:
    receipt = candidate.build_receipt(ROOT)
    adjudication = receipt["publication_adjudication"]
    assert adjudication["scientifically_interesting"] is True
    assert adjudication["worth_preparing_as_narrow_theory_note"] is True
    assert adjudication["worth_claiming_as_successful_gravity_model"] is False
    claims = receipt["claim_boundary"]
    assert claims["exact_two_theorem_pair_established"] is True
    assert claims["worth_independent_expert_review"] is True
    assert claims["historical_novelty_established"] is False
    assert claims["unconditional_causality_violation"] is False
    assert claims["publication_ready"] is False


def test_receipt_is_deterministic_and_complete() -> None:
    first = candidate.build_receipt(ROOT)
    second = candidate.build_receipt(ROOT)
    assert first == second
    assert first["decision"] == candidate.DECISION
    assert first["checks_passed"] == first["checks_total"] == 14
    assert all(first["checks"].values())
    assert first["content_sha256"] == candidate._self_hash(first)
    assert all(value == 0 for value in first["access_ledger"].values())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("novelty_protocol", "historical_novelty_established", True),
        ("publication_adjudication", "scientifically_interesting", False),
        ("publication_adjudication", "worth_claiming_as_successful_gravity_model", True),
        ("claim_boundary", "unconditional_causality_violation", True),
        ("claim_boundary", "publication_ready", True),
        ("access_ledger", "observational_rows_read", 1),
    ],
)
def test_config_overclaim_and_access_mutations_fail_closed(
    section: str, key: str, value: object
) -> None:
    config = copy.deepcopy(_config())
    config[section][key] = value
    with pytest.raises(candidate.KineticGatePublicationCandidateV2Error):
        candidate.validate_config(config)


def test_rehashed_receipt_overclaim_fails_closed() -> None:
    config = candidate.load_config(ROOT)
    receipt = candidate.build_receipt(ROOT)
    receipt["claim_boundary"]["publication_ready"] = True
    receipt["content_sha256"] = candidate._self_hash(receipt)
    with pytest.raises(candidate.KineticGatePublicationCandidateV2Error, match="claims changed"):
        candidate.validate_receipt(receipt, config)


def test_atomic_write_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = {"content_sha256": "one"}
    assert candidate._atomic_no_clobber(path, value) == "CREATED"
    assert candidate._atomic_no_clobber(path, value) == "EXISTING_IDENTICAL"
    with pytest.raises(
        candidate.KineticGatePublicationCandidateV2Error, match="refusing to replace"
    ):
        candidate._atomic_no_clobber(path, {"content_sha256": "two"})
