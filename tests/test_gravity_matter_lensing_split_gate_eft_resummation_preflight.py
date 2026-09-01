from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_split_gate_eft_resummation_preflight as eft,
)


def test_config_identity_and_paper_gate() -> None:
    config = eft.load_config()
    assert config["artifact_id"] == eft.ARTIFACT_ID
    assert config["admission_policy"]["source_class"] == (
        "PRIMARY_PAPERS_PLUS_ENGINEERING_DIMENSION_AND_LIMIT_BENCHMARKS"
    )
    assert config["admission_policy"]["observational_data_required"] is False


def test_publication_candidate_binding_is_exact() -> None:
    receipt = eft.build_receipt()
    binding = receipt["publication_candidate_binding"]
    assert binding["config_sha256"] == (
        "730fdf89ca22827b310cb3481e435ced222ac71dc246d7d0f6254534c7bdb4a4"
    )
    assert binding["receipt_content_sha256"] == (
        "1b1f98f8cb047377e452c15e9865315bf357c56f5dd9a2401209d1da0e1799db"
    )


def test_engineering_dimension_and_scale_checks_pass() -> None:
    checks = eft._symbolic_checks()
    assert len(checks) == 7
    assert all(checks.values())


def test_high_u_means_large_background_gradient_relative_to_gate_scale() -> None:
    config = eft.load_config()
    contract = config["engineering_dimension_contract"]
    assert "u>>1 implies X^(1/4)>>Lambda_g" in contract["high_gate_implication"]
    assert "do not by themselves prove EFT failure" in contract["interpretation_boundary"]


def test_p2_gate_is_finite_polynomial_with_two_higher_operators() -> None:
    contract = eft.load_config()["p2_operator_contract"]
    assert contract["gate"] == ("Z=(1+u)^2=1+2*X^2/Lambda_g^8+X^4/Lambda_g^16")
    assert contract["leading_operator_dimension"] == 10
    assert contract["next_operator_dimension"] == 18
    assert contract["leading_coefficient_dimension"] == -6
    assert contract["next_coefficient_dimension"] == -14


def test_interaction_scales_are_not_confused_with_background_gate_scale() -> None:
    grid = eft._numeric_grid(eft.load_config())
    assert len(grid) == 3
    assert grid[0]["Lambda_6_over_Lambda_g"] == 1.0
    assert grid[0]["Lambda_14_over_Lambda_g"] == 1.0
    assert grid[-1]["Lambda_6_over_Lambda_g"] > grid[-1]["Lambda_14_over_Lambda_g"] > 1.0
    assert (
        "not the same statement"
        in eft.load_config()["p2_operator_contract"]["important_distinction"]
    )


def test_noninteger_gate_requires_full_resummed_function_at_high_u() -> None:
    boundary = eft.load_config()["general_gate_boundary"]
    assert "radius |u|<1" in boundary["noninteger_p"]
    assert "full nonpolynomial function" in boundary["noninteger_p"]


def test_primary_sources_are_exact() -> None:
    papers = eft.load_config()["primary_literature"]
    assert {item["arxiv"] for item in papers} == {
        "2107.00010",
        "2105.13992",
        "1611.08279",
        "2604.20292v2",
    }


def test_result_is_requirement_not_elimination() -> None:
    adjudication = eft.load_config()["adjudication"]
    assert adjudication["high_u_outside_naive_small_u_expansion"] is True
    assert adjudication["automatic_EFT_inconsistency_proved"] is False
    assert adjudication["range_source_theorem_remains_mathematically_valid"] is True
    assert adjudication["physical_model_promotion_allowed"] is False


def test_claim_ceiling_retains_unproven_resummation_and_symmetry() -> None:
    claims = eft.load_config()["claim_boundary"]
    assert claims["naive_expansion_tension_found"] is True
    assert claims["fatal_no_go"] is False
    assert claims["controlled_resummation"] is False
    assert claims["protecting_symmetry"] is False
    assert claims["radiative_stability"] is False
    assert claims["publication_ready"] is False


def test_receipt_is_deterministic_and_self_hashed() -> None:
    first = eft.build_receipt()
    second = eft.build_receipt()
    assert first == second
    assert first["content_sha256"] == eft._self_hash(first)
    assert first["checks_passed"] == 12
    assert all(first["checks"].values())


def test_coherent_fatal_no_go_forgery_differs_from_rebuild() -> None:
    forged = copy.deepcopy(eft.build_receipt())
    forged["claim_boundary"]["fatal_no_go"] = True
    forged["content_sha256"] = eft._self_hash(forged)
    assert forged["content_sha256"] == eft._self_hash(forged)
    assert forged != eft.build_receipt()


def test_config_mutation_rejects_before_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = eft._repo_root()
    path = tmp_path / eft.CONFIG_PATH
    path.parent.mkdir(parents=True)
    config = json.loads((root / eft.CONFIG_PATH).read_text(encoding="utf-8"))
    config["claim_boundary"]["fatal_no_go"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(eft, "EXPECTED_CONFIG_RAW_SHA256", "0" * 64)
    with pytest.raises(eft.SplitGateEFTPreflightError, match="config semantics changed"):
        eft.load_config(tmp_path)


def test_zero_observational_access() -> None:
    assert not any(eft.build_receipt()["access_ledger"].values())


def test_write_replay_and_check_are_no_clobber() -> None:
    assert eft.write_receipt() == "EXISTING_IDENTICAL"
    assert eft.validate_receipt() == eft.build_receipt()
