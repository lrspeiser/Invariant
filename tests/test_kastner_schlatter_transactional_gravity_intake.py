from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_transactional_gravity_intake import (
    FIRST_BLOCKER,
    SOURCE_PDF_SHA256,
    _validate_result,
    build_intake,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_transactional_gravity_intake.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-transactional-gravity-intake.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_intake(_load(CONFIG), ROOT)


def test_exact_rebuild_and_primary_source_binding(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["primary_source"]["arxiv_id"] == "2209.04025"
    assert artifact["primary_source"]["version"] == "v1"
    assert artifact["source_binding"]["official_pdf_sha256"] == SOURCE_PDF_SHA256
    assert artifact["synthetic_preflight_counts"] == {"pass": 7, "reject": 0, "block": 1}


def test_equation_contracts_are_exactly_scoped(rebuilt: dict[str, object]) -> None:
    contracts = {item["contract_id"]: item for item in rebuilt["formula_contracts"]}
    assert len(contracts) == 8
    assert contracts["transaction_poisson_rate_and_pressure"]["paper_equations"] == [33, 34]
    assert contracts["standard_poisson_cuda_reference"]["paper_equations"] == []
    assert (
        contracts["standard_poisson_cuda_reference"]["classification"]
        == "standard_implementation_of_paper_poisson_assertion_not_printed_equation"
    )
    assert contracts["transaction_cosmological_term"]["paper_equations"] == [35, 36]
    assert contracts["einstein_trace_reversed_recovery_scope"]["paper_equations"] == [38, 39]
    assert contracts["schwarzschild_de_sitter_background"]["paper_equations"] == [42, 44, 45]
    assert contracts["sds_effective_entropy_quadratic"]["paper_equations"] == [55, 59, 60]
    assert contracts["sds_mond_galaxy_relation"]["paper_equations"] == [62, 65, 68, 69]


def test_equation_35_normalization_ambiguity_blocks_not_rejects(
    rebuilt: dict[str, object],
) -> None:
    check = next(
        item
        for item in rebuilt["synthetic_checks"]
        if item["check_id"] == "transaction_pressure_lambda_identity"
    )
    assert check["status"] == "block"
    assert check["pressure_to_middle_exact_residual"] == "0"
    assert check["printed_chain_exact_residual"] == "4*pi**2*l_P**2*q"
    assert (
        check["first_missing_premise"]
        == "equation_35_h_vs_hbar_factor_normalization_clarification"
    )
    assert rebuilt["decision"] == "blocked"
    assert rebuilt["first_blocker"] == FIRST_BLOCKER


def test_synthetic_positive_and_negative_controls(rebuilt: dict[str, object]) -> None:
    checks = {item["check_id"]: item for item in rebuilt["synthetic_checks"]}
    assert checks["poisson_float64_recurrence"]["status"] == "pass"
    assert checks["poisson_symbolic_mean_variance"]["status"] == "pass"
    assert checks["einstein_trace_reversal_four_dimensions"]["exact_residual"] == "0"
    assert checks["sds_entropy_quadratic_positive_root"]["exact_residual"] == "0"
    assert checks["sds_mond_main_formula_normalized"]["exact_value"] == "1/2 + sqrt(2)/2"
    assert checks["mond_circular_velocity_relation"]["exact_v_fourth"] == "G*M*a_0"
    negative = checks["negative_control_wrong_quadratic_branch"]
    assert negative["status"] == "pass"
    assert negative["mutation_rejected"] is True
    assert negative["exact_residual"] != "0"


def test_all_scientific_and_data_seals_remain_closed(rebuilt: dict[str, object]) -> None:
    assert rebuilt["action_contract"] == {
        "contract_kind": "equation_only_proposal_intake",
        "fundamental_action": None,
        "field_content_closed": False,
        "variational_principle_registered": False,
        "euler_lagrange_map_registered": False,
        "boundary_terms_registered": False,
        "candidate_action_hash": None,
    }
    assert rebuilt["claim_seals"] == {
        "fundamental_action_registered": False,
        "formal_gr_equivalence_proven": False,
        "dark_matter_elimination_proven": False,
        "dark_energy_elimination_proven": False,
        "observational_pass": False,
        "theory_validity_claimed": False,
        "cuda_execution_performed": False,
        "automatic_downstream_enqueue_performed": False,
    }
    assert all(value is False for value in rebuilt["data_seals"].values())
    assert rebuilt["external_paid_llm_calls"] is False


def test_cuda_handoff_is_synthetic_only(rebuilt: dict[str, object]) -> None:
    handoff = rebuilt["cuda_handoff_contract"]
    assert handoff["ready_for_later_execution"] is True
    assert handoff["execution_performed"] is False
    assert handoff["numeric_type"] == "IEEE-754 binary64"
    assert len(handoff["kernels"]) == 5
    assert "equation 35" in handoff["blocked_constant_normalization"]


def test_config_and_artifact_tampering_fail_closed() -> None:
    config = _load(CONFIG)
    tampered_config = copy.deepcopy(config)
    tampered_config["primary_source"]["pdf_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="primary source binding changed"):
        build_intake(tampered_config, ROOT)

    artifact = _load(ARTIFACT)
    tampered_artifact = copy.deepcopy(artifact)
    tampered_artifact["claim_seals"]["formal_gr_equivalence_proven"] = True
    body = {key: value for key, value in tampered_artifact.items() if key != "content_sha256"}
    tampered_artifact["content_sha256"] = _sha(body)
    with pytest.raises(ValueError, match="scientific claim seals changed"):
        _validate_result(tampered_artifact)
