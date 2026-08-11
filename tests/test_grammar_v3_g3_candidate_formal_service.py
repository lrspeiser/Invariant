from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_g3_candidate_formal_service import (
    GrammarV3G3CandidateFormalService,
    portable_status,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_g3_candidate_formal_service.json"
ARTIFACT = ROOT / "runs" / "engine" / "grammar-v3-g3-candidate-formal-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _enabled() -> dict:
    config = _load(CONFIG)
    config["execution_enabled"] = True
    return config


def _run(directory: Path) -> tuple[GrammarV3G3CandidateFormalService, dict]:
    service = GrammarV3G3CandidateFormalService(directory, _enabled(), ROOT)
    assert service.enqueue()["accepted"] == 32
    result = service.run_bounded()
    assert result["executed"] == 32
    return service, result["status"]


def test_checked_in_disabled_and_exact_32_candidate_grid_is_bound(tmp_path: Path) -> None:
    service = GrammarV3G3CandidateFormalService(tmp_path / "disabled", _load(CONFIG), ROOT)
    assert len(service.work_items) == 32
    assert {item["beta"] for item in service.work_items} == {
        str(Fraction(index, 3200)) for index in range(1, 33)
    }
    assert all(item["typed_action_ir_sha256"] for item in service.work_items)
    assert all(item["preflight_result_sha256"] for item in service.work_items)
    assert all(item["admission_result_sha256"] for item in service.work_items)
    assert all(item["candidate_evidence_sha256"] for item in service.work_items)
    assert service.status()["execution_enabled"] is False
    with pytest.raises(RuntimeError, match="disabled"):
        service.enqueue()


def test_strongest_reviewed_g3_gates_pass_but_af_global_claims_block(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "real"
    _, status = _run(directory)
    assert status["work_state_counts"] == {"succeeded": 32}
    assert status["decision_counts"] == {"blocked": 32}
    assert status["blocker_counts"] == {
        "uniformly_invertible_Delta_N_on_AF_decaying_gradient_domain": 32
    }
    pass_gates = {
        "candidate_action_preflight_admission_binding",
        "exact_parameter_cell_and_weak_envelope",
        "covariant_G2_G3_variation_noether",
        "adm_primary_degeneracy",
        "uniform_local_principal_symbol",
        "uniform_local_common_time_and_BSSN_cone",
        "all_spatial_covector_directions",
        "full_candidate_lapse_operator_derivation",
        "periodic_lapse_coercivity_and_zero_mode_exclusion",
        "distributed_Dirac_on_periodic_cell",
        "af_finite_scalar_energy_tail",
        "af_reference_principal_common_cone",
    }
    blocked_gates = {
        "af_uniform_lapse_Dirac_invertibility",
        "af_Einstein_constraint_solution",
        "global_hamiltonian_energy",
        "full_formal_completion",
    }
    assert all(status["gate_counts"][gate] == {"pass": 32} for gate in pass_gates)
    assert all(status["gate_counts"][gate] == {"blocked": 32} for gate in blocked_gates)
    assert status["af_global_constraint_solution_proved"] is False
    assert status["global_positive_energy_proved"] is False
    assert status["full_formal_pass_count"] == 0
    assert status["necessary_condition_rejection_count"] == 0
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert status["observational_data_opened"] is False
    assert status["paid_llm_spend_usd"] == 0.0

    resumed = GrammarV3G3CandidateFormalService(directory, _enabled(), ROOT)
    assert resumed.enqueue()["duplicate"] == 32
    replay = resumed.run_bounded()
    assert replay["executed"] == 0
    assert portable_status(replay["status"]) == portable_status(status)


def test_candidate_evidence_is_exact_and_margins_cover_full_grid(tmp_path: Path) -> None:
    service = GrammarV3G3CandidateFormalService(tmp_path / "evidence", _load(CONFIG), ROOT)
    assert len(service.candidate_evidence) == 32
    for item in service.work_items:
        evidence = service.candidate_evidence[item["candidate_id"]]
        margins = evidence["principal_uniform_margins"]
        assert margins["common_time_covector_upper_P00"] < 0
        assert margins["spatial_block_eigenvalue_lower"] > 0
        assert margins["slicing_cone_polynomial_upper"] < 0
        assert Fraction(evidence["periodic_lapse_lower_bound"]) > 0
        assert evidence["af_lapse_status"] == (
            "blocked_not_boundedly_invertible_on_L2_R3"
        )


def test_missing_adapter_and_expired_lease_recover_fail_closed(tmp_path: Path) -> None:
    missing = frozenset({"candidate_componentwise_principal_common_cone"})
    directory = tmp_path / "missing"
    service = GrammarV3G3CandidateFormalService(
        directory, _enabled(), ROOT, missing_adapter_ids=missing
    )
    service.enqueue()
    abandoned = service.coordinator.claim("cpu", "crashed", lease_seconds=-1)
    assert abandoned is not None
    resumed = GrammarV3G3CandidateFormalService(
        directory, _enabled(), ROOT, missing_adapter_ids=missing
    )
    result = resumed.run_bounded()
    assert result["recovered"] == {"recovered": 1, "failed": 0}
    status = result["status"]
    assert status["decision_counts"] == {"blocked": 32}
    assert status["gate_counts"]["uniform_local_principal_symbol"] == {"blocked": 32}
    assert status["blocker_counts"] == {
        "reviewed_g3_adapter_missing:candidate_componentwise_principal_common_cone": 32
    }


def test_upstream_hash_config_and_persisted_payload_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    config = _enabled()
    config["af_audit"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        GrammarV3G3CandidateFormalService(tmp_path / "binding", config, ROOT)

    unsealed = _enabled()
    unsealed["data_eligibility"]["observational_data_opened"] = True
    with pytest.raises(ValueError, match="eligibility"):
        GrammarV3G3CandidateFormalService(tmp_path / "unsealed", unsealed, ROOT)

    directory = tmp_path / "tamper"
    service = GrammarV3G3CandidateFormalService(directory, _enabled(), ROOT)
    service.enqueue()
    with service.coordinator.connect() as connection:
        row = connection.execute(
            "SELECT work_id,payload_json FROM work ORDER BY ordinal LIMIT 1"
        ).fetchone()
        payload = json.loads(row["payload_json"])
        payload["candidate_evidence_sha256"] = "0" * 64
        connection.execute(
            "UPDATE work SET payload_json=? WHERE work_id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), row["work_id"]),
        )
    with pytest.raises(ValueError, match="payload was tampered"):
        service.status()


def test_committed_portable_status_is_exact(tmp_path: Path) -> None:
    _, status = _run(tmp_path / "artifact")
    assert portable_status(status) == _load(ARTIFACT)
