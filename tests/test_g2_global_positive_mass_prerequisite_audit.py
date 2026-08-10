from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g2_global_positive_mass_prerequisite_audit import (
    _sha,
    _validate_contract,
    _validate_target,
    build_g2_global_positive_mass_prerequisite_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g2_global_positive_mass_prerequisite_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g2-global-positive-mass-prerequisite-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g2_global_positive_mass_prerequisite_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "77b7a6ba27d372ca872c8d7ee44f89165f8c00121511b2ee9b07f2353ffbcbdb"
    )


def test_global_contract_binds_function_space_lapse_constraints_and_charge(rebuilt: dict) -> None:
    contract = rebuilt["global_contract"]
    assert rebuilt["global_contract_sha256"] == _sha(contract)
    assert contract["contract_kind"] == "conditional_theorem_domain_not_claim_of_data_existence"
    assert contract["initial_slice"]["weighted_regularities"] == {
        "p": "p>3",
        "q": "1/2<q<1",
        "h_minus_delta": "W^{2,p}_{-q}",
        "K_ij": "W^{1,p}_{-1-q}",
    }
    assert contract["hamiltonian_smearing"]["lapse"] == "N>=N_min>0 and N=1+O_2(r^-q)"
    assert contract["constraint_domain"]["restricted_core_slice"] == "K=0"
    assert contract["boundary_contract"]["gravitational_charge"] == (
        "ADM_four_momentum_of_the_asymptotically_euclidean_end"
    )


def test_candidate_dec_and_scalar_boundary_falloff_are_exact(rebuilt: dict) -> None:
    by_id = {item["seed_id"]: item for item in rebuilt["candidate_records"]}
    expected = {
        "G3-a82c572555e5d79686bc4a4a": {
            "rho": "X+(3/4)*X^2+(1+(1/2)*X)*s_squared",
            "rho_squared_minus_j_squared": (
                "(X+(3/4)*X^2)^2+(1/2)*X^2*(1+(1/2)*X)*s_squared"
            ),
        },
        "G3-e8b35002cfc9c60691a2f67b": {
            "rho": "X+(3/8)*X^2+(1+(1/4)*X)*s_squared",
            "rho_squared_minus_j_squared": (
                "(X+(3/8)*X^2)^2+(1/4)*X^2*(1+(1/4)*X)*s_squared"
            ),
        },
    }
    for seed_id, expressions in expected.items():
        certificate = by_id[seed_id]["dec_and_boundary_certificate"]
        assert certificate["stress_projections"]["rho"] == expressions["rho"]
        assert certificate["stress_projections"]["rho_squared_minus_j_squared"] == (
            expressions["rho_squared_minus_j_squared"]
        )
        assert certificate["dominant_energy_condition"]["status"] == "pass"
        assert certificate["falloff_consequences"] == {
            "X": "O(r^-4)",
            "rho_and_j": "O(r^-4)",
            "matter_sources_integrable": True,
            "scalar_surface_variation": "O(r^-1)->0",
            "extra_scalar_boundary_charge": False,
        }


def test_restricted_positive_mass_pass_does_not_promote_general_phase_space(
    rebuilt: dict,
) -> None:
    assert rebuilt["restricted_maximal_slice_pass_count"] == 2
    assert rebuilt["decision_counts"] == {"blocked": 2}
    assert rebuilt["full_formal_pass_count"] == 0
    for record in rebuilt["candidate_records"]:
        gates = record["gate_ledger"]
        assert gates["candidate_DEC_on_contract_domain"]["status"] == "pass"
        assert gates["restricted_maximal_slice_positive_mass"]["status"] == "pass"
        assert gates["general_nonmaximal_positive_mass"]["status"] == "blocked"
        assert gates["global_positive_energy"]["status"] == "blocked"
        assert record["decision"] == "blocked"
        assert record["first_missing_premise"] == (
            "hash_bound_general_nonmaximal_positive_mass_theorem"
        )
        assert record["negative_total_energy_counterexample_found"] is False


def test_replayed_core_is_not_substituted_for_candidate_evidence_and_seals_hold(
    rebuilt: dict,
) -> None:
    core = rebuilt["positive_mass_core_replay"]
    assert core["generic_status"] == "unresolved"
    assert core["eligible_as_direct_candidate_evidence"] is False
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    assert all(not item["solar_bundle"]["generated"] for item in rebuilt["candidate_records"])


def test_weakened_contract_and_tampered_action_are_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    weakened = copy.deepcopy(config["global_contract"])
    weakened["constraint_domain"]["restricted_core_slice"] = "unrestricted_K"
    with pytest.raises(ValueError, match="global function-space or boundary contract changed"):
        _validate_contract(weakened)

    predecessor = json.loads((ROOT / config["predecessor"]["path"]).read_text(encoding="utf-8"))
    target = copy.deepcopy(config["target_seeds"][0])
    record = next(
        item for item in predecessor["candidate_records"] if item["seed_id"] == target["seed_id"]
    )
    target["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="action hash mismatch"):
        _validate_target(record, target)
