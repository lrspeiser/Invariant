from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.aether_seed_positive_energy_premise_campaign import (
    _sha,
    _validate_predecessor_record,
    build_aether_seed_positive_energy_premise_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "aether_seed_positive_energy_premise_campaign.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "aether-seed-positive-energy-premise-campaign.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_aether_seed_positive_energy_premise_campaign(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "ab71b868fd455985b0541ac77a34e0dfe1e7e97d8f6ce21cebff646f19872418"
    )


def test_two_rational_points_pass_exact_coupling_and_linear_energy_checks(rebuilt: dict) -> None:
    assert rebuilt["target_seed_count"] == 2
    assert rebuilt["decision_counts"] == {"blocked": 2}
    by_id = {item["seed_id"]: item for item in rebuilt["candidate_records"]}
    first = by_id["G3-0b8cb2d5591bf50d2465978d"]["exact_specialization"]
    assert first["coupling_combinations"] == {
        "c13": "1/4",
        "c14": "5/16",
        "c123": "3/8",
        "twist_sector_c1_minus_c3": "1/4",
    }
    assert first["restricted_theorem_coefficients"] == {
        "c14_times_one_minus_c14_over_two": "135/512",
        "one_minus_c13": "3/4",
    }
    second = by_id["G3-94086fa702500475b35ab002"]["exact_specialization"]
    assert second["coupling_combinations"] == {
        "c13": "3/8",
        "c14": "23/60",
        "c123": "11/24",
        "twist_sector_c1_minus_c3": "7/24",
    }
    assert second["restricted_theorem_coefficients"] == {
        "c14_times_one_minus_c14_over_two": "2231/7200",
        "one_minus_c13": "5/8",
    }
    for record in by_id.values():
        assert all(record["exact_specialization"]["coupling_domain"].values())
        assert record["premise_ledger"]["linearized_physical_energy"]["status"] == "pass"
        assert record["premise_ledger"]["restricted_theorem_coupling_domain"]["status"] == "pass"


def test_exact_twist_witness_is_the_first_fail_closed_premise(rebuilt: dict) -> None:
    witness = rebuilt["twisting_unit_aether_witness"]
    assert witness["unit_constraint_residual"] == "0"
    assert witness["frobenius_component_at_x_zero"] == "-1"
    assert witness["hypersurface_orthogonality_rejected_for_witness"] is True
    for record in rebuilt["candidate_records"]:
        assert record["decision"] == "blocked"
        assert record["first_missing_premise"] == "hypersurface_orthogonal_aether"
        assert record["negative_energy_counterexample_found"] is False
        assert record["restricted_subsector_theorem_status"] == "pass_in_restricted_subsector"
        assert record["premise_ledger"]["hypersurface_orthogonal_aether"]["status"] == "blocked"
        assert record["premise_ledger"]["remaining_restricted_theorem_premises"]["status"] == (
            "not_evaluated_after_first_blocker"
        )
        assert record["premise_ledger"]["generic_nonlinear_hamiltonian_stability"]["status"] == "blocked"


def test_adapters_and_eligibility_remain_exactly_sealed(rebuilt: dict) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert rebuilt["invoked_adapter_entrypoints"] == [
        item["entrypoint"] for item in config["formal_adapters"]
    ]
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["solar_bundles_generated"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_action_binding_is_rejected() -> None:
    predecessor = json.loads(
        (ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-compilation-campaign.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    record = next(
        item for item in predecessor["candidate_records"] if item["seed_id"] == config["target_seeds"][0]["seed_id"]
    )
    target = copy.deepcopy(config["target_seeds"][0])
    target["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="action hash mismatch"):
        _validate_predecessor_record(record, target)
