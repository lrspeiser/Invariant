from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_nonlinear_lift_characteristic_gate import (
    CHARACTERISTIC_BLOCKER,
    IFT_BLOCKER,
)
from sigma_theory_compiler.future_aether_regular_adm_inverse_margin_gate import (
    BLOCKER,
    build_future_aether_regular_adm_inverse_margin_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_regular_adm_inverse_margin_gate.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-regular-adm-inverse-margin-gate.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-nonlinear-lift-characteristic-gate.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_regular_adm_inverse_margin_gate(_config(), ROOT)


def test_exact_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {
        BLOCKER: 3,
        CHARACTERISTIC_BLOCKER: 11,
    }
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0


def test_candidate_action_and_predecessor_bindings_are_preserved(rebuilt: dict) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    expected = {item["candidate_id"]: item for item in source["candidate_records"]}
    for item in rebuilt["candidate_records"]:
        predecessor = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert item["compilation_receipt_sha256"] == predecessor["compilation_receipt_sha256"]
        assert item["source_characteristic_record_sha256"] == predecessor["content_sha256"]
        assert item["exact_specialization"] == predecessor["exact_specialization"]


def test_forced_characteristic_candidates_remain_not_reached(rebuilt: dict) -> None:
    forced = [
        item
        for item in rebuilt["candidate_records"]
        if item["first_blocker"] == CHARACTERISTIC_BLOCKER
    ]
    assert len(forced) == 11
    for item in forced:
        assert item["regular_ADM_inverse_margin_certificate"] is None
        assert (
            item["gate_ledger"]["regular_ADM_characteristic_free_seed"]["status"] == "not_reached"
        )
        assert (
            item["gate_ledger"]["uniform_Aether_Legendre_block_inverse"]["status"] == "not_reached"
        )


def test_regular_candidates_have_exact_inverse_and_negative_energy_certificates(
    rebuilt: dict,
) -> None:
    regular = [item for item in rebuilt["candidate_records"] if item["first_blocker"] == BLOCKER]
    assert len(regular) == 3
    assert rebuilt["uniform_Aether_Legendre_block_inverse_pass_count"] == 3
    assert rebuilt["strict_negative_source_margin_pass_count"] == 3
    assert rebuilt["kinetic_block_inverse_bound_counts"] == {
        "1": 1,
        "1490944/175187": 1,
        "5963776/1271033": 1,
    }
    for item in regular:
        certificate = item["regular_ADM_inverse_margin_certificate"]
        margin = Fraction(certificate["uniform_normalized_Legendre_margin"])
        inverse = Fraction(certificate["kinetic_block_inverse_bound"])
        energy_upper = Fraction(certificate["static_source_energy_upper_bound_over_pi"])
        energy_margin = Fraction(certificate["strict_negative_source_margin"])
        assert margin > 0
        assert inverse == 1 / margin
        assert energy_upper < 0
        assert energy_margin == -energy_upper
        assert certificate["uniform_Aether_Legendre_block_inverse_proven"] is True
        assert item["gate_ledger"]["uniform_Aether_Legendre_block_inverse"]["status"] == "pass"
        assert item["gate_ledger"]["strict_negative_source_energy_margin"]["status"] == "pass"


def test_result_does_not_overclaim_the_missing_coupled_theorem(rebuilt: dict) -> None:
    assert rebuilt["weighted_full_constraint_operator_isomorphism_pass_count"] == 0
    assert rebuilt["nonlinear_Frechet_remainder_bound_pass_count"] == 0
    assert rebuilt["completed_boundary_sign_persistence_count"] == 0
    assert rebuilt["constraint_satisfying_negative_total_energy_datum_count"] == 0
    for item in rebuilt["candidate_records"]:
        assert item["decision"] == "blocked"
        assert item["formal_pass"] is False
        assert item["candidate_rejection_authorized"] is False
        assert item["constraint_satisfying_negative_total_energy_datum_proven"] is False
        assert item["full_formal_completion_claimed"] is False


def test_hash_provenance_and_all_data_seals(rebuilt: dict) -> None:
    assert rebuilt["bounded_regular_ADM_inverse_margin_gate_completed"] is True
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    provenance = rebuilt["provenance"]
    provenance_body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert provenance["binding_sha256"] == _sha(provenance_body)
    for item in rebuilt["candidate_records"]:
        body = {key: value for key, value in item.items() if key != "content_sha256"}
        assert item["content_sha256"] == _sha(body)
        assert item["data_eligibility"] == rebuilt["data_eligibility"]
        assert item["observational_data_opened"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.update(
                data_eligibility={
                    **config["data_eligibility"],
                    "observational_data_opened": True,
                }
            ),
            "eligibility is open",
        ),
        (lambda config: config.update(observational_authorization=True), "opened observations"),
        (lambda config: config.update(external_paid_llm_calls=True), "enabled paid LLM calls"),
        (
            lambda config: config["budget"].update(maximum_regular_adm_candidates=4),
            "budget is not exact",
        ),
        (
            lambda config: config["source_characteristic_artifact"].update(content_sha256="0" * 64),
            "content hash mismatch",
        ),
        (
            lambda config: config["campaign_implementation"].update(file_sha256="0" * 64),
            "file hash mismatch",
        ),
    ],
)
def test_open_seals_budget_and_hash_tampering_fail_closed(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_future_aether_regular_adm_inverse_margin_gate(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_characteristic_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_regular_adm_inverse_margin_gate(config, ROOT)


def test_predecessor_partition_was_exactly_the_expected_three_regular_records() -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    assert sum(item["first_blocker"] == IFT_BLOCKER for item in source["candidate_records"]) == 3
