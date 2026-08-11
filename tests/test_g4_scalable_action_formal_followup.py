from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.g4_scalable_action_formal_followup as followup_module
from sigma_theory_compiler.g4_scalable_action_formal_followup import (
    _sha,
    build_g4_scalable_action_formal_followup,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g4_scalable_action_formal_followup.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g4-scalable-action-formal-followup.json"
EXPECTED_CONFIG_FILE_SHA256 = "f7aa1022fbfb6abaa18d94f9674a7c26edb3e4ec3cc4181b6d31eabc820cae14"
EXPECTED_SOURCE_FILE_SHA256 = "6e359eb0f9f1f9ee61bbd0d4ccf186dc67e2fbbebd623a8d23410c0b361076d9"
EXPECTED_ARTIFACT_FILE_SHA256 = "b8aefbcfef9d9db3059c5deaa235de84dbb7f234873ea5e4d2a73592026c9d9c"
EXPECTED_CONTENT_SHA256 = "7f470af2f26051da8429cd9663ea846277a84d624555e2c4bd48baecc08989db"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_g4_scalable_action_formal_followup(_config(), ROOT)


def test_replay_matches_hash_bound_portable_artifact(artifact: dict) -> None:
    assert artifact == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == build_g4_scalable_action_formal_followup(_config(), ROOT)
    assert artifact["content_sha256"] == EXPECTED_CONTENT_SHA256
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == EXPECTED_CONFIG_FILE_SHA256
    assert (
        hashlib.sha256(
            (ROOT / "src/sigma_theory_compiler/g4_scalable_action_formal_followup.py").read_bytes()
        ).hexdigest()
        == EXPECTED_SOURCE_FILE_SHA256
    )
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == EXPECTED_ARTIFACT_FILE_SHA256
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)


def test_equivalence_is_action_level_not_family_label(artifact: dict) -> None:
    certificate = artifact["equivalence_certificate"]
    assert artifact["candidate_id"] == "G3A-e0eff4150989e3522dc6ba03"
    assert artifact["representative_cell_id"] == "G3PC-5c546b9ecf5d2a38c8af24c2"
    assert artifact["equivalent_parameter_cell_alias_count"] == 32
    assert certificate["full_typed_action_hashes_equal"] is False
    assert certificate["scalable_action_sha256"] == (
        "7dd636e53f7cc161feabcb02b1f575bc1da3bd6b84033e870d2d9024c6cd5d21"
    )
    assert certificate["reviewed_seed_action_sha256"] == (
        "6ddd6502d110ead90ff494a6569213ec2e61a0b046dfa86344bb1980df6abc90"
    )
    assert certificate["action_density_projection_equal"] is True
    assert certificate["operator_densities_equal"] is True
    assert certificate["action_parameters_equal"] == {"G2": True, "G4": True}
    assert certificate["universal_matter_coupling_equal"] is True
    assert certificate["family_label_used_as_equivalence_evidence"] is False
    assert certificate["action_density_projection_sha256"] == (
        "614f33a57c0aeb27de2f36d8fe857c0e2080062adb83aa5a2e80426d0fdb53e7"
    )


def test_reviewed_formal_pass_transfers_only_on_narrower_domain(artifact: dict) -> None:
    certificate = artifact["equivalence_certificate"]
    assert certificate["scalable_representative_domain"] == "abs(phi)<=1/32"
    assert certificate["reviewed_seed_domain"] == "abs(phi)<=1"
    assert certificate["representative_domain_is_subset"] is True
    assert certificate["all_alias_domains_inside_reviewed_domain"] is True
    assert artifact["preflight_decision"] == "blocked"
    assert artifact["preflight_blocker"] == "family_prerequisite_not_passed"
    assert artifact["formal_followup_decision"] == "pass"
    assert artifact["decision_counts"] == {"pass": 1}
    assert artifact["formal_pass_count"] == 1
    assert artifact["necessary_condition_rejection_count"] == 0
    assert artifact["gate_ledger"]["full_typed_action_hash_identity"] == {
        "status": "not_equal_expected",
        "used_as_equivalence_proof": False,
    }
    assert artifact["gate_ledger"]["exact_covariant_density_equivalence"] == {"status": "pass"}
    assert artifact["gate_ledger"]["formal_prerequisite_completion"] == {"status": "pass"}


def test_candidate_preflight_and_reviewed_pass_lineage_are_exact(artifact: dict) -> None:
    provenance = artifact["provenance"]
    assert provenance["action_density_equivalence_sha256"] == (
        "e0eff4150989e3522dc6ba03d7169949993be9aae63cdd42bf5a9bfbedc535d5"
    )
    assert provenance["preflight_input_lineage_sha256"] == (
        "5e791aa9eaf3f81b3f8ff4afbbb17290e767a0c67f82cc22c23b4d40a6993cbe"
    )
    assert provenance["preflight_result_sha256"] == (
        "26a676093109d348f1dd2f0638c1582091dab94d07f8618d263bacb2ded1b97f"
    )
    assert provenance["preflight_record_sha256"] == (
        "4e4893abefce5aa8aedd6d477bbe4eed831c04e562bb3c38d619ea500e7caca1"
    )
    provenance_body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert provenance["binding_sha256"] == _sha(provenance_body)


def test_observations_and_downstream_bundles_remain_sealed(artifact: dict) -> None:
    assert artifact["solar_bundle_count"] == 0
    assert artifact["gate_ledger"]["solar_and_observations"] == {"status": "sealed"}
    assert artifact["observational_data_opened"] is False
    assert artifact["dark_matter_or_halo_inputs"] is False
    assert artifact["redshift_distance_inputs"] is False
    assert artifact["paid_llm_spend_usd"] == 0.0
    assert artifact["data_eligibility"] == {
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_calls": False,
    }


def test_density_or_reviewed_action_tamper_and_observation_opening_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_projection = followup_module._density_projection
    calls = 0

    def mismatched_projection(action_ir: dict) -> dict:
        nonlocal calls
        calls += 1
        projection = original_projection(action_ir)
        if calls == 2:
            projection = copy.deepcopy(projection)
            projection["action_parameters"]["G4"] = "1/2+(1/99)*phi^2"
        return projection

    monkeypatch.setattr(followup_module, "_density_projection", mismatched_projection)
    with pytest.raises(ValueError, match="not exact-equivalent"):
        build_g4_scalable_action_formal_followup(_config(), ROOT)
    monkeypatch.setattr(followup_module, "_density_projection", original_projection)

    original_load = followup_module._load_bound

    def tampered_load(root: Path, binding: dict) -> dict:
        value = original_load(root, binding)
        if binding["path"].endswith("g3-g4-nonunitary-gauge-bypass-audit.json"):
            value = copy.deepcopy(value)
            record = next(item for item in value["candidate_records"] if item["family"] == "G4")
            record["action_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(followup_module, "_load_bound", tampered_load)
    with pytest.raises(ValueError, match="formal pass action binding changed"):
        build_g4_scalable_action_formal_followup(_config(), ROOT)
    monkeypatch.setattr(followup_module, "_load_bound", original_load)

    opened = _config()
    opened["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened observations"):
        build_g4_scalable_action_formal_followup(opened, ROOT)
