from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.covariant_grammar_v3_seed_compilation_campaign import (
    _canonical,
    _sha,
    _validate_seed,
    build_covariant_grammar_v3_seed_compilation_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "covariant_grammar_v3_seed_compilation_campaign.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-compilation-campaign.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_covariant_grammar_v3_seed_compilation_campaign(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "88c72002d12fd57a8ef79b166363319fe5dc372b52901c1e581bb382fa3e0c21"
    )


def test_all_six_candidate_actions_are_hash_bound_and_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["seed_count"] == 6
    assert rebuilt["decision_counts"] == {"blocked": 6}
    assert rebuilt["adapter_invocation_count"] == 9
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    for record in rebuilt["candidate_records"]:
        action = record["typed_action_ir"]
        action_body = {key: value for key, value in action.items() if key != "content_sha256"}
        assert action["content_sha256"] == _sha(action_body)
        assert record["provenance"]["action_ir_sha256"] == action["content_sha256"]
        assert record["declared_adapter_entrypoints"] == record["invoked_adapter_entrypoints"]
        assert all(item["callable_invoked"] for item in record["adapter_invocations"])
        assert record["solar_known_answer_bundle"] == {
            "generated": False,
            "status": "blocked",
            "reason": "formal_prerequisites_incomplete",
        }
        serialized_action = _canonical(action)
        assert "Q_a_u" not in serialized_action
        assert "T2_m" not in serialized_action
        assert "z_b" not in serialized_action


def test_gate_ledgers_do_not_promote_generic_or_conditional_controls(rebuilt: dict) -> None:
    by_family: dict[str, list[dict]] = {}
    for record in rebuilt["candidate_records"]:
        by_family.setdefault(record["family_id"], []).append(record)
    for record in by_family["AETHER_K1234_PARAMETER_CELL"]:
        assert record["gate_ledger"]["adm_dirac"]["status"] == "pass"
        assert record["gate_ledger"]["principal_symbol"]["status"] == "pass"
        assert record["gate_ledger"]["hamiltonian_stability"]["status"] == "unresolved"
        health = record["adapter_invocations"][0]["evidence_summary"]
        assert list(health["gate_statuses"].values()).count("pass") == 12
        assert list(health["gate_statuses"].values()).count("unresolved") == 1
    for record in by_family["KESSENCE_G2_CONVEX"]:
        assert record["parameter_certificate"]["local_convexity"] == "pass"
        assert record["gate_ledger"]["principal_symbol"]["status"] == "blocked"
        assert record["gate_ledger"]["hamiltonian_stability"]["status"] == "blocked"
    g3 = by_family["CUBIC_HORNDESKI_G3_WEAK_CELL"][0]
    assert g3["parameter_certificate"]["status"] == "conditional_unresolved"
    assert g3["gate_ledger"]["principal_symbol"]["status"] == "blocked"
    g4 = by_family["CONFORMAL_G4_PHI_SCALAR_TENSOR"][0]
    assert g4["parameter_certificate"]["G4_positive"] == "pass"
    assert g4["gate_ledger"]["adm_dirac"]["status"] == "blocked"
    assert g4["gate_ledger"]["principal_symbol"]["status"] == "blocked"


def test_tampered_seed_lineage_is_rejected_before_adapter_invocation() -> None:
    manifest = json.loads(
        (ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    seed = copy.deepcopy(manifest["scalable_generator_hook"]["concrete_seeds"][0])
    family = next(
        item for item in manifest["typed_family_seeds"] if item["family_id"] == seed["family_id"]
    )
    seed["parameters"]["c1"] = "999"
    with pytest.raises(ValueError, match="seed lineage"):
        _validate_seed(seed, family)
