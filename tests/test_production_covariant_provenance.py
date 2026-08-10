import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.formal_backend import load_field_contract
from sigma_theory_compiler.high_throughput import (
    build_basis,
    candidate_id,
    correction_expression,
    decode_ordinal,
)
from sigma_theory_compiler.production_covariant_provenance import (
    map_candidate_to_covariant_action,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"
ARTIFACT = ROOT / "runs" / "engine" / "production-covariant-provenance-campaign.json"
SOURCE_SHA = "330c8f04e2da2e64dce39cf43f570e8a908367d3fd6e402ad90a8df55b913399"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(ordinal: int) -> dict:
    generator = _load(GENERATOR)
    decoded = decode_ordinal(
        generator["basis_count"], generator["max_action_terms"], ordinal
    )
    return {
        "candidate_id": candidate_id(generator["protocol_version"], decoded),
        "ordinal": ordinal,
        "term_ids": list(decoded["term_ids"]),
        "signs": list(decoded["signs"]),
        "correction_expression": correction_expression(
            decoded, build_basis(generator["basis_count"])
        ),
        "source_manifest_sha256": "a" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }


def _map(ordinal: int) -> dict:
    return map_candidate_to_covariant_action(
        _candidate(ordinal),
        _load(GENERATOR),
        _load(GRAMMAR),
        load_field_contract(CONTRACT),
        source_sha256=SOURCE_SHA,
    )


def test_exact_linear_q_and_q_minus_sqrt_x_mappings_are_hash_bound() -> None:
    pure_q = _map(7)
    mixed = _map(689)
    assert pure_q["decision"] == mixed["decision"] == "mapped"
    assert pure_q["covariant_action_provenance"]["basis_decomposition"] == {
        "AETHER_Q1": 1
    }
    assert mixed["covariant_action_provenance"]["basis_decomposition"] == {
        "AETHER_Q1": 1,
        "AETHER_X_SQRT1P": -1,
    }
    for result in (pure_q, mixed):
        provenance = result["covariant_action_provenance"]
        binding = provenance.pop("provenance_binding_sha256")
        assert binding == hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        provenance["provenance_binding_sha256"] = binding
        assert provenance["exact_static_shape_match"] is True
        assert provenance["universal_matter_coupling_preserved"] is True
        assert result["formal_preflight"]["decision"] == (
            "reject_higher_jet_regularity"
        )
        assert result["data_eligibility"] == ELIGIBILITY


def test_unsupported_and_nonlinear_q_atoms_remain_blocked_and_z_is_rejected() -> None:
    sqrt_q = _map(677)
    assert sqrt_q["decision"] == "blocked"
    assert "unsupported_generator_atom_in_covariant_action_dsl" in sqrt_q["blockers"]

    nonlinear = _map(3008915)
    assert nonlinear["decision"] == "blocked"
    assert "nonlinear_q_power_requires_separate_formal_derivation" in nonlinear["blockers"]

    z_candidate = _map(0)
    assert z_candidate["decision"] == "reject"
    assert z_candidate["reason"] == "forbidden_baryonic_action_atom"


def test_production_campaign_artifact_is_closed_and_content_hash_bound() -> None:
    artifact = _load(ARTIFACT)
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert artifact["input_candidate_count"] == 70
    assert artifact["decision_counts"] == {"blocked": 68, "mapped": 2}
    assert artifact["unblocked_covariant_lift_count"] == 2
    assert artifact["formal_preflight_counts"] == {
        "reject_higher_jet_regularity": 2
    }
    assert [record["ordinal"] for record in artifact["mapped_records"]] == [7, 689]
    assert artifact["rejected_records"] == []
    assert artifact["data_eligibility"] == ELIGIBILITY
