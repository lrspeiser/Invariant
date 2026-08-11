from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.future_candidate_action_dossier import (
    build_future_candidate_action_dossier,
    iter_future_candidate_action_dossiers,
    validate_future_candidate_action_dossier,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/future_candidate_action_dossier.json"
ARTIFACT = ROOT / "runs/engine/future-candidate-action-dossier.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_exact_rebuild_and_formula_dossier_ledger() -> None:
    config = _load(CONFIG)
    artifact = _load(ARTIFACT)
    assert build_future_candidate_action_dossier(config, ROOT) == artifact
    validate_future_candidate_action_dossier(artifact)
    records = list(iter_future_candidate_action_dossiers(artifact))
    assert artifact["candidate_count"] == len(records) == 19
    assert artifact["family_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": 16,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 3,
    }
    assert artifact["decision_counts"] == {"blocked": 17, "reject": 2}
    assert artifact["ranked_candidate_count"] == 0
    assert len({record["candidate_id"] for record in records}) == 19
    assert len({record["action"]["action_sha256"] for record in records}) == 19
    assert all(record["comparison_contract"]["rank"] is None for record in records)
    assert all(record["comparison_contract"]["rank_eligible"] is False for record in records)


def test_human_readable_action_is_exact_density_display_not_family_template() -> None:
    artifact = _load(ARTIFACT)
    for record in artifact["dossiers"]:
        action = record["action"]
        display = action["human_readable_action"]
        densities = [operator["density"] for operator in action["ordered_operator_densities"]]
        assert display["display_text"] == (
            "S = integral d^4x [" + " + ".join(f"({density})" for density in densities) + "]"
        )
        assert display["display_kind"] == ("verbatim_ordered_covariant_density_concatenation")
        assert "Family labels supply no action terms" in display["scope"]
        assert action["matter_coupling"] == {"metric": "g_mu_nu", "universal": True}


def test_proof_hierarchy_distinguishes_formula_formal_and_observation_scope() -> None:
    artifact = _load(ARTIFACT)
    rejected = 0
    for record in artifact["dossiers"]:
        nodes = {node["node_id"]: node for node in record["hierarchy_nodes"]}
        assert set(nodes) == {
            "exact_compiler_action",
            "reviewed_formal_evidence",
            "downstream_observational_evidence",
        }
        assert nodes["exact_compiler_action"]["status"] == "proven"
        assert nodes["downstream_observational_evidence"]["status"] == "blocked"
        assert nodes["downstream_observational_evidence"]["evidence"] == {
            "source": "sealed_policy",
            "observation_opening_allowed": False,
        }
        if record["formal_decision"] == "reject":
            rejected += 1
            assert nodes["reviewed_formal_evidence"]["status"] == "rejected"
        else:
            assert record["formal_decision"] == "blocked"
            assert nodes["reviewed_formal_evidence"]["status"] == "blocked"
    assert rejected == 2
    g3_records = [
        record
        for record in artifact["dossiers"]
        if record["family_id"] == "CUBIC_HORNDESKI_G3_WEAK_CELL"
    ]
    assert len(g3_records) == 3
    assert all(
        record["first_blocker"] == "asymptotically_flat_or_global_energy_domain_missing"
        and "uniform principal/common cone" in record["hierarchy_nodes"][1]["scope"]
        and "periodic Dirac gates pass" in record["hierarchy_nodes"][1]["scope"]
        for record in g3_records
    )
    aether_records = [
        record
        for record in artifact["dossiers"]
        if record["family_id"] == "AETHER_K1234_PARAMETER_CELL"
        and record["formal_decision"] == "blocked"
    ]
    assert len(aether_records) == 14
    assert all(
        record["first_blocker"]
        == "constraint_satisfying_asymptotically_Euclidean_completion_of_negative_twist_witness"
        and "fails both coupled constraints" in record["hierarchy_nodes"][1]["scope"]
        for record in aether_records
    )


def test_artifact_and_source_tampering_fail_closed(tmp_path: Path) -> None:
    artifact = _load(ARTIFACT)
    tampered = copy.deepcopy(artifact)
    tampered["dossiers"][0]["action"]["parameters"]["c1"] = "999"
    with pytest.raises(ValueError, match="artifact is invalid"):
        validate_future_candidate_action_dossier(tampered)

    config = _load(CONFIG)
    altered = copy.deepcopy(config)
    altered["source_bindings"]["preflight"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="preflight file hash mismatch"):
        build_future_candidate_action_dossier(altered, ROOT)


def test_artifact_is_portable_and_secret_safe() -> None:
    raw = ARTIFACT.read_bytes()
    artifact = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == (
        "16fc9ea0ca0605a5858ecc65181ef0e4dfc9434dd68459de893ff735383c130f"
    )
    encoded = raw.decode("utf-8")
    assert "C:\\" not in encoded
    assert "/Users/" not in encoded
    assert "/home/" not in encoded
    assert artifact["data_eligibility"] == {
        "dark_matter_or_halo_inputs": False,
        "observational_data_opened": False,
        "paid_llm_calls": False,
        "redshift_distance_inputs": False,
    }
    assert artifact["observational_authorization"] is False
    assert artifact["observational_data_opened"] is False
    assert artifact["paid_llm_spend_usd"] == 0.0
