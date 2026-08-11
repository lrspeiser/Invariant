from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.scalable_formal_candidate_evidence_export import (
    _sha,
    build_scalable_formal_candidate_evidence_export,
    iter_scalable_formal_candidate_evidence_records,
    validate_scalable_formal_candidate_evidence_export,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "scalable_formal_candidate_evidence_export.json"
ARTIFACT = (
    ROOT / "runs" / "engine" / "scalable-formal-candidate-evidence-export-v1.1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def export() -> dict:
    return build_scalable_formal_candidate_evidence_export(_load(CONFIG), ROOT)


def test_exact_candidate_alias_and_final_decision_accounting(export: dict) -> None:
    records = iter_scalable_formal_candidate_evidence_records(export)
    assert export["parameter_cell_count"] == 256
    assert export["candidate_count"] == 163
    assert export["alias_count"] == 93
    assert export["final_decision_counts"] == {"blocked": 158, "pass": 3, "reject": 2}
    assert export["formal_pass_count"] == 3
    assert export["rank_eligible_count"] == 5
    assert export["family_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": {
            "candidate_count": 128,
            "decision_counts": {"blocked": 126, "reject": 2},
            "alias_count": 0,
        },
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": {
            "candidate_count": 1,
            "decision_counts": {"pass": 1},
            "alias_count": 31,
        },
        "CUBIC_HORNDESKI_G3_WEAK_CELL": {
            "candidate_count": 32,
            "decision_counts": {"blocked": 32},
            "alias_count": 0,
        },
        "KESSENCE_G2_CONVEX": {
            "candidate_count": 2,
            "decision_counts": {"pass": 2},
            "alias_count": 62,
        },
    }
    assert len({record["candidate_id"] for record in records}) == 163
    assert len({record["action_sha256"] for record in records}) == 163
    assert sum(record["alias_count"] for record in records) == 93
    for record in records:
        formula = record["theory_formula_inputs"]
        formula_body = {
            key: value for key, value in formula.items() if key != "formula_inputs_sha256"
        }
        assert formula["formula_inputs_sha256"] == _sha(formula_body)
        assert formula["action_content_sha256"] == record["action_sha256"]
        assert formula["fields"]
        assert formula["parameters"]
        assert formula["ordered_operator_densities"]


def test_blocked_candidates_are_unranked_and_only_exact_rejects_are_eligible(
    export: dict,
) -> None:
    decisions = Counter()
    for record in iter_scalable_formal_candidate_evidence_records(export):
        board = record["leaderboard_contract"]
        decisions[record["final_decision"]] += 1
        assert board["rank"] is None
        assert board["promotion_eligible"] is False
        if record["final_decision"] == "blocked":
            assert board["rank_eligible"] is False
            assert board["gate_completeness"] == "incomplete"
            assert board["comparison_data_class"] is None
        elif record["final_decision"] == "reject":
            assert record["family_id"] == "AETHER_K1234_PARAMETER_CELL"
            assert board["rank_eligible"] is True
            assert board["gate_completeness"] == "complete_for_category"
            assert board["comparison_data_class"] == (
                "aether_aligned_minkowski_principal_necessary_condition"
            )
            assert record["direct_metrics"] == {
                "c123": "0",
                "spin_0_principal_speed_squared": "0",
            }
        else:
            assert board["rank_eligible"] is True
            assert board["comparison_data_class"] == "full_formal_action_evidence"
            if record["family_id"] == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
                assert record["candidate_id"] == "G3A-e0eff4150989e3522dc6ba03"
                assert record["preflight_decision"] == "blocked"
                assert record["direct_metrics"] == {
                    "action_density_projection_equal": True,
                    "equivalent_parameter_cell_alias_count": 32,
                    "formal_pass_count": 1,
                }
            else:
                assert record["family_id"] == "KESSENCE_G2_CONVEX"
                assert record["preflight_decision"] == "pass"
                assert record["direct_metrics"] == {
                    "general_nonmaximal_positive_mass_theorem": True,
                    "actual_initial_data_set_instantiated": False,
                    "cell_preservation_or_global_evolution_proved": False,
                }
    assert decisions == Counter({"blocked": 158, "pass": 3, "reject": 2})


def test_candidate_metrics_are_directly_provenanced_and_never_aggregate(export: dict) -> None:
    records = iter_scalable_formal_candidate_evidence_records(export)
    g3 = [
        record
        for record in records
        if record["family_id"] == "CUBIC_HORNDESKI_G3_WEAK_CELL"
    ]
    assert len(g3) == 32
    for record in g3:
        assert record["direct_metrics"] == {}
        assert record["metric_source_sha256"] is None
        assert record["leaderboard_contract"]["rank_eligible"] is False
    g2 = [
        record for record in records if record["family_id"] == "KESSENCE_G2_CONVEX"
    ]
    assert len(g2) == 2
    assert all(record["metric_source_sha256"] for record in g2)


def test_validator_rejects_blocked_scoring_aggregate_metrics_and_record_tamper(
    export: dict,
) -> None:
    aggregate = copy.deepcopy(export)
    family_index = aggregate["candidate_record_columns"].index("family_id")
    metrics_index = aggregate["candidate_record_columns"].index("direct_metrics")
    metric_source_index = aggregate["candidate_record_columns"].index("metric_source_sha256")
    record = next(
        item for item in aggregate["candidate_records"] if item[family_index] == "KESSENCE_G2_CONVEX"
    )
    record[metrics_index] = {"aggregate_pass_count": 12}
    record[metric_source_index] = "1" * 64
    aggregate["candidate_record_registry_root_sha256"] = _sha(aggregate["candidate_records"])
    export_body = {key: value for key, value in aggregate.items() if key != "content_sha256"}
    aggregate["content_sha256"] = _sha(export_body)
    with pytest.raises(ValueError, match="aggregate"):
        validate_scalable_formal_candidate_evidence_export(aggregate)

    tampered = copy.deepcopy(export)
    action_index = tampered["candidate_record_columns"].index("action_sha256")
    tampered["candidate_records"][0][action_index] = "0" * 64
    export_body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = _sha(export_body)
    with pytest.raises(ValueError, match="record hash mismatch"):
        validate_scalable_formal_candidate_evidence_export(tampered)

    formula_tamper = copy.deepcopy(export)
    formula_index = formula_tamper["candidate_record_columns"].index(
        "theory_formula_inputs"
    )
    formula_tamper["candidate_records"][0][formula_index]["fields"].append("fake_field")
    formula_tamper["candidate_record_registry_root_sha256"] = _sha(
        formula_tamper["candidate_records"]
    )
    export_body = {
        key: value for key, value in formula_tamper.items() if key != "content_sha256"
    }
    formula_tamper["content_sha256"] = _sha(export_body)
    with pytest.raises(ValueError, match="formula inputs"):
        validate_scalable_formal_candidate_evidence_export(formula_tamper)


def test_binding_budget_and_data_tamper_fail_closed(tmp_path: Path) -> None:
    binding = _load(CONFIG)
    binding["g3_status"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        build_scalable_formal_candidate_evidence_export(binding, ROOT)

    g4_binding = _load(CONFIG)
    g4_binding["g4_followup"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        build_scalable_formal_candidate_evidence_export(g4_binding, ROOT)

    g2_binding = _load(CONFIG)
    g2_binding["g2_followup"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        build_scalable_formal_candidate_evidence_export(g2_binding, ROOT)

    opened = _load(CONFIG)
    opened["data_eligibility"]["observational_data_opened"] = True
    with pytest.raises(ValueError, match="eligibility"):
        build_scalable_formal_candidate_evidence_export(opened, ROOT)

    overflow = _load(CONFIG)
    overflow["budget"]["maximum_candidates"] = 164
    with pytest.raises(ValueError, match="budget"):
        build_scalable_formal_candidate_evidence_export(overflow, ROOT)

    assert not (tmp_path / "campaign-v1-live.sqlite").exists()


def test_committed_export_is_exact(export: dict) -> None:
    assert export == _load(ARTIFACT)
    validate_scalable_formal_candidate_evidence_export(export)
