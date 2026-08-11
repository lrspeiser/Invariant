from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.scalable_candidate_structural_metrics_export import (
    EQUIVALENCE_CLASS,
    STRUCTURAL_CLASS,
    _metrics,
    _sha,
    build_scalable_candidate_structural_metrics_export,
    validate_scalable_candidate_structural_metrics_export,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "scalable_candidate_structural_metrics_export.json"
ARTIFACT = ROOT / "runs" / "engine" / "scalable-candidate-structural-metrics.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def export() -> dict:
    return build_scalable_candidate_structural_metrics_export(_load(CONFIG), ROOT)


def _rehash_record(record: dict) -> None:
    record["content_sha256"] = _sha(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )


def _rehash_export(export: dict) -> None:
    export["candidate_record_registry_root_sha256"] = _sha(
        [record["content_sha256"] for record in export["candidate_records"]]
    )
    export["content_sha256"] = _sha(
        {key: value for key, value in export.items() if key != "content_sha256"}
    )


def test_exact_population_alias_and_family_metrics(export: dict) -> None:
    assert export["candidate_count"] == 163
    assert export["alias_count"] == 93
    assert export["parameter_cell_count"] == 256
    assert export["representative_exact_action_class_count"] == 163
    assert export["representative_exact_action_duplicate_count"] == 0
    assert export["formal_decision_counts"] == {"blocked": 160, "pass": 1, "reject": 2}
    assert export["structural_measurement_counts"] == {"measured": 163}
    assert export["family_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": 128,
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": 1,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
        "KESSENCE_G2_CONVEX": 2,
    }
    assert export["parameter_cell_class_size_counts"] == {"1": 160, "32": 3}
    assert Counter(
        (
            record["family_id"],
            record["structural_metrics"]["operator_count"],
            record["structural_metrics"]["field_count"],
            record["structural_metrics"]["parameter_count"],
        )
        for record in export["candidate_records"]
    ) == Counter(
        {
            ("AETHER_K1234_PARAMETER_CELL", 6, 3, 4): 128,
            ("CONFORMAL_G4_PHI_SCALAR_TENSOR", 2, 2, 3): 1,
            ("CUBIC_HORNDESKI_G3_WEAK_CELL", 3, 2, 3): 32,
            ("KESSENCE_G2_CONVEX", 2, 2, 2): 2,
        }
    )


def test_formula_inputs_rederive_every_metric_and_exact_action_class(export: dict) -> None:
    for record in export["candidate_records"]:
        formula = record["formula_structure"]
        formula_body = {
            "fields": formula["fields"],
            "parameters": formula["parameters"],
            "ordered_operator_densities": formula["ordered_operator_densities"],
            "action_content_sha256": formula["action_content_sha256"],
        }
        assert formula["formula_inputs_sha256"] == _sha(formula_body)
        assert formula["action_content_sha256"] == record["action_sha256"]
        assert record["structural_metrics"] == _metrics(formula)
        equivalence = record["equivalence_evidence"]
        assert equivalence["exact_action_equivalence_class_sha256"] == record["action_sha256"]
        assert equivalence["representative_action_class_size"] == 1
        assert equivalence["parameter_cell_class_size"] == equivalence["alias_count"] + 1
        assert equivalence["literature_novelty_claimed"] is False


def test_tied_top10_pareto_and_comparison_classes_are_non_scalar(export: dict) -> None:
    assert export["comparison_classes"] == {
        "simplicity_complexity": STRUCTURAL_CLASS,
        "internal_exact_equivalence": EQUIVALENCE_CLASS,
    }
    simplicity = export["simplicity_top10"]
    alias = export["alias_multiplicity_top10"]
    assert len(simplicity["candidate_ids"]) == len(alias["candidate_ids"]) == 10
    assert simplicity["cutoff_tied_rank"] == 4
    assert simplicity["boundary_tie_total_count"] == 13
    assert alias["cutoff_tied_rank"] == 4
    assert alias["boundary_tie_total_count"] == 160
    assert export["simplicity_pareto_front"] == {
        "candidate_count": 2,
        "candidate_ids": [
            "G3A-2f8983c88f504150381064f2",
            "G3A-58e59412e5fe77cd54caf863",
        ],
        "registry_root_sha256": "5d7bfbe9a4751689fc99ffee363bcce989ac425101b15949b8bab863ae6e1614",
    }
    text = json.dumps(export, sort_keys=True).lower()
    assert "truth_score" not in text
    assert "overall_score" not in text
    assert "global_score" not in text


def test_blocked_formal_candidates_are_measured_without_validity_promotion(
    export: dict,
) -> None:
    blocked = [
        record
        for record in export["candidate_records"]
        if record["formal_context"]["decision"] == "blocked"
    ]
    assert len(blocked) == 160
    assert all(record["structural_evidence_status"] == "measured" for record in blocked)
    assert all(record["scientific_validity_inference"] is False for record in blocked)
    assert all(
        record["formal_context"]["used_for_structural_rank"] is False
        for record in blocked
    )
    assert all(record["simplicity_tied_rank"] >= 1 for record in blocked)


def test_validator_rejects_formula_alias_action_and_validity_tamper(export: dict) -> None:
    formula_tamper = copy.deepcopy(export)
    record = formula_tamper["candidate_records"][0]
    record["formula_structure"]["fields"].append("invented_field")
    _rehash_record(record)
    _rehash_export(formula_tamper)
    with pytest.raises(ValueError, match="rederive"):
        validate_scalable_candidate_structural_metrics_export(formula_tamper)

    alias_tamper = copy.deepcopy(export)
    record = alias_tamper["candidate_records"][0]
    record["equivalence_evidence"]["alias_count"] += 1
    _rehash_record(record)
    _rehash_export(alias_tamper)
    with pytest.raises(ValueError, match="equivalence"):
        validate_scalable_candidate_structural_metrics_export(alias_tamper)

    duplicate = copy.deepcopy(export)
    duplicate["candidate_records"][1]["action_sha256"] = duplicate["candidate_records"][0][
        "action_sha256"
    ]
    _rehash_record(duplicate["candidate_records"][1])
    _rehash_export(duplicate)
    with pytest.raises(ValueError, match="equivalence dedup"):
        validate_scalable_candidate_structural_metrics_export(duplicate)

    promoted = copy.deepcopy(export)
    record = next(
        item
        for item in promoted["candidate_records"]
        if item["formal_context"]["decision"] == "blocked"
    )
    record["scientific_validity_inference"] = True
    _rehash_record(record)
    _rehash_export(promoted)
    with pytest.raises(ValueError, match="validity"):
        validate_scalable_candidate_structural_metrics_export(promoted)

    top10_tamper = copy.deepcopy(export)
    top10_tamper["simplicity_top10"]["candidate_ids"].reverse()
    top10_tamper["content_sha256"] = _sha(
        {key: value for key, value in top10_tamper.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="top-10"):
        validate_scalable_candidate_structural_metrics_export(top10_tamper)


def test_binding_budget_and_seals_fail_closed() -> None:
    config = _load(CONFIG)
    config["source_export"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        build_scalable_candidate_structural_metrics_export(config, ROOT)

    config = _load(CONFIG)
    config["campaign_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file hash mismatch"):
        build_scalable_candidate_structural_metrics_export(config, ROOT)

    config = _load(CONFIG)
    config["budget"]["maximum_aliases"] = 94
    with pytest.raises(ValueError, match="budget"):
        build_scalable_candidate_structural_metrics_export(config, ROOT)

    config = _load(CONFIG)
    config["data_eligibility"]["observational_data_opened"] = True
    with pytest.raises(ValueError, match="eligibility"):
        build_scalable_candidate_structural_metrics_export(config, ROOT)


def test_committed_artifact_is_exact(export: dict) -> None:
    assert export == _load(ARTIFACT)
    validate_scalable_candidate_structural_metrics_export(export)
