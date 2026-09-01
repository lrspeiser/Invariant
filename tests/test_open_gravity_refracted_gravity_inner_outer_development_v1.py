from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_inner_outer_development_v1 as campaign,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return campaign.load_config()


@pytest.fixture(scope="module")
def receipt(config: dict) -> dict:
    return campaign.build_receipt(config)


def test_config_and_source_policy_are_frozen(config: dict) -> None:
    campaign.validate_config(config)
    assert config["builder_solver_admission"]["disposition"].startswith("DATA_AND_PAPER")
    assert len(config["real_source_and_paper_anchors"]) == 7


def test_predecessor_evidence_is_exact(config: dict) -> None:
    rows = campaign.validate_predecessors(config)
    assert set(rows) == {
        "PRIMARY_3D_OPERATOR_BENCHMARK",
        "FINE_AND_CONVERGENCE_3D_SOURCES",
        "FIXED_MEDIAN_NEGATIVE_DEVELOPMENT_CONTROL",
        "BLOCKED_NINE_CELL_SCAN_SOURCE_FIELD_CACHE_AND_COUNTEREVIDENCE",
    }
    assert (
        rows["BLOCKED_NINE_CELL_SCAN_SOURCE_FIELD_CACHE_AND_COUNTEREVIDENCE"][
            "phangs_object_scores"
        ]
        == []
    )


def test_one_numerical_cell_does_not_prune_eight(config: dict, receipt: dict) -> None:
    gate = receipt["source_admission"]
    assert gate["admitted"] == config["parameter_contract"]["source_admitted_cells"]
    assert gate["blocked"] == ["PRIOR_CORNER_E0.1_Q2_R-23"]
    blocked = next(
        row for row in gate["cells"] if row["parameter_id"] == "PRIOR_CORNER_E0.1_Q2_R-23"
    )
    assert blocked["maximum_fraction_over_gate"] > 0.25
    assert all(
        row["maximum_fraction_over_gate"] < 0.05
        for row in gate["cells"]
        if row["disposition"] == "SOURCE_ADMITTED_FOR_SCORING"
    )


def test_primary_result_retains_mixed_signal(receipt: dict) -> None:
    result = receipt["primary_adjudication"]
    assert result["best_candidate"] == "PRIOR_CORNER_E0.1_Q2_R-27"
    assert result["phangs_fractional_improvement"] == pytest.approx(0.2786733223573877)
    assert result["sparc_fractional_improvement"] == pytest.approx(-18.72235733511512)
    assert result["global_cross_tracer_development_signal"] is False
    assert result["checks"] == {
        "phangs_improvement_over_2_percent": True,
        "at_least_two_phangs_objects_support": True,
        "sparc_same_direction": False,
    }


def test_inner_signal_and_outer_failure_are_both_retained(receipt: dict) -> None:
    diagnostic = receipt["inner_outer_diagnostic"]
    assert diagnostic["radius_max_kpc"] == pytest.approx(5.675)
    best = next(
        row
        for row in diagnostic["candidate_improvements"]
        if row["candidate"] == "PRIOR_CORNER_E0.1_Q2_R-27"
    )
    assert best["phangs_inner_improvement"] == pytest.approx(0.8196550855569067)
    assert best["sparc_inner_improvement"] == pytest.approx(0.5385425947615915)
    assert best["sparc_outer_improvement"] == pytest.approx(-64.14169994348231)
    assert diagnostic["median_absolute_fractional_tracer_difference"] > 0.29


def test_outer_failure_is_not_simple_source_truncation(receipt: dict) -> None:
    coverage = receipt["ngc2903_source_coverage"]["total"]
    assert coverage["outside_5p675_fraction"] > 0.38
    assert coverage["inside_15_fraction"] > 0.94
    assert coverage["maximum_nonzero_radius_kpc"] > 28.0


def test_access_and_claim_ceilings_are_honest(receipt: dict) -> None:
    access = receipt["access_accounting"]
    assert access["registered_multiplicity"] == 9
    assert access["source_admitted_cells_scored"] == 8
    assert access["numerically_blocked_cells_scored"] == 0
    for key in (
        "confirmation_rows_opened",
        "independent_rows_opened",
        "group_rows_opened",
        "lensing_rows_opened",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        assert access[key] == 0
    assert receipt["claim_boundary"]["publication_ready"] is False


def test_config_mutations_fail_closed(config: dict) -> None:
    for path, value in (
        (("status",), "CONFIRMED"),
        (("parameter_contract", "multiplicity_charge"), 1),
        (("source_admission_gate", "per_radius_relative_difference_max"), 1.0),
        (("claim_boundary", "publication_ready"), True),
    ):
        mutated = copy.deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(campaign.InnerOuterDevelopmentError):
            campaign.validate_config(mutated)


def test_receipt_mutation_fails_closed(config: dict, receipt: dict, monkeypatch) -> None:
    monkeypatch.setattr(campaign, "build_receipt", lambda _: receipt)
    mutated = copy.deepcopy(receipt)
    mutated["claim_boundary"]["publication_ready"] = True
    with pytest.raises(campaign.InnerOuterDevelopmentError):
        campaign.validate_receipt_payload(config, mutated)


def test_atomic_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = campaign.canonical_bytes({"a": 1})
    assert campaign._atomic_no_clobber(path, payload) == "CREATED"
    assert campaign._atomic_no_clobber(path, payload) == "EXISTING_IDENTICAL"
    with pytest.raises(campaign.InnerOuterDevelopmentError):
        campaign._atomic_no_clobber(path, campaign.canonical_bytes({"a": 2}))


def test_stored_receipt_matches_rebuild(config: dict, receipt: dict) -> None:
    path = campaign._repo_path(campaign.OUTPUT_PATH)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored == receipt
