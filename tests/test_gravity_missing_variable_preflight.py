from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_missing_variable_preflight as preflight

ROOT = Path(__file__).resolve().parents[1]


def test_registry_is_complete_dimensionless_and_target_blind() -> None:
    receipt = preflight.build_receipt(ROOT)
    assert receipt["decision"] == (
        "FOUR_DEFINED_XCOP_PROXIES_EXECUTABLE_TWO_CONTINUOUS_VARIABLES_"
        "SOURCE_DEFINITION_BLOCKED_ZERO_MEASUREMENTS_SIXTEEN_APPLICABLE_"
        "ROWS_SOURCE_BLOCKED_NO_SCORE"
    )
    assert receipt["counts"] == {
        "bound_source_receipts_or_artifacts": 5,
        "public_source_requirements": 11,
        "variable_families": 7,
        "lane_applicability_rows": 21,
        "executable_proxy_only_rows": 4,
        "source_blocked_applicable_rows": 16,
        "not_applicable_rows": 1,
        "continuous_measurement_ready_rows": 0,
        "defined_proxy_contracts": 4,
        "source_definition_blocked_variables": 2,
        "continuous_definition_frozen_source_data_blocked_variables": 5,
        "predecessor_public_predictor_rows_read": 8,
        "new_predictor_source_payload_rows_opened": 0,
        "response_or_target_rows_opened": 0,
        "scientific_scores_computed": 0,
    }
    assert tuple(row["variable_id"] for row in receipt["variable_registry"]) == (
        preflight.VARIABLE_IDS
    )
    for row in receipt["variable_registry"]:
        assert row["dimensionless_predictor"]["units"] == "1"
        assert tuple(item["lane_id"] for item in row["domain_applicability"]) == (
            preflight.LANE_IDS
        )
        assert row["prospective_test"]["run_status"] == "NOT_RUN_NO_RESPONSE_ACCESS"
        assert row["prospective_test"]["falsification_rule"]
        assert row["prospective_test"]["ablation_rule"]
        assert row["physical_provenance"]
        assert row["covariance_nuisance_treatment"]
        assert row["missingness_rule"]
        assert row["forbidden_response_derived_substitutes"]


def test_proxy_rows_are_separate_from_measurements_and_source_blocks() -> None:
    receipt = preflight.build_receipt(ROOT)
    summary = receipt["registry_summary"]
    assert summary["continuous_measurement_ready"] == []
    assert summary["executable_proxy_only"] == [
        {
            "variable_id": "geometry_3d",
            "lane_id": "XCOP_EIGHT_EXPOSED_DEVELOPMENT",
        },
        {
            "variable_id": "nonthermal_pressure",
            "lane_id": "XCOP_EIGHT_EXPOSED_DEVELOPMENT",
        },
        {
            "variable_id": "boundary_conditions",
            "lane_id": "XCOP_EIGHT_EXPOSED_DEVELOPMENT",
        },
        {
            "variable_id": "assembly_history",
            "lane_id": "XCOP_EIGHT_EXPOSED_DEVELOPMENT",
        },
    ]
    assert len(summary["source_blocked"]) == 16
    assert summary["not_applicable"] == [
        {"variable_id": "clumping", "lane_id": "SPARC_DEVELOPMENT_ONLY"}
    ]
    assert summary["source_definition_blocked_variables"] == [
        "nonthermal_pressure",
        "calibration",
    ]
    assert len(summary["continuous_definition_frozen_source_data_blocked_variables"]) == 5
    assert receipt["claim_boundary"]["defined_proxy_contracts_frozen"] is True
    assert receipt["claim_boundary"]["all_variable_predictor_contracts_frozen"] is False
    assert receipt["claim_boundary"]["source_definition_blockers_present"] is True
    assert receipt["claim_boundary"]["proxy_executable_is_measurement_ready"] is False
    assert receipt["claim_boundary"]["continuous_missing_variables_measured"] is False
    assert receipt["claim_boundary"]["scientific_scoring_executed"] is False
    assert receipt["claim_boundary"]["cause_identified"] is False
    assert receipt["claim_boundary"]["scientific_claim_allowed"] is False


def test_nonthermal_predictor_cannot_leak_scored_temperature_or_pressure() -> None:
    receipt = preflight.build_receipt(ROOT)
    by_id = {row["variable_id"]: row for row in receipt["variable_registry"]}
    nonthermal = by_id["nonthermal_pressure"]
    assert nonthermal["definition_status"] == (
        "SOURCE_DEFINITION_BLOCKED_DISJOINT_THERMAL_REFERENCE_NOT_SELECTED"
    )
    xcop = nonthermal["dimensionless_predictor"]["lane_specific_conventions"][
        "XCOP_EIGHT_EXPOSED_DEVELOPMENT"
    ]
    assert "disjoint from scored T_X and P_SZ" in xcop
    assert "separately hash-frozen predictor source" in xcop
    assert {
        "scored X-COP T_X",
        "scored X-COP P_SZ",
        "thermal pressure derived from any scored response",
    } <= set(nonthermal["forbidden_response_derived_substitutes"])
    catalog = {row["source_id"]: row for row in receipt["public_source_catalog"]}
    source = catalog["XCOP_GAS_MOTION_NONTHERMAL"]
    assert source["current_state"] == ("SOURCE_DEFINITION_BLOCKED_DISJOINT_PRODUCT_NOT_SELECTED")
    assert "independent_predictor_temperature_if_needed" in source["required_fields"]
    assert "provenance_disjoint_from_scored_T_X_and_P_SZ" in source["required_fields"]


def test_reference_conventions_are_explicit_or_definition_blocked() -> None:
    receipt = preflight.build_receipt(ROOT)
    by_id = {row["variable_id"]: row for row in receipt["variable_registry"]}
    calibration = by_id["calibration"]
    assert calibration["definition_status"] == (
        "SOURCE_DEFINITION_BLOCKED_EXACT_RELEASE_REFERENCES_NOT_SELECTED"
    )
    conventions = calibration["dimensionless_predictor"]["lane_specific_conventions"]
    assert set(conventions) == set(preflight.LANE_IDS)
    assert any("S_k=D" in item for item in conventions["SPARC_DEVELOPMENT_ONLY"])
    assert any("S_k=A_X(E)" in item for item in conventions["XCOP_EIGHT_EXPOSED_DEVELOPMENT"])
    assert any("S_k=A_inst(E)" in item for item in conventions["GROUP_SOURCE_AUDIT_LANES"])

    boundary = by_id["boundary_conditions"]["dimensionless_predictor"]
    assert boundary["definition"] == (
        "b_P=G*P_out/a0^2 for hydrostatic lanes; b_R=R_last/Rbar90 for the rotation lane"
    )
    assert "unscored outer SZ boundary" in boundary["reference_convention"]["hydrostatic_P_out"]
    assert boundary["reference_convention"]["hydrostatic_P_ref"].startswith("a0^2/G")
    assert (
        "Vobs availability cannot set R_last" in boundary["reference_convention"]["rotation_R_last"]
    )

    environment = by_id["environment"]["dimensionless_predictor"]
    assert environment["definition"] == ("max(||g_ext||_2/a0, ||T_ext^TF||_F*Rbar50/a0)")
    assert "sqrt(sum_ij(T_ij^2))" in environment["reference_convention"]
    assert "Rbar50 encloses 50 percent" in environment["reference_convention"]

    assembly = by_id["assembly_history"]["dimensionless_predictor"]
    assert "mu>=1/4" in assembly["event_convention"]
    assert "pericenter<=2*Rbar90" in assembly["event_convention"]
    assert assembly["dynamical_time_convention"].startswith(
        "t_dyn(Rbar50)=sqrt(Rbar50^3/(G*Mbar(<Rbar50)))"
    )


def test_leakage_access_and_next_acquisition_are_explicit() -> None:
    receipt = preflight.build_receipt(ROOT)
    lane = receipt["lane_contract"]
    assert lane["object_identity_predictor_forbidden"] is True
    assert lane["domain_identity_predictor_forbidden"] is True
    assert lane["lane_ids_are_metadata_not_predictors"] is True
    assert lane["post_response_variable_selection_forbidden"] is True
    assert lane["response_derived_normalization_forbidden"] is True
    access = receipt["chronology_and_access"]
    assert access["predecessor_public_predictor_rows_read"] == 8
    assert access["new_predictor_source_payload_rows_opened"] == 0
    assert access["response_rows_loaded"] == 0
    assert access["confirmation_rows_loaded"] == 0
    assert access["holdout_rows_loaded"] == 0
    assert access["independent_rows_loaded"] == 0
    assert access["scientific_scores_computed"] == 0
    assert receipt["next_actionable_acquisition"]["source_id"] == (
        "GROUP_ALIAS_DIRECT_ENDPOINT_PACKET"
    )
    assert "does not by itself unlock a score" in receipt["next_actionable_acquisition"]["unlocks"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["source_bindings"][0].__setitem__("file_sha256", "0" * 64),
        lambda value: value["lane_contract"].__setitem__(
            "domain_identity_predictor_forbidden", False
        ),
        lambda value: value["public_source_catalog"][0].__setitem__("current_state", "READY"),
        lambda value: value["variable_registry"][0]["dimensionless_predictor"].__setitem__(
            "units", "kpc"
        ),
        lambda value: value["variable_registry"][1].__setitem__(
            "definition_status", "CONTINUOUS_DEFINITION_FROZEN"
        ),
        lambda value: value["variable_registry"][1]["domain_applicability"][1].__setitem__(
            "current_execution_status", "MEASUREMENT_READY"
        ),
        lambda value: value["variable_registry"][2]["prospective_test"].__setitem__(
            "run_status", "PASSED"
        ),
        lambda value: value["chronology_and_access_contract"].__setitem__(
            "response_rows_loaded", 1
        ),
        lambda value: value["claim_boundary"].__setitem__("scientific_claim_allowed", True),
        lambda value: value["next_actionable_acquisition"].__setitem__(
            "unlocks", "scientific score"
        ),
    ],
)
def test_every_nested_contract_class_is_hash_sealed(mutation: object) -> None:
    config = copy.deepcopy(preflight.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(preflight.GravityMissingVariablePreflightError, match="config"):
        preflight.validate_config(config)


def test_source_hash_and_semantic_mutations_fail_closed() -> None:
    config = preflight.load_config(ROOT)
    bindings = copy.deepcopy(config["source_bindings"])
    bindings[0]["file_sha256"] = "0" * 64
    with pytest.raises(preflight.GravityMissingVariablePreflightError, match="source file"):
        preflight._load_sources(ROOT, bindings)

    sources = preflight._load_sources(ROOT, config["source_bindings"])
    changed = copy.deepcopy(sources)
    changed["group_scale_source_audit"]["counts"]["ready_lanes"] = 1
    with pytest.raises(preflight.GravityMissingVariablePreflightError, match="group source"):
        preflight._validate_source_semantics(changed)


def test_atomic_no_clobber_preserves_existing_and_race_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = Path("sealed/receipt.json")
    payload = b'{"sealed":true}\n'
    path, disposition = preflight._atomic_no_clobber(tmp_path, relative, payload)
    assert disposition == "CREATED"
    assert path.read_bytes() == payload
    stat_before = path.stat()
    _, disposition = preflight._atomic_no_clobber(tmp_path, relative, payload)
    assert disposition == "EXISTING_IDENTICAL"
    assert path.stat().st_mtime_ns == stat_before.st_mtime_ns
    with pytest.raises(preflight.GravityMissingVariablePreflightError, match="overwrite"):
        preflight._atomic_no_clobber(tmp_path, relative, b"different\n")
    assert path.read_bytes() == payload

    race_relative = Path("sealed/race.json")
    race_path = tmp_path / race_relative
    real_link = os.link

    def concurrent_creator(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        Path(target).write_bytes(b"race-winner\n")
        real_link(source, target)

    monkeypatch.setattr(preflight.os, "link", concurrent_creator)
    with pytest.raises(preflight.GravityMissingVariablePreflightError, match="overwrite"):
        preflight._atomic_no_clobber(tmp_path, race_relative, payload)
    assert race_path.read_bytes() == b"race-winner\n"


def test_stored_receipt_rebuilds_exactly_and_fails_on_claim_tamper() -> None:
    stored = json.loads((ROOT / preflight.OUTPUT_PATH).read_text(encoding="utf-8"))
    preflight.validate_receipt(stored, ROOT)
    assert stored == preflight.build_receipt(ROOT)
    assert stored["implementation_binding"]["test_path"] == preflight.TEST_PATH.as_posix()
    assert stored["implementation_binding"]["test_file_sha256"] == preflight._file_sha(
        ROOT / preflight.TEST_PATH
    )
    changed = copy.deepcopy(stored)
    changed["claim_boundary"]["cause_identified"] = True
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = preflight._sha(body)
    with pytest.raises(preflight.GravityMissingVariablePreflightError, match="receipt"):
        preflight.validate_receipt(changed, ROOT)
