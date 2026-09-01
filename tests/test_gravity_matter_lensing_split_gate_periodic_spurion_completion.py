from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_split_gate_periodic_spurion_completion as periodic,
)


def test_config_identity_and_primary_paper_gate() -> None:
    config = periodic.load_config()
    assert config["artifact_id"] == periodic.ARTIFACT_ID
    assert config["admission_policy"]["source_class"] == (
        "PRIMARY_PAPERS_PLUS_EXACT_SYMBOLIC_AND_NUMERIC_BENCHMARKS"
    )


def test_eft_preflight_is_exactly_bound() -> None:
    binding = periodic.build_receipt()["eft_preflight_binding"]
    assert binding["config_sha256"] == (
        "e5a12688778fe3e7d14093af23ec01e3be331927a002f0040b29c3842ff77272"
    )
    assert binding["receipt_content_sha256"] == (
        "3cd01afa9078f7fdd26240d466bdf80a5eb0223a0cfc74cad347bd373175d5ec"
    )


def test_periodic_symmetry_and_small_field_recovery() -> None:
    checks = periodic._symbolic_checks()
    assert checks["S02_PERIODIC_AND_SHIFT_IDENTITIES"] is True
    assert checks["S03_SMALL_FIELD_POTENTIAL_RECOVERY"] is True
    assert checks["S04_SMALL_FIELD_MIXING_RECOVERY"] is True


def test_mass_curvature_and_range_are_background_phase_dependent() -> None:
    checks = periodic._symbolic_checks()
    assert checks["S05_MASS_CURVATURE_AND_RANGE"] is True
    contract = periodic.load_config()["principal_and_health_contract"]
    assert "cos(theta)>0" in contract["chi_curvature"]
    assert "sqrt(Z*cos(theta))" in contract["bare_range"]


def test_exact_health_and_source_phase_ceiling() -> None:
    checks = periodic._symbolic_checks()
    assert checks["S06_EXACT_HEALTH_PHASE_CEILING"] is True
    assert checks["S07_EXACT_SOURCE_CAPACITY"] is True
    source = periodic.load_config()["exact_source_capacity"]
    assert "sqrt(2*h-h^2)" in source["stable_branch_ceiling"]


def test_strong_gate_recovers_range_source_theorem() -> None:
    checks = periodic._symbolic_checks()
    assert checks["S08_STRONG_GATE_ASYMPTOTICS"] is True
    assert periodic.load_config()["adjudication"]["p2_range_source_theorem_recovered"] is True


def test_low_power_branch_is_changed_not_hidden() -> None:
    assert periodic._symbolic_checks()["S09_LOW_POWER_BRANCH_CHANGE"] is True
    source = periodic.load_config()["exact_source_capacity"]
    assert "1/2<s<1" in source["low_power_change"]
    assert "X^(s/2)" in source["low_power_change"]


def test_numeric_slopes_cover_low_borderline_and_strong_gate_cases() -> None:
    evidence = periodic._numeric_evidence(periodic.load_config())
    assert evidence["all_passed"] is True
    assert [record["s"] for record in evidence["slope_records"]] == [0.75, 1.0, 2.0, 4.0]
    low = evidence["slope_records"][0]
    high = evidence["slope_records"][-1]
    assert low["product_target"] == 0.375
    assert high["product_target"] == 0.5


def test_designed_phase_failures_are_retained() -> None:
    records = periodic._numeric_evidence(periodic.load_config())["phase_records"]
    assert len(records) == 4
    assert all(record["passed"] for record in records)
    by_id = {record["id"]: record for record in records}
    assert by_id["SMALL_PHASE_HEALTHY"]["K"] > 0.0
    assert by_id["HEALTH_BOUND_FAILURE"]["K"] < 0.0
    assert abs(by_id["MASS_BOUNDARY"]["mass_curvature"]) < 1.0e-14
    assert by_id["TACHYONIC_MAXIMUM"]["mass_curvature"] < 0.0


def test_primary_literature_is_exact() -> None:
    papers = periodic.load_config()["primary_literature"]
    assert {item["arxiv"] for item in papers} == {
        "hep-ph/9503331",
        "1611.08279",
        "2107.00010",
        "2604.20292v2",
    }


def test_periodicity_is_partial_remedy_not_full_eft_solution() -> None:
    adjudication = periodic.load_config()["adjudication"]
    assert adjudication["continuous_shift_spurion_structure_present"] is True
    assert adjudication["radiative_stability_proved"] is False
    assert adjudication["large_u_Z_resummation_solved"] is False
    assert adjudication["standalone_publication_candidate"] is False
    assert adjudication["supporting_section_for_range_source_note"] is True


def test_claim_ceiling_does_not_eliminate_or_overpromote() -> None:
    claims = periodic.load_config()["claim_boundary"]
    assert claims["fatal_no_go"] is False
    assert claims["strong_gate_tradeoff_recovery"] is True
    assert claims["radiative_stability"] is False
    assert claims["successful_gravity_model"] is False
    assert claims["publication_ready"] is False


def test_receipt_is_deterministic_self_hashed_and_zero_access() -> None:
    first = periodic.build_receipt()
    second = periodic.build_receipt()
    assert first == second
    assert first["content_sha256"] == periodic._self_hash(first)
    assert first["checks_passed"] == 12
    assert not any(first["access_ledger"].values())


def test_coherent_overclaim_forgery_differs_from_rebuild() -> None:
    forged = copy.deepcopy(periodic.build_receipt())
    forged["claim_boundary"]["large_u_resummation"] = True
    forged["content_sha256"] = periodic._self_hash(forged)
    assert forged["content_sha256"] == periodic._self_hash(forged)
    assert forged != periodic.build_receipt()


def test_config_mutation_rejects_before_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = periodic._repo_root()
    path = tmp_path / periodic.CONFIG_PATH
    path.parent.mkdir(parents=True)
    config = json.loads((root / periodic.CONFIG_PATH).read_text(encoding="utf-8"))
    config["claim_boundary"]["publication_ready"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(periodic, "EXPECTED_CONFIG_RAW_SHA256", "0" * 64)
    with pytest.raises(periodic.SplitGatePeriodicCompletionError, match="config semantics changed"):
        periodic.load_config(tmp_path)


def test_write_replay_and_check_are_no_clobber() -> None:
    assert periodic.write_receipt() == "EXISTING_IDENTICAL"
    assert periodic.validate_receipt() == periodic.build_receipt()
