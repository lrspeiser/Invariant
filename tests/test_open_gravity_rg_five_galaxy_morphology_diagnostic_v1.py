from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_rg_five_galaxy_morphology_diagnostic_v1 as diagnostic


@pytest.fixture(scope="module")
def config() -> dict:
    return diagnostic.load_config(verify_package=False)


@pytest.fixture(scope="module")
def receipt(config: dict) -> dict:
    return diagnostic.build_receipt(config)


def test_scope_is_post_hoc_not_confirmation(config: dict) -> None:
    assert config["status"] == "FROZEN_POST_HOC_HYPOTHESIS_GENERATOR"
    assert config["statistical_contract"]["new_formula_fit"] is False
    assert config["statistical_contract"]["threshold_tuning"] is False
    assert config["claim_boundary"]["matched_pair_is_confirmation"] is False
    assert config["claim_boundary"]["publication_ready"] is False


def test_bound_receipts_are_exact_and_zero_new_access(config: dict) -> None:
    source, score = diagnostic._load_bindings(config)
    assert source["claims"]["scientific_response_scored"] is False
    assert score["adjudication"]["family_eliminated"] is False
    assert set(config["access_scope"].values()) == {0}


def test_exact_permutation_contract() -> None:
    rho, p_value = diagnostic._exact_permutation_p([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert rho == pytest.approx(1.0)
    assert p_value == pytest.approx(2.0 / 120.0)


def test_feature_ledger_and_object_count(receipt: dict) -> None:
    assert len(receipt["object_feature_rows"]) == 5
    assert len(receipt["correlations"]) == 11
    for row in receipt["object_feature_rows"]:
        assert len(row["features"]) == 11


def test_no_feature_is_familywise_significant(receipt: dict) -> None:
    assert receipt["familywise_significant_features"] == []
    assert all(not row["familywise_significant"] for row in receipt["correlations"])
    assert (
        receipt["diagnostic_conclusions"]["single_source_feature_explains_fixed_rg_outcome"]
        is False
    )


def test_source_nearest_pair_has_opposite_response_direction(receipt: dict) -> None:
    pair = receipt["matched_pair"]
    assert pair["anchor_object"] == "NGC2976"
    assert pair["source_nearest_neighbor"] == "NGC4214"
    assert pair["opposite_support_direction"] is True
    assert (
        receipt["diagnostic_conclusions"]["simple_density_potential_compactness_is_sufficient"]
        is False
    )


def test_next_2d_builder_has_data_paper_benchmark_gate(config: dict) -> None:
    next_test = config["next_test_contract"]
    assert "THINGS" in next_test["required_new_data"]
    assert "Walter et al. 2008" in next_test["primary_paper"]
    assert len(next_test["independent_benchmarks"]) == 3
    assert next_test["missing_velocity_field_disposition"] == "SOURCE_BLOCKED"
    assert next_test["general_3d_claim_allowed"] is False


def test_receipt_is_deterministic(config: dict, receipt: dict) -> None:
    assert diagnostic.build_receipt(config) == receipt
    assert receipt["content_sha256"] == diagnostic.content_sha256({**receipt, "content_sha256": ""})


def test_config_mutations_fail_closed(config: dict) -> None:
    for path, value in (
        (("status",), "CONFIRMED"),
        (("statistical_contract", "new_formula_fit"), True),
        (("statistical_contract", "feature_count"), 1),
        (("matched_pair_contract", "anchor_object"), "NGC4214"),
        (("next_test_contract", "missing_velocity_field_disposition"), "READY"),
        (("claim_boundary", "publication_ready"), True),
    ):
        mutated = copy.deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(diagnostic.MorphologyDiagnosticError):
            diagnostic.validate_config(mutated)


def test_receipt_mutation_fails(config: dict, receipt: dict) -> None:
    mutated = copy.deepcopy(receipt)
    mutated["claim_boundary"]["new_theory_supported"] = True
    mutated["content_sha256"] = diagnostic.content_sha256({**mutated, "content_sha256": ""})
    with pytest.raises(diagnostic.MorphologyDiagnosticError):
        diagnostic.validate_receipt_payload(config, mutated)


def test_atomic_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert diagnostic._atomic_no_clobber(output, b"one\n") == "CREATED"
    assert diagnostic._atomic_no_clobber(output, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(diagnostic.MorphologyDiagnosticError):
        diagnostic._atomic_no_clobber(output, b"two\n")
