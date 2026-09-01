from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_phangs_things_full3d_source_systematics_v1 as systematic,
)


@pytest.fixture(scope="module")
def config() -> dict:
    value = json.loads(Path(systematic.CONFIG_PATH).read_text(encoding="utf-8"))
    systematic.validate_config(value)
    return value


@pytest.fixture(scope="module")
def receipt(config: dict) -> dict:
    return systematic.build_receipt(config)


def test_full_source_grid_and_failure_policy_are_frozen(config: dict) -> None:
    cells = config["cell_contract"]
    assert cells["primary_cells_per_object"] == 72
    assert cells["total_cells_per_object"] == 75
    assert cells["total_cells"] == 225
    assert cells["response_based_selection"] is False
    assert cells["retain_all_failures"] is True
    assert config["gate_contract"]["no_family_pruning_from_one_failure"] is True
    assert "public S4G, THINGS, PHANGS" in config["admission_rule"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "PUBLICATION_READY"),
        (("objects",), ["NGC2903"]),
        (("cell_contract", "total_cells"), 3),
        (("cell_contract", "response_based_selection"), True),
        (("cell_contract", "retain_all_failures"), False),
        (("screen_contract", "nodes_per_axis"), 9),
        (("screen_contract", "operators"), ["NEWTON"]),
        (("gate_contract", "no_family_pruning_from_one_failure"), False),
        (("gate_contract", "primary_17_vs_bound_25_radial_relative_difference_max"), 1.0),
        (("scientific_boundary", "response_rows_opened"), 1),
        (("scientific_boundary", "scores_computed"), 1),
        (("claim_boundary", "observational_preference_established"), True),
    ],
)
def test_material_config_mutations_fail_closed(
    config: dict, path: tuple[str, ...], replacement: object
) -> None:
    mutated = copy.deepcopy(config)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(systematic.SourceSystematicsError):
        systematic.validate_config(mutated)


def test_cell_identity_program_has_exact_cartesian_coverage(config: dict) -> None:
    axes = config["cell_contract"]["primary_axes"]
    identities = {
        systematic._cell_id(beam, stellar, co, ratio, gas)
        for beam in axes["beam"]
        for stellar in axes["stellar_mass_to_light"]
        for co in axes["co_source"]
        for ratio in axes["stellar_height_over_exponential_scale"]
        for gas in axes["gas_height_pc"]
    }
    assert len(identities) == 72
    assert "ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS0.136986301369863:HG200" in identities


def test_receipt_retains_every_cell_and_numerical_counterexample(receipt: dict) -> None:
    assert receipt["cell_count"] == 225
    assert len(receipt["cells"]) == 225
    assert len({(row["object_id"], row["cell_id"]) for row in receipt["cells"]}) == 225
    assert receipt["numerical_pass_cell_count"] + receipt["numerical_counterexample_count"] == 225
    assert len(receipt["counterexamples"]) == receipt["numerical_counterexample_count"]
    for row in receipt["counterexamples"]:
        assert row["failed_gates"]
        cell = next(
            value
            for value in receipt["cells"]
            if value["object_id"] == row["object_id"] and value["cell_id"] == row["cell_id"]
        )
        assert cell["future_response_disposition"].startswith("RETAINED_NUMERICAL_COUNTEREXAMPLE")


def test_every_primary_object_has_72_cells_and_three_controls(receipt: dict) -> None:
    for object_id in ("NGC2903", "NGC3351", "NGC3627"):
        rows = [row for row in receipt["cells"] if row["object_id"] == object_id]
        assert sum(row["cell_kind"] == "PRIMARY_CARTESIAN" for row in rows) == 72
        assert sum(row["cell_kind"] == "NUMERICAL_SOURCE_CONTROL" for row in rows) == 3


def test_primary_cells_quantify_and_retain_bound_resolution_failures(receipt: dict) -> None:
    primary_id = "ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS0.136986301369863:HG200"
    rows = [row for row in receipt["cells"] if row["cell_id"] == primary_id]
    assert len(rows) == 3
    retained_failures = 0
    for row in rows:
        assert row["primary_vs_bound_bridge_relative"] is not None
        expected_pass = max(row["primary_vs_bound_bridge_relative"].values()) <= 0.2
        assert row["gates"]["primary_vs_bound_bridge"] is expected_pass
        if not expected_pass:
            retained_failures += 1
            assert row["all_numerical_gates_pass"] is False
            assert row["future_response_disposition"].startswith(
                "RETAINED_NUMERICAL_COUNTEREXAMPLE"
            )
    assert retained_failures >= 1


def test_all_passed_cells_have_finite_fields_and_exact_mass_accounting(receipt: dict) -> None:
    for row in receipt["cells"]:
        assert row["dimensionless_mass_relative_error"] < 1.0e-12
        assert row["source_builder_mass_relative_error"] < 2.0e-9
        assert len(set(row["field_hashes"].values())) == len(row["field_hashes"])
        if row["all_numerical_gates_pass"]:
            assert row["solver_metrics"]["aqual_converged"] is True
            assert row["solver_metrics"]["newton_relative_residual"] < 1.0e-12
            assert row["solver_metrics"]["qumond_relative_residual"] < 1.0e-12
            assert row["solver_metrics"]["aqual_relative_residual"] < 2.0e-7


def test_source_systematic_envelopes_are_complete_and_nontrivial(receipt: dict) -> None:
    envelopes = receipt["source_systematic_envelopes"]
    assert len(envelopes) == 27
    valid = [row for row in envelopes if row["status"] == "SOURCE_SYSTEMATIC_ENVELOPE"]
    assert valid
    assert all(row["valid_cell_count"] <= 72 for row in valid)
    assert all(row["maximum_to_minimum_ratio"] >= 1.0 for row in valid)
    assert any(row["maximum_to_minimum_ratio"] > 1.05 for row in valid)


def test_equivalence_ledger_accounts_for_every_cell(receipt: dict) -> None:
    ledger = receipt["equivalence_ledger"]
    assert sum(row["multiplicity"] for row in ledger) == 225
    assert receipt["equivalence_group_count"] == len(ledger)
    assert receipt["equivalence_ledger_root_sha256"] == systematic.content_sha256(ledger)
    assert receipt["cell_ledger_root_sha256"] == systematic.content_sha256(receipt["cells"])


def test_zero_response_and_narrow_claim_boundary(receipt: dict) -> None:
    boundary = receipt["scientific_boundary"]
    assert boundary["source_files_opened_per_build"] == 21
    assert boundary["response_files_opened"] == 0
    assert boundary["response_rows_opened"] == 0
    assert boundary["response_values_opened"] == 0
    assert boundary["scores_computed"] == 0
    claims = receipt["claim_boundary"]
    assert claims["all_source_systematics_propagated"] is True
    assert claims["response_fit_tested"] is False
    assert claims["publication_ready"] is False


def test_atomic_no_clobber_is_replay_safe(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert systematic._atomic_no_clobber(path, b"first") == "CREATED"
    assert systematic._atomic_no_clobber(path, b"first") == "EXISTING_IDENTICAL"
    with pytest.raises(systematic.SourceSystematicsError):
        systematic._atomic_no_clobber(path, b"second")


def test_package_hash_pins_match_after_seal() -> None:
    if systematic._MODULE_SEMANTIC_SHA256 == "0" * 64 or systematic._TEST_RAW_SHA256 == "0" * 64:
        pytest.skip("self pins are installed at final mutation seal")
    assert (
        systematic.module_semantic_sha256(systematic._repo_path(systematic.MODULE_PATH))
        == systematic._MODULE_SEMANTIC_SHA256
    )
    assert (
        systematic.file_sha256(systematic._repo_path(systematic.TEST_PATH))
        == systematic._TEST_RAW_SHA256
    )
