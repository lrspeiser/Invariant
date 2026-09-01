from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1 as matrix,
)
from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import DataRole
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import BindingStatus
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    decide_scenario_eligibility,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def derived() -> tuple[dict, dict[str, bytes]]:
    return matrix.derive_release()


def test_config_is_synthetic_only_and_preserves_every_bound_predecessor_byte() -> None:
    config = matrix.load_config()
    matrix.validate_config(config)
    assert config["status"] == "FROZEN_SYNTHETIC_ONLY_PRE_RESPONSE"
    assert config["empirical_authority"] == "NONE"
    assert all(value == 0 for value in config["access_contract"].values())
    for row in config["upstream_bindings"]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]


def test_source_shape_is_exactly_tng_eight_and_camels_fifteen_without_payload_decode() -> None:
    config = matrix.load_config()
    assert len(config["source_families"]["tng100"]["cadence_age_gyr"]) == 8
    assert len(config["source_families"]["camels"]["cadence_scale_factors"]) == 15
    assert "object-level cross-tree match" in config["source_families"]["camels"]["role"]
    assert config["access_contract"]["simulation_payload_structures_opened"] == 0
    assert config["access_contract"]["simulation_scientific_rows_opened"] == 0
    assert config["access_contract"]["real_dmo_group_rows_opened"] == 0


def test_formula_inventory_has_six_executable_and_ten_honest_nonexecutables(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    statuses = Counter(binding.status for binding in result["bindings"])
    assert statuses == {
        BindingStatus.EXECUTABLE: 6,
        BindingStatus.SOURCE_BLOCKED: 2,
        BindingStatus.UNADAPTED: 8,
    }
    assert set(result["config"]["nonexecutable_formulas"]) >= {
        "DC02_INELASTIC_CLOUD_SHOCK",
        "DC03_CHANDRASEKHAR_WAKE_TRANSFER",
        "DC04_GRAVITATIONAL_WAVE_CAPTURE",
        "ORDINARY_GAS_SHOCK_COOLING",
        "ESTABLISHED_DYNAMICAL_FRICTION",
        "DISSIPATIVE_DARK_MATTER",
        "COLLISIONLESS_VIOLENT_RELAXATION",
    }


def test_response_and_truth_are_invisible_to_every_formula(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    catalogue = result["catalogue"]
    experiment = result["config"]["experiment_id"]
    visible = catalogue.visible_features(experiment)
    assert all(not value.startswith(("response.", "truth.")) for value in visible)
    for binding in result["bindings"]:
        assert all(
            not value.startswith(("response.", "truth.")) for value in binding.required_features
        )
        assert all(
            not value.startswith(("response.", "truth.")) for value in binding.optional_features
        )
    for element in catalogue.elements:
        if element.element_id.startswith("response."):
            assert element.role_for(experiment) is DataRole.SCORING_ONLY_RESPONSE
        if element.element_id.startswith("truth."):
            assert element.role_for(experiment) is DataRole.LATENT_SYNTHETIC_TRUTH


def test_all_scenarios_are_eligible_only_for_the_six_active_adapters(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    scenario = result["scenarios"][0]
    decisions = {
        binding.formula_id: decide_scenario_eligibility(
            binding, result["catalogue"], scenario
        ).status.value
        for binding in result["bindings"]
    }
    assert Counter(decisions.values()) == {"ELIGIBLE": 6, "SOURCE_BLOCKED": 2, "UNADAPTED": 8}


def test_full_source_truth_noise_and_candidate_cartesian_counts(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    discovery = result["matrix"]
    assert discovery.scenario_count == 384
    assert discovery.attempted_cell_count == 6144
    assert discovery.scored_cell_count == 2304
    assert len(discovery.ledger.entries) == 8448
    assert Counter(cell.eligibility for cell in discovery.cells) == {
        "ELIGIBLE": 2304,
        "SOURCE_BLOCKED": 768,
        "UNADAPTED": 3072,
    }


def test_every_cell_has_exact_truth_parameter_scenario_and_replay_lineage(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    discovery = result["matrix"]
    ledger_hashes = {entry.entry_sha256 for entry in discovery.ledger.entries}
    assert all(cell.ledger_entry_sha256 in ledger_hashes for cell in discovery.cells)
    assert all(cell.truth_formula_id in matrix._EXECUTABLE for cell in discovery.cells)
    assert all(
        (cell.parameter_cell_id is not None) == (cell.eligibility == "ELIGIBLE")
        for cell in discovery.cells
    )


def test_hidden_truth_is_direct_adapter_output_before_one_frozen_noise_draw(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    zero_record = next(row for row in result["records"] if row["noise_family"] == "zero-draw")
    scenario_id = zero_record["scenario"]["scenario_id"]
    runtime = result["runtime"][scenario_id]
    truth_id = result["truths"][scenario_id]
    direct = matrix._truth_prediction(
        truth_id,
        runtime.formula_values,
        result["config"]["executable_formulas"][truth_id],
    )
    for prediction in matrix._PREDICTIONS:
        assert np.array_equal(
            direct[prediction], runtime.response_values[matrix._RESPONSES[prediction]]
        )
    assert zero_record["noise_draws_per_response_vector"] == 0
    noisy_record = next(
        row for row in result["records"] if row["noise_family"] == "analytic-diagonal"
    )
    assert noisy_record["noise_draws_per_response_vector"] == 1


def test_public_cadence_becomes_bracketed_interval_censored_encounter_time(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    lengths = Counter()
    for scenario in result["scenarios"]:
        runtime = result["runtime"][scenario.scenario_id]
        values = runtime.formula_values
        times = values["source.vector.encounter-time"]
        lower = values["source.vector.cadence-interval-lower"]
        upper = values["source.vector.cadence-interval-upper"]
        assert np.all(lower <= times)
        assert np.all(times <= upper)
        assert np.all(np.diff(times) > 0)
        lengths[len(times)] += 1
    assert lengths == {8: 192, 15: 192}


def test_state_derived_energy_limits_and_inactive_role_invariance_pass(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    invariants = result["diagnostics"]["invariants_and_limits"]
    assert max(invariants["state_derived_max_total_energy_residual_by_formula"].values()) < 5e-6
    assert invariants["maximum_zero_gamma_or_unit_force_limit_residual"] == 0.0
    assert invariants["maximum_inactive_hydro_dmo_role_feature_residual"] == 0.0
    assert invariants["future_cadence_or_response_accessed_by_rhs"] is False
    assert invariants["real_response_accessed"] is False


def test_cross_hydro_dmo_packet_is_explicitly_nonidentifying_without_public_matching(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    cross = result["diagnostics"]["cross_hydro_dmo_identifiability"]
    assert cross["synthetic_pair_roles_only"] is True
    assert cross["public_object_level_matching_claimed"] is False
    assert cross["inactive_role_field_prediction_residual"] == 0.0


def test_truth_recovery_is_computed_not_hand_ranked_and_failures_are_retained(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    discovery = result["matrix"]
    assert discovery.truth_recovery_count == 383
    assert discovery.distinct_truth_recovery_count == 377
    assert result["config"]["scoring"]["no_hand_ranking"] is True
    assert (
        sum(cell.discovery_status in {"SOURCE_BLOCKED", "UNADAPTED"} for cell in discovery.cells)
        == 3840
    )


def test_same_inputs_replay_to_identical_matrix_and_artifact_bytes(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    first_result, first_artifacts = derived
    second_result, second_artifacts = matrix.derive_release()
    assert second_result["matrix"].content_sha256 == first_result["matrix"].content_sha256
    assert (
        second_result["matrix"].ledger.content_sha256
        == first_result["matrix"].ledger.content_sha256
    )
    assert second_artifacts == first_artifacts


def test_frozen_receipt_and_every_artifact_are_self_consistent() -> None:
    receipt = matrix.check()
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert matrix._json_sha256(body) == receipt["content_sha256"]
    assert receipt["matrix_counts"] == {
        "attempted_cells": 6144,
        "replay_entries": 8448,
        "retained_nonexecutable_cells": 3840,
        "scenarios": 384,
        "scored_cells": 2304,
        "truth_distinctly_recovered": 377,
        "truth_recovered": 383,
    }
    for artifact in receipt["artifacts"]:
        payload = (ROOT / artifact["path"]).read_bytes()
        assert len(payload) == artifact["bytes"]
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
    raw_receipt = json.loads((ROOT / matrix.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert raw_receipt == receipt
