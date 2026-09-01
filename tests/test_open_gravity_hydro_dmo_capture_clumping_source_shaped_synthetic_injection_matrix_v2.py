from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1 as v1,
)
from sigma_theory_compiler import (
    open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v2 as v2,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import BindingStatus

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def derived() -> tuple[dict, dict[str, bytes]]:
    return v2.derive_release()


def test_v2_config_binds_the_blocked_audit_and_every_v1_byte() -> None:
    config = v2.load_config()
    v2.validate_config(config)
    assert config["blocked_audit"] == {
        "path": "work/audits/open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v1-independent-audit-block-aca83d89.json",
        "raw_sha256": "763cebc3b9d3c4adf513f891fd8a49e23a55e4814b411d113ad6c8b9fac2f623",
        "status": "BLOCK",
    }
    predecessor = config["predecessor"]
    for prefix in ("config", "parameter_schema", "module", "test", "receipt"):
        assert (
            hashlib.sha256((ROOT / predecessor[f"{prefix}_path"]).read_bytes()).hexdigest()
            == predecessor[f"{prefix}_raw_sha256"]
        )
    for name, expected in predecessor["artifact_sha256"].items():
        path = (
            ROOT
            / "runs/gravity/open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v1"
            / name
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_repair_is_current_state_derived_and_only_numerical_resolution_changes() -> None:
    repair = v2.load_config()["repair"]
    assert repair["cm01_receiver_energy"] == (
        "current three-body Hamiltonian minus current visible-pair energy"
    )
    assert repair["cm01_state_dependency"] == (
        "all current positions, velocities, masses and pair-third interactions"
    )
    assert repair["cm01_integration_dt"] == 0.0004
    assert repair["two_body_integration_dt"] == 0.005
    assert repair["retain_changed_recovery_without_tuning"] is True


def test_formula_and_block_inventory_is_identical_to_v1(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    assert tuple(sorted(result["inherited"]["executable_formulas"])) == v1._EXECUTABLE
    assert (
        result["inherited"]["nonexecutable_formulas"] == v1.load_config()["nonexecutable_formulas"]
    )
    assert Counter(binding.status for binding in result["bindings"]) == {
        BindingStatus.EXECUTABLE: 6,
        BindingStatus.SOURCE_BLOCKED: 2,
        BindingStatus.UNADAPTED: 8,
    }


def test_full_cartesian_counts_and_replay_adjacency_are_unchanged(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    discovery = result["matrix"]
    assert (
        discovery.scenario_count,
        discovery.attempted_cell_count,
        discovery.scored_cell_count,
        len(discovery.ledger.entries),
    ) == (384, 6144, 2304, 8448)
    assert Counter(cell.eligibility for cell in discovery.cells) == {
        "ELIGIBLE": 2304,
        "SOURCE_BLOCKED": 768,
        "UNADAPTED": 3072,
    }
    ledger_hashes = {entry.entry_sha256 for entry in discovery.ledger.entries}
    assert all(cell.ledger_entry_sha256 in ledger_hashes for cell in discovery.cells)


def test_current_state_three_body_and_full_grid_energy_gates_pass(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    invariants = result["diagnostics"]["invariants_and_limits"]
    assert invariants["cm01_receiver_energy_uses_current_three_body_hamiltonian"] is True
    assert invariants["cm01_receiver_energy_is_initial_total_deficit"] is False
    assert invariants["all_full_grid_energy_gates_pass"] is True
    assert invariants["maximum_state_derived_total_energy_residual"] < 5e-6
    assert (
        invariants["state_derived_max_total_energy_residual_by_formula"][
            "CM01_CONSERVATIVE_THREE_BODY_CAPTURE"
        ]
        < 2e-7
    )
    assert (
        invariants["state_derived_max_total_energy_residual_by_formula"][
            "DC01_STATIC_FORCE_AMPLIFICATION_CONTROL"
        ]
        < 2e-7
    )


def test_receiver_entropy_limits_and_inactive_role_invariance_pass(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    invariants = result["diagnostics"]["invariants_and_limits"]
    assert invariants["maximum_receiver_temperature_entropy_identity_residual"] == 0.0
    assert invariants["maximum_zero_gamma_or_unit_force_limit_residual"] == 0.0
    assert invariants["maximum_inactive_hydro_dmo_role_feature_residual"] == 0.0
    assert invariants["future_cadence_or_response_accessed_by_rhs"] is False
    assert invariants["real_response_accessed"] is False


def test_recovery_changed_is_reported_without_tuning_or_failure_erasure(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    discovery = result["matrix"]
    diagnostics = result["diagnostics"]
    assert discovery.truth_recovery_count == 384
    assert discovery.distinct_truth_recovery_count == 382
    assert diagnostics["nonrecovered_scenarios"] == []
    assert len(diagnostics["truth_recovered_but_nondistinct_scenarios"]) == 2
    assert result["config"]["repair"]["retain_changed_recovery_without_tuning"] is True


def test_source_cadence_and_interval_censoring_are_unchanged(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    lengths = Counter()
    for scenario in result["scenarios"]:
        values = result["runtime"][scenario.scenario_id].formula_values
        times = values["source.vector.encounter-time"]
        lower = values["source.vector.cadence-interval-lower"]
        upper = values["source.vector.cadence-interval-upper"]
        assert np.all(lower <= times)
        assert np.all(times <= upper)
        lengths[len(times)] += 1
    assert lengths == {8: 192, 15: 192}


def test_direct_hidden_truth_and_noise_draw_contract_remain_response_blind(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    zero_record = next(row for row in result["records"] if row["noise_family"] == "zero-draw")
    scenario_id = zero_record["scenario"]["scenario_id"]
    runtime = result["runtime"][scenario_id]
    truth_id = result["truths"][scenario_id]
    direct = v2._truth_prediction(
        truth_id,
        runtime.formula_values,
        result["inherited"]["executable_formulas"][truth_id],
    )
    for prediction in v2._PREDICTIONS:
        assert np.array_equal(
            direct[prediction], runtime.response_values[v2._RESPONSES[prediction]]
        )
    for binding in result["bindings"]:
        assert all(
            not value.startswith(("response.", "truth.")) for value in binding.required_features
        )
        assert all(
            not value.startswith(("response.", "truth.")) for value in binding.optional_features
        )


def test_source_and_response_access_ceiling_remains_zero(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    result, _artifacts = derived
    assert all(value == 0 for value in result["config"]["access_contract"].values())
    assert result["diagnostics"]["cross_hydro_dmo_identifiability"] == {
        "synthetic_pair_roles_only": True,
        "public_object_level_matching_claimed": False,
        "inactive_role_field_prediction_residual": 0.0,
        "interpretation": (
            "The executable laws do not consume gas/cooling/shock/wake/relaxation controls, "
            "so paired role changes alone remain intentionally non-identifying."
        ),
    }


def test_same_inputs_replay_to_identical_bytes(
    derived: tuple[dict, dict[str, bytes]],
) -> None:
    first_result, first_artifacts = derived
    second_result, second_artifacts = v2.derive_release()
    assert first_result["matrix"].content_sha256 == second_result["matrix"].content_sha256
    assert (
        first_result["matrix"].ledger.content_sha256
        == second_result["matrix"].ledger.content_sha256
    )
    assert first_artifacts == second_artifacts


def test_frozen_receipt_and_artifacts_are_exact() -> None:
    receipt = v2.check()
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert v2._json_sha256(body) == receipt["content_sha256"]
    assert receipt["matrix_counts"] == {
        "attempted_cells": 6144,
        "scenarios": 384,
        "scored_cells": 2304,
        "source_blocked_cells": 768,
        "truth_distinctly_recovered": 382,
        "truth_recovered": 384,
        "unadapted_cells": 3072,
    }
    assert receipt["replay_entries"] == 8448
    for artifact in receipt["artifacts"]:
        payload = (ROOT / artifact["path"]).read_bytes()
        assert len(payload) == artifact["bytes"]
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
    assert json.loads((ROOT / v2.RECEIPT_PATH).read_text(encoding="utf-8")) == receipt
