from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_solar_planetary_source_shaped_synthetic_injection_matrix_v1 as solar_matrix,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import BindingStatus
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import SyntheticReplayLedger
from sigma_theory_compiler.sigma_core import SchemaViolation

ROOT = Path(__file__).resolve().parents[1]


def test_config_is_source_only_hash_bound_and_response_blind() -> None:
    config = solar_matrix.load_config()
    solar_matrix.validate_config(config)
    access = config["access_contract"]
    assert access["de440_kernel_bytes_hash_read"] == 119_799_808
    assert access["de440_state_values_extracted"] == 0
    assert access["observational_response_files_opened"] == 0
    assert access["observational_response_rows_opened"] == 0
    assert access["observational_residual_values_opened"] == 0
    assert access["response_calibrated_parameters"] == 0
    assert access["network_calls"] == access["model_calls"] == access["paid_calls"] == 0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("targets",), ["EARTH"]),
        (("epochs_centuries_from_j2000",), [0.0] * 13),
        (("noise", "fractional_sigma"), 0.03),
        (("scoring", "pairwise_degeneracy_relative_rms_max"), 1e-3),
        (("access_contract", "observational_response_rows_opened"), 1),
    ],
)
def test_material_config_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(solar_matrix.load_config())
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(SchemaViolation):
        solar_matrix.validate_config(config, verify_upstreams=False)


def test_population_domains_phases_and_orbital_radii_are_explicit() -> None:
    config = solar_matrix.load_config()
    assert len(config["targets"]) == 8
    assert len(config["source_domains"]) == 4
    assert len(config["epochs_centuries_from_j2000"]) == 13
    assert len(config["refined_epochs_centuries_from_j2000"]) == 25
    np.testing.assert_array_equal(
        np.asarray(config["refined_epochs_centuries_from_j2000"])[::2],
        np.asarray(config["epochs_centuries_from_j2000"]),
    )
    for target in solar_matrix._TARGETS:
        values = solar_matrix._source_arrays(
            config,
            np.asarray(config["epochs_centuries_from_j2000"], dtype=np.float64),
            target,
        )
        assert values["source.matrix.body-position-au"].shape == (13, 47, 3)
        assert values["source.vector.target-heliocentric-radius-au"].shape == (13,)
        assert np.all(values["source.vector.target-heliocentric-radius-au"] > 0.0)


def test_common_abi_bindings_are_honest_and_block_incompatible_formulas() -> None:
    config = solar_matrix.load_config()
    bindings = solar_matrix._bindings(config)
    executable = [row for row in bindings if row.status is BindingStatus.EXECUTABLE]
    blocked = [row for row in bindings if row.status is not BindingStatus.EXECUTABLE]
    assert [row.formula_id for row in executable] == config["mechanisms"]
    assert len(executable) == 5
    assert len(blocked) == 7
    assert all(row.domains == ("solar-system",) for row in executable)
    assert all("no tunable parameters" not in row.approximation_ceiling for row in blocked)


def test_frozen_release_counts_invariance_and_claim_boundary() -> None:
    receipt = json.loads((ROOT / solar_matrix.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["status"] == "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_AUDIT"
    assert receipt["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL"
    assert receipt["scientific_claim"] == "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION"
    assert receipt["independent_audit_completed"] is False
    assert receipt["distinct_independent_audit_required"] is True
    assert receipt["target_count"] == 8
    assert receipt["epoch_count"] == 13
    assert receipt["mechanism_count"] == 5
    assert receipt["scenario_count"] == 8 * 5 * 3
    assert receipt["common_abi_execution_count"] == 8 * 5
    assert receipt["successful_common_abi_execution_count"] == 8 * 5
    assert receipt["candidate_comparison_count"] == 8 * 5 * 3 * 5
    assert receipt["blocked_ledger_entry_count"] == 8 * 5 * 3 * 7
    assert receipt["replay_entry_count"] == 8 * 5 * 3 * (7 + 2 * 5)
    assert receipt["access_accounting"]["orbit_fits_performed"] == 0
    assert receipt["access_accounting"]["observational_residual_values_opened"] == 0
    assert receipt["invariance_gates"]["pass"] is True
    assert receipt["invariance_gates"]["maximum_translation_error_m_s2"] < 1e-14
    assert receipt["invariance_gates"]["maximum_rotation_error_m_s2"] < 1e-14
    assert receipt["invariance_gates"]["maximum_nested_refinement_error_m_s2"] < 1e-14


def test_confusion_identifiability_and_every_failure_are_machine_recorded() -> None:
    confusion = json.loads((ROOT / solar_matrix.CONFUSION_PATH).read_text(encoding="utf-8"))
    diagnostics = json.loads((ROOT / solar_matrix.DIAGNOSTICS_PATH).read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / solar_matrix.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert confusion["scenario_count"] == 120
    assert confusion["candidate_comparison_count"] == 600
    assert confusion["numerical_failure_count"] == 0
    assert confusion["no_hand_ranking"] is True
    assert set(confusion["winner_membership_counts"]) == set(confusion["truth_formula_ids"])
    assert diagnostics["pair_count"] == 8 * 10
    assert len(diagnostics["translation_rotation"]) == 8 * 5
    assert len(diagnostics["nested_epoch_refinement"]) == 8 * 5
    assert len(diagnostics["planet_orbital_radius_ranges"]) == 8
    assert len(diagnostics["source_domain_moment_ranges"]) == 4
    assert receipt["numerical_failures"] == []
    assert len(receipt["adapter_blocks"]) == 7


def test_typed_scenarios_and_append_only_replay_ledger_are_complete() -> None:
    rows = [json.loads(line) for line in (ROOT / solar_matrix.SCENARIOS_PATH).read_text(encoding="utf-8").splitlines()]
    ledger_payload = json.loads((ROOT / solar_matrix.LEDGER_PATH).read_text(encoding="utf-8"))
    ledger = SyntheticReplayLedger(
        ledger_payload["ledger_id"],
        tuple(
            __import__(
                "sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1",
                fromlist=["ReplayEntry"],
            ).ReplayEntry(**{
                **entry,
                "status": __import__(
                    "sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1",
                    fromlist=["DiscoveryStatus"],
                ).DiscoveryStatus(entry["status"]),
                "reason_codes": tuple(entry["reason_codes"]),
                "observable_ids": tuple(entry["observable_ids"]),
            })
            for entry in ledger_payload["entries"]
        ),
        ledger_payload["schema_version"],
    )
    assert len(rows) == 120
    assert len(ledger.entries) == 2040
    assert all(row["scenario"]["schema_version"] == "open-gravity-synthetic-scenario-packet-1.0" for row in rows)
    assert all(row["scenario"]["domain"] == "solar-system" for row in rows)
    assert all(row["scenario"]["anchors"][0]["anchor_id"] == "anchor.de440-public-kernel" for row in rows)
    assert all(entry.claim_class == "SYNTHETIC_DIRECTIONAL_SIGNAL" for entry in ledger.entries)


def test_atomic_frozen_replay_is_byte_identical() -> None:
    receipt = solar_matrix.check()
    assert receipt["content_sha256"] == json.loads(
        (ROOT / solar_matrix.RECEIPT_PATH).read_text(encoding="utf-8")
    )["content_sha256"]

