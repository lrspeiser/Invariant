from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pytest

import sigma_theory_compiler.open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v1 as matrix
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import BindingStatus
from sigma_theory_compiler.sigma_core import SchemaViolation


@pytest.fixture(scope="module")
def release_payloads():
    return matrix.derive_release()


def test_config_and_all_vq_branches_are_explicit() -> None:
    config = matrix.load_config()
    matrix.validate_config(config)
    bindings = matrix._bindings(config)
    by_formula = {row.formula_id: row for row in bindings}
    assert set(by_formula) == set(matrix._EXECUTABLE) | matrix._BLOCKED
    assert len(by_formula) == 15
    assert {row.formula_id for row in bindings if row.status is BindingStatus.EXECUTABLE} == set(
        matrix._EXECUTABLE
    )
    assert {
        row.formula_id for row in bindings if row.status is BindingStatus.SOURCE_BLOCKED
    } == set(matrix._BLOCKED)
    assert {f"VQ{index:02d}" for index in range(11)} == {
        formula_id.split("_", 1)[0] for formula_id in by_formula if formula_id.startswith("VQ")
    }


def _features(distance: float = 100.0, length: float = 40.0) -> dict[str, np.ndarray]:
    values = {
        "source.scalar.delta-h-km-s-mpc": np.asarray([6.74], dtype=np.float64),
        "source.scalar.distance-modulus-mag": np.asarray([30.0], dtype=np.float64),
        "source.scalar.distance-modulus-uncertainty-mag": np.asarray([0.1], dtype=np.float64),
        "source.scalar.distance-mpc": np.asarray([distance], dtype=np.float64),
        "source.scalar.h-m-km-s-mpc": np.asarray([67.4], dtype=np.float64),
        "source.scalar.mask-neighborhood-fraction": np.asarray([1.0], dtype=np.float64),
        "source.scalar.maximum-chord-mpc": np.asarray([20.0], dtype=np.float64),
        "source.scalar.null-void-length-mpc": np.asarray([30.0], dtype=np.float64),
        "source.scalar.observer-endpoint-chord-mpc": np.asarray([5.0], dtype=np.float64),
        "source.scalar.target-endpoint-chord-mpc": np.asarray([10.0], dtype=np.float64),
        "source.scalar.void-fraction": np.asarray([length / distance], dtype=np.float64),
        "source.scalar.void-length-mpc": np.asarray([length], dtype=np.float64),
        "source.vector.direction-cartesian": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "source.vector.flow-shear-design": np.asarray(
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        ),
    }
    assert set(values) == set(matrix._FEATURES)
    return values


def test_exact_distance_path_endpoint_and_max_chord_laws() -> None:
    values = _features()
    c = 299792.458
    baseline = 67.4 * 100.0 / c
    expected = {
        matrix.standard_flrw_adapter: baseline,
        matrix.two_phase_void_adapter: (67.4 * 100.0 + 6.74 * 40.0) / c,
        matrix.observer_endpoint_adapter: (67.4 * 100.0 + 6.74 * 5.0) / c,
        matrix.target_endpoint_adapter: (67.4 * 100.0 + 6.74 * 10.0) / c,
        matrix.maximum_chord_adapter: (67.4 * 100.0 + 6.74 * 20.0) / c,
        matrix.bounded_fraction_null_adapter: (67.4 * 100.0 + 6.74 * 30.0) / c,
    }
    for adapter, target in expected.items():
        observed = float(adapter(values, {})[matrix._OUTPUT][0])
        assert observed == target

    nulled = {key: np.array(value, copy=True) for key, value in values.items()}
    nulled["source.scalar.delta-h-km-s-mpc"][:] = 0.0
    assert all(float(adapter(nulled, {})[matrix._OUTPUT][0]) == baseline for adapter in expected)


def test_exposure_above_distance_fails_closed() -> None:
    values = _features(distance=10.0, length=11.0)
    with pytest.raises(SchemaViolation, match="outside path"):
        matrix.two_phase_void_adapter(values, {})


def test_fraction_permutation_never_moves_absolute_lengths() -> None:
    rows: list[Mapping[str, object]] = []
    for index, (distance, fraction) in enumerate(
        ((10.0, 0.1), (20.0, 0.2), (30.0, 0.3), (40.0, 0.4)), start=1
    ):
        rows.append(
            {
                "identifier": index,
                "distance_path_mpc": distance,
                "planck": {"void_fraction": fraction},
            }
        )
    permuted = matrix._fraction_permutation(rows, "planck")
    assert permuted == {1: 0.2, 2: 0.3, 3: 0.4, 4: 0.1}
    rebuilt = [permuted[int(row["identifier"])] * float(row["distance_path_mpc"]) for row in rows]
    assert rebuilt == [2.0, 6.0, 12.0, 4.0]
    assert all(
        0.0 <= length <= float(row["distance_path_mpc"])
        for length, row in zip(rebuilt, rows, strict=True)
    )


def test_cf4_source_decoder_has_no_response_field() -> None:
    payload = bytearray(b" " * 157)
    payload[0:7] = b"     12"
    payload[8:14] = b"30.000"
    payload[15:20] = b"0.100"
    payload[21:26] = b" 10.0"
    payload[39:44] = b"XXXXX"
    payload[83:91] = b"  0.0000"
    payload[92:100] = b"  0.0000"
    decoded = matrix._parse_permitted_cf4_source(bytes(payload), 12)
    assert set(decoded) == {"identifier", "DMzp", "e_DMzp", "Dist", "RAdeg", "DEdeg"}
    assert decoded["identifier"] == 12


def test_release_is_complete_response_blind_and_geometry_valid(release_payloads) -> None:
    receipt, values, scenarios, ledger, confusion, diagnostics = release_payloads
    assert receipt["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL"
    assert receipt["scientific_claim"] == "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION"
    assert receipt["object_count"] == 8
    assert receipt["geometry_variant_count"] == 3
    assert receipt["noise_family_count"] == 5
    assert receipt["truth_formula_count"] == 6
    assert receipt["scenario_count"] == 720
    assert receipt["attempted_cell_count"] == 10800
    assert receipt["scored_cell_count"] == 4320
    assert receipt["replay_entry_count"] == 15120
    access = receipt["access_accounting"]
    assert access["real_response_values_decoded"] == 0
    assert access["cf4_measured_velocity_fields_decoded"] == 0
    assert access["cf4_published_peculiar_velocity_fields_decoded"] == 0
    assert access["validation_source_fields_decoded"] == 0
    assert access["confirmation_source_fields_decoded"] == 0
    assert access["pantheon_files_opened"] == 0
    assert receipt["geometry_gates"] == {
        "zero_le_l_void_le_distance": True,
        "zero_le_l_null_le_distance": True,
        "absolute_length_permutation_used": False,
    }
    assert len(scenarios.splitlines()) == 720
    assert len(values) > 0 and len(ledger) > 0 and len(confusion) > 0 and len(diagnostics) > 0


def test_every_block_is_retained_once_per_scenario(release_payloads) -> None:
    receipt, _, _, ledger_bytes, _, _ = release_payloads
    ledger = json.loads(ledger_bytes)
    blocked = [row for row in ledger["entries"] if row["status"] == "SOURCE_BLOCKED"]
    assert len(blocked) == receipt["scenario_count"] * 9
    assert {row["formula_id"] for row in blocked} == matrix._BLOCKED
    assert all(row["result_sha256"] is None for row in blocked)


def test_artifact_hashes_and_content_hash_are_exact(release_payloads) -> None:
    receipt, values, scenarios, ledger, confusion, diagnostics = release_payloads
    payloads = {
        "values.npz": values,
        "scenarios.jsonl": scenarios,
        "ledger.json": ledger,
        "confusion-matrix.json": confusion,
        "geometry-and-identifiability.json": diagnostics,
    }
    assert receipt["artifact_sha256"] == {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == matrix._json_sha256(body)
