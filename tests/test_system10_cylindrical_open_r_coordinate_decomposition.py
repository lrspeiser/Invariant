from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_open_r_coordinate_decomposition import (
    DECISION,
    _sealed,
    _verify_packet,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_open_r_coordinate_decomposition.json"
OUTPUT = ROOT / "runs/math/system10-cylindrical-open-r-coordinate-decomposition"


@pytest.fixture(scope="module")
def built() -> tuple[dict, list[dict]]:
    return build_receipt(CONFIG, output_dir=OUTPUT, root=ROOT)


def test_all_48_coordinate_decompositions_close(built: tuple[dict, list[dict]]) -> None:
    receipt, packets = built
    assert receipt["decision"] == DECISION
    assert receipt["counts"]["coordinate_decomposition_rows_closed"] == 48
    assert receipt["counts"]["symbolic_coordinate_zero_residuals"] == 48
    assert len(packets) == 12
    assert all(len(packet["rows"]) == 4 for packet in packets)


def test_exact_cylindrical_terms_and_time_coefficients(built: tuple[dict, list[dict]]) -> None:
    _, packets = built
    rows = packets[0]["rows"]
    assert [row["constraint_time_coefficient"] for row in rows] == ["-1", "1", "r**2", "1"]
    assert "3*r*S12" in rows[2]["covariant_divergence_expression"]
    assert "C1(t, r, theta, z)/r" in rows[0]["covariant_divergence_expression"]
    assert [row["lower_nu"] for row in rows] == [0, 1, 2, 3]


def test_every_atomic_packet_replays(built: tuple[dict, list[dict]]) -> None:
    _, packets = built
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for index, packet in enumerate(packets):
        _verify_packet(packet, config, index)


def test_successor_audit_binds_all_closed_inputs(built: tuple[dict, list[dict]]) -> None:
    receipt, _ = built
    counts = receipt["counts"]
    assert counts["physical_constraint_rows_bound"] == 96
    assert counts["full_rhs_rows_bound"] == 1020
    assert counts["equation_origin_seals_bound"] == 1020
    assert counts["all_first_spatial_rhs_jets_bound"] == 396


def test_sharp_normalization_bridge_block_has_acceptance_contract(
    built: tuple[dict, list[dict]],
) -> None:
    receipt, _ = built
    missing = receipt["materialization"]["first_missing_primitive"]
    assert missing["status"] == "BLOCK_NORMALIZATION_AND_FORCE_ORIGIN_MAP_UNREGISTERED"
    assert missing["required_candidate_instances"] == 12
    assert len(missing["required_mappings_per_candidate"]["six_spatial_metric_components"]) == 6
    assert "factor" in missing["acceptance"]
    assert receipt["claims"]["constraint_propagation_closed"] is False


def test_geometry_and_normalization_negative_controls_reject(
    built: tuple[dict, list[dict]],
) -> None:
    receipt, _ = built
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "flip_angular_metric_factor",
        "omit_cylindrical_connection",
        "infer_origin_normalization",
    }
    assert all(control["rejected"] is True for control in controls.values())


def test_checked_outputs_replay_and_tamper_fails(built: tuple[dict, list[dict]]) -> None:
    receipt, packets = built
    assert receipt == json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert packets[0] == json.loads((OUTPUT / "candidate-00.json").read_text(encoding="utf-8"))
    tampered = copy.deepcopy(packets[0])
    tampered["rows"][2]["constraint_time_coefficient"] = "1"
    assert _sealed(tampered) is False
