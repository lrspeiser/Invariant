from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_open_r_euler_normalization_bridge import (
    DECISION,
    _sealed,
    _verify_packet,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_open_r_euler_normalization_bridge.json"
OUTPUT = ROOT / "runs/math/system10-cylindrical-open-r-euler-normalization-bridge"


@pytest.fixture(scope="module")
def built() -> tuple[dict, list[dict]]:
    return build_receipt(CONFIG, output_dir=OUTPUT, root=ROOT)


def test_all_twelve_normalization_bridges_close(built: tuple[dict, list[dict]]) -> None:
    receipt, packets = built
    assert receipt["decision"] == DECISION
    assert receipt["counts"]["normalization_bridges_closed"] == 12
    assert len(packets) == 12
    assert all(
        packet["claims"]["equation_origin_to_off_shell_normalization_closed"] for packet in packets
    )


def test_six_spatial_metric_weights_are_exact(built: tuple[dict, list[dict]]) -> None:
    _, packets = built
    mappings = packets[0]["spatial_metric_euler_normalizations"]
    assert [item["source_field_pair"] for item in mappings] == [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 2],
        [2, 3],
        [3, 3],
    ]
    assert [item["source_to_off_shell_coefficient"] for item in mappings] == [
        "1",
        "1/sqrt(2)",
        "1/sqrt(2)",
        "1",
        "1/sqrt(2)",
        "1",
    ]


def test_scalar_matter_and_divq_normalizations_are_explicit(
    built: tuple[dict, list[dict]],
) -> None:
    _, packets = built
    packet = packets[0]
    assert packet["gravity_scalar_euler_normalization"]["bridge"] == "E_phi_g=-AW_equation[10]"
    matter = packet["matter_euler_force_normalization"]
    assert matter["sector_euler_force_coefficients"] == ["1", "1", "1"]
    sectors = {item["sector"]: item for item in matter["sectors"]}
    assert "2*kappa" in sectors["barotropic_irrotational_fluid"]["dynamic_to_covariant_euler"]
    assert "nabla_mu(C_Maxwell)" in sectors["source_free_maxwell"]["dynamic_to_covariant_euler"]
    assert [item["bridge_coefficient"] for item in packet["divQ_normalization"]["mappings"]] == [
        "1",
        "1",
        "1",
        "1",
    ]


def test_every_packet_replays_and_binds_candidate_origins(built: tuple[dict, list[dict]]) -> None:
    _, packets = built
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for index, packet in enumerate(packets):
        _verify_packet(packet, config, index)
        assert len(packet["source_bindings"]["full_rhs_equation_origin_set_sha256"]) == 64


def test_successor_audit_has_sharp_divq_factorization_block(
    built: tuple[dict, list[dict]],
) -> None:
    receipt, _ = built
    missing = receipt["materialization"]["first_missing_primitive"]
    assert missing["status"] == "BLOCK_FOUR_DIVQ_TO_C_FACTORIZATION_ROWS_UNREGISTERED"
    assert missing["required_rows"] == 4
    assert missing["registered_rows"] == 0
    assert receipt["claims"]["constraint_propagation_closed"] is False


def test_all_normalization_mutations_reject(built: tuple[dict, list[dict]]) -> None:
    receipt, _ = built
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "drop_offdiagonal_sqrt2",
        "flip_gravity_scalar_sign",
        "omit_fluid_rescale",
        "identify_maxwell_reduced_with_action_euler",
        "drop_M2_scale",
    }
    assert all(control["rejected"] is True for control in controls.values())


def test_checked_outputs_replay_and_tamper_fails(built: tuple[dict, list[dict]]) -> None:
    receipt, packets = built
    assert receipt == json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert packets[0] == json.loads((OUTPUT / "candidate-00.json").read_text(encoding="utf-8"))
    tampered = copy.deepcopy(packets[0])
    tampered["gravity_scalar_euler_normalization"]["source_to_off_shell_coefficient"] = "1"
    assert _sealed(tampered) is False
