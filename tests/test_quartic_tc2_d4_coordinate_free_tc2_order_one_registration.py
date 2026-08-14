from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import quartic_tc2_d4_coordinate_free_tc2_order_one_registration as gate

ROOT = Path(__file__).resolve().parents[1]


def _artifact() -> dict[str, object]:
    return json.loads((ROOT / gate.OUTPUT_PATH).read_text(encoding="utf-8"))


def test_all_15_tc2_order_one_packets_replay() -> None:
    artifact = _artifact()
    gate.validate_campaign(artifact, ROOT)
    assert len(artifact["registered_coordinate_free_TC2_Taylor_order_one_packets"]) == 15
    assert artifact["counts"]["product_rule_nonzero_remainders"] == 0
    assert artifact["counts"]["candidate_coefficient_derivative_nonzero_terms"] == 0
    assert artifact["counts"]["manifest_registered_after"] == 109
    assert artifact["counts"]["manifest_missing_after"] == 195
    assert artifact["claims"]["global_H7_closed"] is False


def test_tampered_tc2_packet_fails_closed() -> None:
    tampered = copy.deepcopy(_artifact())
    packet = tampered["registered_coordinate_free_TC2_Taylor_order_one_packets"][1]
    packet["unit_TC2_Taylor_order_one_matrix"]["candidate_coefficient_derivative"] = "1"
    with pytest.raises(gate.CoordinateFreeTC2OrderOneRegistrationError, match="content hash"):
        gate.validate_campaign(tampered, ROOT)


def test_nonzero_candidate_parameter_derivative_is_rejected() -> None:
    config = json.loads((ROOT / gate.CONFIG_PATH).read_text(encoding="utf-8"))
    config["target"]["candidate_coefficient_derivative"] = "1"
    with pytest.raises(gate.CoordinateFreeTC2OrderOneRegistrationError, match="invalid"):
        gate._validate_config(config)
