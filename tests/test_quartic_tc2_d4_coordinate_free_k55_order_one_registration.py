from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import quartic_tc2_d4_coordinate_free_k55_order_one_registration as gate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    return json.loads((ROOT / gate.OUTPUT_PATH).read_text(encoding="utf-8"))


def test_all_15_packets_replay_and_match_artifact(artifact: dict[str, object]) -> None:
    gate.validate_campaign(artifact, ROOT)
    assert len(artifact["registered_coordinate_free_K55_Taylor_order_one_packets"]) == 15
    assert artifact["counts"]["differentiated_identity_nonzero_remainders"] == 0
    assert artifact["counts"]["manifest_registered_after"] == 94
    assert artifact["counts"]["manifest_missing_after"] == 210
    assert artifact["claims"]["global_H7_closed"] is False


def test_tampered_packet_fails_closed(artifact: dict[str, object]) -> None:
    tampered = copy.deepcopy(artifact)
    packet = tampered["registered_coordinate_free_K55_Taylor_order_one_packets"][0]
    packet["K55_Taylor_order_one_matrix"]["entries"][0]["terms"][0]["coefficient"] = "999"
    with pytest.raises(gate.CoordinateFreeK55OrderOneRegistrationError, match="content hash"):
        gate.validate_campaign(tampered, ROOT)


def test_broad_claim_tamper_is_rejected() -> None:
    config = json.loads((ROOT / gate.CONFIG_PATH).read_text(encoding="utf-8"))
    config["claims_policy"]["global_H7_closed"] = True
    with pytest.raises(gate.CoordinateFreeK55OrderOneRegistrationError, match="invalid"):
        gate._validate_config(config)
