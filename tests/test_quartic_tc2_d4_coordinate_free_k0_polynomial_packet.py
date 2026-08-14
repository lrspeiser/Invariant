from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import quartic_tc2_d4_coordinate_free_k0_polynomial_packet as gate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def campaign() -> dict[str, object]:
    return gate.build_campaign(ROOT, ROOT / gate.CONFIG_PATH)


def test_exact_polynomial_packet_matches_immutable_artifact(
    campaign: dict[str, object],
) -> None:
    artifact = json.loads((ROOT / gate.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert campaign == artifact
    gate.validate_campaign(artifact, ROOT)
    packet = artifact["exact_K0_polynomial_packet"]
    assert packet["shape"] == [55, 55]
    assert packet["normal_form_terms"] > 0
    assert artifact["exact_replay"]["K0_P55_symmetrizer_remainder_entries"] == 0
    assert artifact["counts"]["K55_order_one_packets_authorized_for_construction"] == 15
    assert artifact["counts"]["K55_order_one_packets_registered"] == 0
    assert artifact["counts"]["manifest_registered_after"] == 79


def test_tampered_packet_fails_closed(campaign: dict[str, object]) -> None:
    tampered = copy.deepcopy(campaign)
    tampered["exact_K0_polynomial_packet"]["entries"][0]["terms"][0]["coefficient"] = "999"
    with pytest.raises(gate.CoordinateFreeK0PolynomialPacketError, match="content hash"):
        gate.validate_campaign(tampered, ROOT)


def test_noncanonical_term_fails_closed(campaign: dict[str, object]) -> None:
    tampered = copy.deepcopy(campaign)
    tampered["exact_K0_polynomial_packet"]["entries"][0]["terms"][0]["powers"][0] = 2
    packet = tampered["exact_K0_polynomial_packet"]
    packet["content_sha256"] = gate._content_hash(packet)
    tampered["content_sha256"] = gate._content_hash(tampered)
    with pytest.raises(gate.CoordinateFreeK0PolynomialPacketError, match="noncanonical"):
        gate.validate_campaign(tampered, ROOT)


def test_config_tamper_rejected() -> None:
    config = json.loads((ROOT / gate.CONFIG_PATH).read_text(encoding="utf-8"))
    config["claims_policy"]["global_H7_closed"] = True
    with pytest.raises(gate.CoordinateFreeK0PolynomialPacketError, match="invalid"):
        gate._validate_config(config)
