from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_gain_persistence_gp01_xcop_source_preflight import (
    CONFIG_CONTENT_SHA256,
    CONFIG_PATH,
    CONFIG_SHA256,
    DECISION,
    MODULE_PATH,
    TEST_PATH,
    GP01XcopSourcePreflightError,
    _atomic_no_clobber,
    build_receipt,
    content_sha256,
    file_sha256,
    load_config,
    receipt_content_sha256,
    validate_config,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_upstream_contract_are_exact() -> None:
    config = load_config(ROOT)
    assert file_sha256(ROOT / CONFIG_PATH) == CONFIG_SHA256
    assert content_sha256(config) == CONFIG_CONTENT_SHA256
    for binding in config["upstream_bindings"]:
        assert file_sha256(ROOT / binding["path"]) == binding["sha256"]


def test_source_only_receipt_rebuilds_and_opens_no_response() -> None:
    receipt = build_receipt(ROOT)
    assert receipt["decision"] == DECISION
    assert receipt["content_sha256"] == receipt_content_sha256(receipt)
    access = receipt["source_access"]
    assert access["source_files_opened"] == 13
    assert access["source_bytes_opened"] == 308160
    assert access["density_rows_opened"] == 521
    assert access["stellar_rows_opened"] > 0
    assert {row["role"] for row in access["opened_files"]} == {"density", "stellar_mass"}
    assert access["pressure_files_opened"] == 0
    assert access["temperature_files_opened"] == 0
    assert access["response_rows_opened"] == 0
    assert access["scientific_scores_computed"] == 0


def test_every_cluster_lacks_the_frozen_y100_transport_anchor() -> None:
    receipt = build_receipt(ROOT)
    expected_maxima = {
        "A1644": 0.05583,
        "A1795": 0.9825,
        "A2142": 0.4005,
        "A2255": 0.07973,
        "A2319": 0.3299,
        "A3266": 0.09668,
        "A85": 1.049,
        "ZW1215": 0.4754,
    }
    assert len(receipt["clusters"]) == 8
    for row in receipt["clusters"]:
        assert row["nodes_at_or_above_y100"] == 0
        assert row["outward_y100_crossing_count"] == 0
        assert row["transport_status"] == "SOURCE_BLOCKED_NO_UNIQUE_Y100_ANCHOR"
        assert row["y_max"] == pytest.approx(expected_maxima[row["cluster"]], rel=5e-4)
    assert receipt["adjudication"]["transport_source_ready"] == 0
    assert receipt["adjudication"]["transport_source_blocked"] == 8


def test_local_aqual_and_elliptic_source_dispositions_are_honest() -> None:
    receipt = build_receipt(ROOT)
    for row in receipt["clusters"]:
        assert row["local_status"] == "SOURCE_READY_LOCAL_RADIAL_CONTROL"
        assert row["aqual_status"] == "EQUIVALENCE_LINK_SPHERICAL_SCORE_ONCE"
        assert row["elliptic_status"] == "SOURCE_READY_PENDING_EXACT_SPHERICAL_SOLVER"
        assert row["telegraph_status"] == "SOURCE_BLOCKED_NO_SOURCE_HISTORY"
        assert row["R_b_r90_kpc"] > 0.0
        assert row["rho_reference_kg_m3"] > 0.0
        assert row["tidal_reference_s_minus_2"] > 0.0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CONFIRMED"),
        (("source_contract", "allowed_roles"), ["density", "pressure"]),
        (("frozen_source_mapping", "a_star_m_s2"), 9.9e-10),
        (("frozen_source_mapping", "transport_anchor_y"), 1.0),
        (("branch_adjudication", "GP01-T1"), "SOURCE_READY"),
        (("claim_boundary", "response_scoring_authorized"), True),
        (("claim_boundary", "pressure_or_temperature_opened"), True),
        (("claim_boundary", "observational_signal_claimed"), True),
        (("zero_access", "pressure_rows_opened"), 1),
    ],
)
def test_material_config_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = load_config(ROOT)
    mutated = copy.deepcopy(config)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(GP01XcopSourcePreflightError):
        validate_config(mutated)


def test_stored_receipt_is_exact_and_bound_to_implementation() -> None:
    stored = validate_receipt(ROOT)
    assert stored == build_receipt(ROOT)
    assert stored["bindings"]["config_sha256"] == file_sha256(ROOT / CONFIG_PATH)
    assert stored["bindings"]["module_sha256"] == file_sha256(ROOT / MODULE_PATH)
    assert stored["bindings"]["test_sha256"] == file_sha256(ROOT / TEST_PATH)


def test_coherently_rehashed_receipt_forgery_is_rejected(tmp_path: Path) -> None:
    stored = build_receipt(ROOT)
    forged = copy.deepcopy(stored)
    forged["adjudication"]["transport_source_ready"] = 8
    forged["adjudication"]["transport_source_blocked"] = 0
    forged["decision"] = "CONFIRMED"
    forged["content_sha256"] = receipt_content_sha256(forged)
    assert forged["content_sha256"] == receipt_content_sha256(forged)
    assert forged != build_receipt(ROOT)
    target = tmp_path / "forged.json"
    target.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(GP01XcopSourcePreflightError):
        validate_receipt(ROOT, target)


def test_atomic_writer_is_no_clobber_and_preserves_other_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    payload = b'{"safe":true}\n'
    assert _atomic_no_clobber(target, payload) == "CREATED"
    assert _atomic_no_clobber(target, payload) == "EXISTING_IDENTICAL"
    assert target.read_bytes() == payload
    target.write_bytes(b"different\n")
    with pytest.raises(GP01XcopSourcePreflightError):
        _atomic_no_clobber(target, payload)
    assert target.read_bytes() == b"different\n"


def test_no_forbidden_tokens_enter_opened_source_ledger() -> None:
    receipt = build_receipt(ROOT)
    rendered = json.dumps(receipt["source_access"]["opened_files"], sort_keys=True).lower()
    for token in ("pressure", "temperature", "vobs", "lensing", "confirmation"):
        assert token not in rendered
    assert hashlib.sha256((ROOT / CONFIG_PATH).read_bytes()).hexdigest() == CONFIG_SHA256
