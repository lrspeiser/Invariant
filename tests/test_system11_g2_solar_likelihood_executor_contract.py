from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.sigma_core import canonical_json_bytes
from sigma_theory_compiler.system11_g2_solar_likelihood_executor_contract import (
    CONFIG_PATH,
    OUTPUT_PATH,
    PRODUCTION_PACKET,
    SYNTHETIC_PACKET,
    System11LikelihoodExecutorError,
    _semantic_sha256,
    build_receipt,
    build_synthetic_packet,
    execute_packet,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "G3A-2f8983c88f504150381064f2",
    "G3A-58e59412e5fe77cd54caf863",
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _production_fixture(candidate_id: str = CANDIDATES[0]) -> dict[str, object]:
    packet = build_synthetic_packet(ROOT, candidate_id, perturbed=False)
    packet["packet_class"] = PRODUCTION_PACKET
    authorization = packet["independent_authorization"]["document"]
    authorization["status"] = "independently_authorized_future_opening"
    packet["independent_authorization"]["semantic_sha256"] = _semantic_sha256(authorization)
    return packet


def _copy_bound_root(tmp_path: Path) -> Path:
    config = _read(ROOT / CONFIG_PATH)
    _write(tmp_path / CONFIG_PATH, config)
    for descriptor in config["source_bindings"].values():
        relative = descriptor["path"]
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    source = "src/sigma_theory_compiler/system11_g2_solar_likelihood_executor_contract.py"
    test = "tests/test_system11_g2_solar_likelihood_executor_contract.py"
    for relative in (source, test):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def test_checked_receipt_replays_and_remains_blocked_on_eight_obligations() -> None:
    receipt = build_receipt(ROOT)
    assert receipt == _read(ROOT / OUTPUT_PATH)
    assert receipt["decision"] == "block"
    assert receipt["executor_status"] == ("registered_and_synthetic_verified_real_records_sealed")
    assert receipt["counts"]["missing_external_opening_obligations"] == 8
    assert receipt["counts"]["real_data_evaluations"] == 0
    assert receipt["data_boundary"]["observational_data_opened"] is False
    assert receipt["claims"]["observational_result_exists"] is False


def test_synthetic_known_answers_cover_pass_and_reject_for_both_actions() -> None:
    receipt = build_receipt(ROOT)
    controls = receipt["synthetic_known_answer_controls"]
    assert len(controls) == 4
    assert {control["candidate_id"] for control in controls} == set(CANDIDATES)
    assert {
        (control["control"], control["observed_decision"], control["observed_chi_square"])
        for control in controls
    } == {
        ("exact_match_pass", "pass", "0/1"),
        ("perturbed_reject", "reject", "3/1"),
    }


def test_future_authorized_packet_scores_exactly_and_deterministically() -> None:
    packet = _production_fixture()
    first = execute_packet(ROOT, packet)
    second = execute_packet(ROOT, packet)
    assert first == second
    assert first["packet_class"] == PRODUCTION_PACKET
    assert first["decision"] == "pass"
    assert first["score"] == {
        "chi_square": "0/1",
        "relative_log_likelihood": "0/1",
        "accept_if_chi_square_lte": "2/1",
        "candidate_independent_normalization_included": False,
        "arithmetic": "exact_rational",
    }
    assert first["record_count"] == 3


def test_synthetic_packet_cannot_enter_production_path() -> None:
    packet = build_synthetic_packet(ROOT, CANDIDATES[0], perturbed=False)
    assert packet["packet_class"] == SYNTHETIC_PACKET
    with pytest.raises(System11LikelihoodExecutorError, match="not a production"):
        execute_packet(ROOT, packet)


def test_wrong_action_fails_closed() -> None:
    packet = _production_fixture()
    packet["action_sha256"] = "0" * 64
    with pytest.raises(System11LikelihoodExecutorError, match="candidate/action"):
        execute_packet(ROOT, packet)


def test_wrong_domain_fails_closed_even_when_descriptor_is_resealed() -> None:
    packet = _production_fixture()
    domain = packet["source_domain"]["document"]
    domain["domain_id"] = "wrong-domain"
    packet["source_domain"]["semantic_sha256"] = _semantic_sha256(domain)
    with pytest.raises(System11LikelihoodExecutorError, match="authorization binding"):
        execute_packet(ROOT, packet)


def test_wrong_split_fails_closed_even_when_descriptor_is_resealed() -> None:
    packet = _production_fixture()
    split = packet["split_commitment"]["document"]
    split["split_id"] = "wrong-split"
    packet["split_commitment"]["semantic_sha256"] = _semantic_sha256(split)
    with pytest.raises(System11LikelihoodExecutorError, match="authorization binding"):
        execute_packet(ROOT, packet)


def test_wrong_primary_root_fails_closed() -> None:
    packet = _production_fixture()
    packet["calibrated_records"][0]["primary_record_root_sha256"] = "f" * 64
    with pytest.raises(System11LikelihoodExecutorError, match="authorization binding"):
        execute_packet(ROOT, packet)


@pytest.mark.parametrize("field", ["observed", "predicted"])
def test_record_or_action_bound_prediction_tamper_fails_closed(field: str) -> None:
    packet = _production_fixture()
    packet["calibrated_records"][0][field]["numerator"] += 1
    with pytest.raises(System11LikelihoodExecutorError, match="authorization binding"):
        execute_packet(ROOT, packet)


@pytest.mark.parametrize("contract", ["parser_contract", "calibration_contract"])
def test_wrong_parser_or_calibration_schema_fails_closed(contract: str) -> None:
    packet = _production_fixture()
    packet[contract]["schema_version"] = "wrong-schema"
    with pytest.raises(System11LikelihoodExecutorError, match="contract changed"):
        execute_packet(ROOT, packet)


def test_bound_authority_tamper_fails_closed(tmp_path: Path) -> None:
    root = _copy_bound_root(tmp_path)
    relative = "runs/engine/solar-direct-signal-calibration-readiness.json"
    calibration = _read(root / relative)
    calibration["observational_authorization"] = True
    _write(root / relative, calibration)
    with pytest.raises(System11LikelihoodExecutorError, match="semantic source changed"):
        build_receipt(root)


def test_cli_executes_only_an_explicit_authorized_packet(tmp_path: Path) -> None:
    packet_path = tmp_path / "authorized-packet.json"
    output_path = tmp_path / "likelihood-result.json"
    _write(packet_path, _production_fixture(CANDIDATES[1]))
    assert (
        main(
            [
                "--root",
                str(ROOT),
                "--packet",
                str(packet_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    result = _read(output_path)
    assert result["decision"] == "pass"
    assert result["candidate_id"] == CANDIDATES[1]
    assert result["claims"]["real_observation_opened_by_committed_receipt"] is False
