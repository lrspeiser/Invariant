from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
)
from sigma_theory_compiler.system10_cylindrical_r_positive_twelve_candidate_aw import (
    System10TwelveCandidateAWError,
    _verify_packet,
    build_candidate_packet,
    build_census_receipt,
    run_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_twelve_candidate_aw.json"
PACKETS = ROOT / "runs/math/system10-cylindrical-r-positive-twelve-candidate-aw"
RECEIPT = PACKETS / "receipt.json"
REPRESENTATIVE_ROWS = (
    ROOT / "runs/math/system10-cylindrical-r-positive-gravity-scalar-aw-materializer"
)


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    return build_census_receipt(CONFIG, PACKETS)


def test_committed_census_replays_exactly(receipt: dict[str, object]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["counts"] == {
        "candidate_packets": 12,
        "rows": 132,
        "A_entries": 1452,
        "W_entries": 132,
        "tube_admitted_candidates": 12,
    }


def test_all_candidate_packets_have_atomic_row_and_entry_seals(
    config: dict[str, object],
) -> None:
    for index in range(12):
        packet = json.loads((PACKETS / f"candidate-{index:02d}.json").read_text())
        _verify_packet(packet, index, config)
        assert packet["row_count"] == 11
        assert packet["A_entry_count"] == 121
        assert packet["W_entry_count"] == 11
        assert all(len(row["A_entries"]) == 11 for row in packet["rows"])
        assert all(row["certificates"]["affine_residual"] == "0" for row in packet["rows"])


def test_general_adapter_exactly_reproduces_representative_A_and_W() -> None:
    packet = json.loads((PACKETS / "candidate-00.json").read_text())
    for row in range(11):
        prior = json.loads((REPRESENTATIVE_ROWS / f"row-{row:02d}.json").read_text())
        current = packet["rows"][row]
        assert [entry["expression"] for entry in current["A_entries"]] == [
            entry["expression"] for entry in prior["A_entries"]
        ]
        assert current["W_entry"]["expression"] == prior["W_entry"]["expression"]


def test_two_candidate_packets_rebuild_deterministically() -> None:
    assert build_candidate_packet(CONFIG, 0) == json.loads(
        (PACKETS / "candidate-00.json").read_text()
    )
    assert build_candidate_packet(CONFIG, 6) == json.loads(
        (PACKETS / "candidate-06.json").read_text()
    )


def test_candidate_coefficients_materially_change_packets() -> None:
    first = json.loads((PACKETS / "candidate-00.json").read_text())
    second = json.loads((PACKETS / "candidate-01.json").read_text())
    assert first["coefficients"] != second["coefficients"]
    assert first["content_sha256"] != second["content_sha256"]
    assert first["rows"][10]["A_entries"] != second["rows"][10]["A_entries"]


def test_common_tube_is_admitted_candidate_by_candidate(receipt: dict[str, object]) -> None:
    assert receipt["decision"] == "BOUNDED_PASS_ALL_TWELVE_A_W_PACKETS_AND_COMMON_LOCAL_TUBE"
    assert receipt["common_preregistered_tube"] == {
        "r": "1",
        "real_v_10_interval": ["-1/4", "1/4"],
        "all_other_candidate_A_symbols": "0",
        "all_candidates_admitted": True,
    }
    assert all(item["tube_admitted"] for item in receipt["candidate_results"])
    assert all(
        item["exact_absolute_determinant_lower_bound"] != "0"
        for item in receipt["candidate_results"]
    )


def test_resume_reuses_verified_packet_and_tamper_fails(tmp_path: Path) -> None:
    first = run_candidates(CONFIG, tmp_path, [0], root=ROOT)
    second = run_candidates(CONFIG, tmp_path, [0], root=ROOT)
    assert first == second
    target = tmp_path / "candidate-00.json"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(System10TwelveCandidateAWError, match="packet seal mismatch"):
        run_candidates(CONFIG, tmp_path, [0], root=ROOT)
    assert target.read_text(encoding="utf-8") == "{}\n"


def test_config_candidate_and_determinant_tamper_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["candidates"][0]["coefficients"]["a10"] = "0"
    tampered = tmp_path / "candidate.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10TwelveCandidateAWError, match="candidate manifest changed"):
        build_candidate_packet(tampered, 0, root=ROOT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["expected_slice_determinants"][0]["determinant"] = "0"
    tampered = tmp_path / "determinant.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10TwelveCandidateAWError, match="packet seal mismatch"):
        build_census_receipt(tampered, PACKETS, root=ROOT)


def test_source_and_receipt_seals_are_closed(
    config: dict[str, object], receipt: dict[str, object]
) -> None:
    for binding in config["source_evidence"].values():
        assert _canonical_lf_sha(ROOT / binding["path"]) == binding["canonical_lf_sha256"]
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_claims_remain_local_and_do_not_infer_global_solvability(
    receipt: dict[str, object],
) -> None:
    claims = receipt["claims"]
    assert claims["all_twelve_candidate_A_W_packets_materialized"] is True
    assert claims["common_local_tube_admitted"] is True
    assert claims["global_candidate_domains_invertible"] is False
    assert claims["accelerations_solved_on_common_tube"] is False
    assert claims["full_rhs"] is False
    assert claims["propagation"] is False
    assert claims["hyperbolicity"] is False
