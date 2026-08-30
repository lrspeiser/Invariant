from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mpmath as mp
import pytest

from sigma_theory_compiler import gravity_extended_source_clock_response as clock

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_predecessor_are_exact() -> None:
    config = clock.load_config(ROOT)
    clock.validate_config(config, ROOT)
    predecessor = clock.validate_predecessor(config, ROOT)
    assert predecessor["git_commit"] == "78a9742bd87efee93db404a26b0d2025db89602d"
    assert predecessor["artifact_count"] == 3
    assert predecessor["valid"] is True


def test_symbolic_derivation_passes() -> None:
    checks = clock.symbolic_checks()
    assert len(checks) == 10
    assert all(row["passed"] for row in checks)
    assert checks[5]["check_id"] == "S06_BETA_RATIO"


def test_acceleration_only_clock_is_exact_rar_rewrite() -> None:
    mp.mp.dps = 50
    for y in (mp.mpf("1e-4"), mp.mpf("0.1"), mp.mpf(1), mp.mpf(10)):
        c = clock.clock_ratio(y, mp.mpf(0))
        assert mp.almosteq(1 / c**2, clock.nu_rar(y))


def test_compact_exterior_solar_limit() -> None:
    controls = clock.evaluate_synthetic_controls()["solar_compact_exterior"]
    assert controls["passes"] is True
    assert all(row["eta"] == "0" for row in controls["rows"])
    assert all(abs(mp.mpf(row["nu_minus_one"])) < mp.mpf("1e-20") for row in controls["rows"])


def test_outer_galaxy_selects_rar_and_passes_frozen_gate() -> None:
    galaxy = clock.evaluate_synthetic_controls()["outer_galaxy"]
    assert galaxy["passes"] is True
    assert galaxy["selected_channel"] == "RAR"
    assert max(mp.mpf(row["eta"]) for row in galaxy["eta_rows"]) < 0
    assert mp.mpf(galaxy["max_speed_fractional_spread"]) < mp.mpf("0.06")
    assert abs(mp.mpf(galaxy["btfr_slope"]) - 4) < mp.mpf("0.3")


def test_lensing_plumbing_selects_rar_but_remains_assumption() -> None:
    lensing = clock.evaluate_synthetic_controls()["lensing"]
    assert lensing["passes"] is True
    assert lensing["assumption_only"] is True
    assert lensing["selected_channel"] == "RAR"
    assert mp.mpf(lensing["worst_flatness"]) < mp.mpf("0.08")
    assert mp.mpf(lensing["worst_consistency"]) < mp.mpf("0.15")
    assert mp.mpf(lensing["max_extended_to_rar_channel_ratio"]) < 1


def test_cluster_extended_channel_matches_exact_control() -> None:
    cluster = clock.evaluate_synthetic_controls()["cluster"]
    assert cluster["passes"] is True
    assert mp.mpf(cluster["max_fractional_deviation"]) < mp.mpf("1e-40")
    assert len(cluster["rows"]) == 5
    assert all(row["selected_channel"] == "extended_source" for row in cluster["rows"])
    assert all(mp.almosteq(mp.mpf(row["g_pred_over_gdyn"]), 1) for row in cluster["rows"])
    assert all(mp.mpf(row["nu_extended"]) > mp.mpf(row["nu_rar"]) for row in cluster["rows"])


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("clock_hypothesis", "combined_response", "forged"),
        ("source_geometry_contract", "target_independence", "forged"),
        ("synthetic_control_contract", "real_observational_rows_authorized", True),
        ("light_and_gravity_contract", "same_action_lensing_derived", True),
        ("adjudication", "real_cluster_score_executed", True),
        ("claim_boundary", "dark_matter_eliminated", True),
        ("next_test_contract", "selection_rule", "fit after target"),
        ("zero_access_and_compute", "real_target_rows_opened", 1),
    ],
)
def test_nested_contract_mutations_fail_closed(section: str, key: str, value: object) -> None:
    forged = copy.deepcopy(clock.load_config(ROOT))
    forged[section][key] = value
    with pytest.raises(clock.ExtendedSourceClockError):
        clock.validate_config(forged, ROOT)


def test_build_receipt_is_bounded_and_zero_access() -> None:
    receipt = clock.build_receipt(ROOT)
    assert receipt["decision"] == clock.DECISION
    assert receipt["counts"]["symbolic_checks"] == 10
    assert receipt["counts"]["cluster_probes"] == 5
    assert receipt["counts"]["real_rows"] == 0
    assert all(
        receipt["zero_access_and_compute"][key] == 0 for key in receipt["zero_access_and_compute"]
    )
    assert receipt["adjudication"]["scientific_claim_allowed"] is False
    assert receipt["claim_boundary"]["publication_readiness_changed"] is False


def test_stored_receipt_matches_exact_rebuild() -> None:
    stored = clock.check_receipt(ROOT)
    assert stored == clock.build_receipt(ROOT)
    payload = dict(stored)
    content = payload.pop("content_sha256")
    assert content == clock._content_sha(payload)


def test_tampered_receipt_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = clock.build_receipt(ROOT)
    receipt["claim_boundary"]["real_cluster_compatibility_established"] = True
    payload = dict(receipt)
    payload.pop("content_sha256")
    receipt["content_sha256"] = clock._content_sha(payload)
    output = tmp_path / "receipt.json"
    output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    monkeypatch.setattr(clock, "OUTPUT_PATH", output.relative_to(tmp_path))
    with pytest.raises(clock.ExtendedSourceClockError):
        clock.check_receipt(tmp_path)


def test_atomic_no_clobber_race_retains_one_complete_payload(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"

    def publish(payload: bytes) -> str:
        try:
            return clock._atomic_no_clobber(target, payload)
        except clock.ExtendedSourceClockError:
            return "REFUSED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (b"first", b"second")))
    assert target.read_bytes() in {b"first", b"second"}
    assert outcomes.count("CREATED") == 1
    assert outcomes.count("REFUSED") == 1


def test_status_preserves_theory_and_data_ceiling() -> None:
    result = clock.status(ROOT)
    assert result["valid"] is True
    assert result["solar_limit"] is True
    assert result["outer_galaxy"] is True
    assert result["lensing_assumption_only"] is True
    assert result["cluster_synthetic"] is True
    assert result["real_rows"] == 0
    assert result["real_cluster_score"] is False
    assert result["covariant_theory"] is False
