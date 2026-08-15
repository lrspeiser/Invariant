from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_twelve_candidate_sourced_metric_euler_binding import (
    QuarticSourcedMetricEulerBindingError,
    _canonical_sha,
    _exact_variation_replay,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_twelve_candidate_sourced_metric_euler_binding.json"
OUTPUT = ROOT / "runs/math/quartic-twelve-candidate-sourced-metric-euler-binding/receipt.json"


def test_all_twelve_sourced_metric_euler_equations_are_bound() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY"
    results = receipt["candidate_results"]
    assert len(results) == 12
    assert {item["outcome"] for item in results} == {"PASS"}
    assert len({item["sourced_metric_euler_sha256"] for item in results}) == 12
    for item in results:
        assert item["sourced_metric_euler_sha256"] == _canonical_sha(
            item["sourced_metric_euler_manifest"]
        )
        assert item["sourced_metric_euler_manifest"]["equation"] == ("E_gf^mu_nu-T_total^mu_nu/2=0")


def test_exact_variation_sign_and_eleven_row_map() -> None:
    replay = _exact_variation_replay()
    assert replay["matter_variation_coefficients"] == ["-1/2", "-1/2", "-1/2"]
    assert replay["wrong_sign_negative"]["independent_sector_residual_coefficients"] == [
        "1",
        "1",
        "1",
    ]
    assert replay["omitted_fluid_negative"]["independent_sector_residual_coefficients"] == [
        "0",
        "0",
        "1/2",
    ]
    rows = replay["eleven_row_source_map"]
    assert len(rows) == 11
    assert rows[0] == {"row": 0, "metric_pair": [0, 0], "source": "-T_total^00/2"}
    assert rows[1] == {
        "row": 1,
        "metric_pair": [0, 1],
        "source": "-sqrt(2) T_total^01/2",
    }
    assert rows[-1]["field"] == "phi_g" and rows[-1]["source"] == "0"


def test_counts_claims_and_content_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["counts"] == {
        "candidates": 12,
        "sourced_metric_euler_bindings_passed": 12,
        "unique_sourced_metric_euler_hashes": 12,
        "metric_equation_rows_per_candidate": 10,
        "unchanged_gravity_scalar_rows_per_candidate": 1,
        "total_registered_gravity_rows": 132,
        "exact_variation_residuals": 2,
        "negative_controls": 2,
        "sourced_acceleration_solutions": 0,
        "rejects": 0,
    }
    claims = receipt["claims"]
    assert claims["all_twelve_sourced_metric_euler_equations_hash_bound"] is True
    assert claims["inverse_metric_variation_normalization_and_sign_replayed"] is True
    assert not any(
        value
        for name, value in claims.items()
        if name
        not in {
            "all_twelve_sourced_metric_euler_equations_hash_bound",
            "inverse_metric_variation_normalization_and_sign_replayed",
        }
    )
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_checked_receipt_is_current_and_path_free() -> None:
    checked = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert checked == build_receipt(CONFIG, root=ROOT)
    rendered = json.dumps(checked, sort_keys=True)
    assert str(ROOT) not in rendered
    assert "\\\\" not in rendered


@pytest.mark.parametrize(
    "binding",
    ["total_action_binding", "vacuum_euler", "scalar_stress", "maxwell_stress", "fluid_stress"],
)
def test_tampered_json_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["json_bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticSourcedMetricEulerBindingError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_tampered_convention_source_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["source_evidence"]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticSourcedMetricEulerBindingError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["json_bindings"]["fluid_stress"]["path"] = "runs/math/missing-fluid-stress.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticSourcedMetricEulerBindingError, match="cannot read bound file"):
        build_receipt(candidate, root=ROOT)


def test_broadened_acceleration_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["sourced_acceleration_solution"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticSourcedMetricEulerBindingError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
