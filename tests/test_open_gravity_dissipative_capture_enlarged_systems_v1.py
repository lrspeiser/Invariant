from __future__ import annotations

import copy
import json

import pytest

from sigma_theory_compiler import open_gravity_dissipative_capture_enlarged_systems_v1 as capture


def test_config_has_exact_mechanism_and_receiver_inventory() -> None:
    config = capture.load_config()
    assert tuple(row["id"] for row in config["mechanisms"]) == capture._MECHANISM_IDS
    assert len(config["mechanisms"]) == 8
    assert all(row["receiver"] and row["energy_rule"] for row in config["mechanisms"])
    assert all(
        row["momentum_rule"] and row["angular_momentum_rule"] for row in config["mechanisms"]
    )
    assert all(row["entropy_rule"] and row["capture_rule"] for row in config["mechanisms"])


def test_conservative_focusing_and_static_amplification_do_not_capture() -> None:
    result = capture.conservative_focusing_fixture(capture.load_config())
    assert result["energy_at_infinity"] > 0.0
    assert result["newtonian_pericenter_energy"] == pytest.approx(
        result["energy_at_infinity"], abs=1.0e-14
    )
    assert result["amplified_pericenter_energy"] == pytest.approx(
        result["energy_at_infinity"], abs=1.0e-14
    )
    assert result["amplified_pericenter_speed"] > result["newtonian_pericenter_speed"]
    assert result["captured_newtonian"] is False
    assert result["captured_static_amplified"] is False
    assert result["receiver_energy_gain"] == 0.0


def test_inelastic_capture_closes_all_ledgers_and_entropy_is_positive() -> None:
    result = capture.inelastic_capture_fixture(capture.load_config())
    assert result["energy_before"] > 0.0 > result["energy_after"]
    assert result["receiver_energy_gain"] > 0.0
    assert result["total_energy_closure_error"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["entropy_gain"] > 0.0
    assert result["ledger"]["linear_momentum_closure_error"] == 0.0
    assert result["ledger"]["angular_momentum_closure_error"] == 0.0


def test_chandrasekhar_wake_capture_tracks_bulk_heat_and_spin() -> None:
    result = capture.chandrasekhar_wake_fixture(capture.load_config())
    assert 0.0 < result["maxwell_fraction"] < 1.0
    assert result["energy_before"] > 0.0 > result["energy_after"]
    assert result["receiver_energy_gain"] > 0.0
    assert result["ledger"]["receiver_bulk_momentum_change_y"] > 0.0
    assert result["ledger"]["receiver_spin_gain"] > 0.0
    assert result["total_energy_closure_error"] == pytest.approx(0.0, abs=1.0e-14)


def test_gr_capture_is_a_receiver_control_and_has_newtonian_limit() -> None:
    config = capture.load_config()
    result = capture.gravitational_wave_capture_fixture(config)
    assert result["energy_at_infinity"] > 0.0 > result["energy_after"]
    assert result["radiated_energy"] > result["energy_at_infinity"]
    assert result["radiated_angular_momentum"] > 0.0
    assert result["radiated_linear_momentum_equal_mass_quadrupole"] == 0.0
    modified = copy.deepcopy(config)
    modified["fixtures"]["gw"]["c"] = 10.0
    weak = capture.gravitational_wave_capture_fixture(modified)
    assert weak["radiated_energy"] == pytest.approx(result["radiated_energy"] / 1.0e5)
    assert weak["captured"] is False


def test_same_current_state_different_history_changes_capture() -> None:
    result = capture.single_memory_hysteresis_fixture(capture.load_config())
    quiet = result["quiet_history"]
    post = result["post_pass_history"]
    assert quiet["energy_before"] == pytest.approx(post["energy_before"])
    assert quiet["memory_state"] < post["memory_state"]
    assert quiet["captured"] is False
    assert post["captured"] is True
    assert quiet["total_energy_closure_error"] == pytest.approx(0.0, abs=1.0e-14)
    assert post["total_energy_closure_error"] == pytest.approx(0.0, abs=1.0e-14)
    assert post["ledger"]["receiver_spin_gain"] > quiet["ledger"]["receiver_spin_gain"]
    assert result["matched_current_activation"] == 1.0


def test_bimodal_fixture_matches_only_the_declared_published_calibration() -> None:
    result = capture.bimodal_persistence_fixture(capture.load_config())
    assert result["fractional_drop_first_1_gyr"] == pytest.approx(0.8, abs=0.01)
    assert result["fractional_drop_next_8_5_gyr"] == pytest.approx(0.5, abs=0.03)
    assert result["calibration_only"] is True
    assert result["independent_prediction"] is False


def test_compression_gate_has_a_time_arrow_but_is_not_called_novel() -> None:
    config = capture.load_config()
    result = capture.compression_gate_fixture(config)
    assert result["inbound_receiver_energy_gain_first_order"] > 0.0
    assert result["outbound_receiver_energy_gain_first_order"] == 0.0
    card = next(row for row in config["mechanisms"] if row["id"] == "DC07_COMPRESSION_GATED_BATH")
    assert "DEGENERATE" in card["novelty_grade"]


def test_target_blind_tng_contract_has_receiver_and_collisionless_fields() -> None:
    preflight = capture._data_preflight(capture.load_config())
    assert preflight["status"] == "PASS_FROZEN_SOURCE_CONTRACT_RESPONSE_UNOPENED"
    assert preflight["response_rows_opened"] == 0
    assert preflight["confirmation_fraction"] == 0.2
    assert len(preflight["predictions"]) == 6
    assert "SOURCE_BLOCKED" in preflight["missing_data_action"]


def test_ranking_keeps_data_and_theory_grades_separate() -> None:
    rows = capture._mechanism_ranking(capture.load_config())
    assert rows[0]["mechanism_id"] == "DC06_TIMEWELL_BIMODAL_MEMORY_BATH"
    assert rows[0]["is_new_publication_lead"] is True
    assert "NOT_ACTION_DERIVED" in rows[0]["theory_grade"]
    assert any(not row["is_new_publication_lead"] for row in rows)
    assert {row["rank"] for row in rows} == set(range(1, 9))


def test_strongest_counterexample_is_collisionless_relaxation() -> None:
    rows = capture._counterexamples()
    assert rows[0]["id"] == "CE01_COLLISIONLESS_VIOLENT_RELAXATION"
    assert rows[0]["strength"] == "STRONGEST"
    assert "without a new dissipative receiver" in rows[0]["statement"]
    assert any(row["id"] == "CE04_MANGA_VISIBLE_DISTURBANCE_FAILURE" for row in rows)


def test_receipt_retains_every_failure_and_has_exact_claim_boundary() -> None:
    receipt = capture.build_receipt()
    assert receipt["status"] == "PASS_ENLARGED_SYSTEM_CAPTURE_SIGNATURES_TNG_PREFLIGHT_FROZEN"
    assert receipt["mechanism_counts"] == {
        "total": 8,
        "conservative_controls": 2,
        "published_dissipative_controls": 3,
        "new_receiver_hypotheses": 3,
    }
    assert receipt["synthetic_fixture_count"] == 7
    assert all(receipt["conservation_checks"].values())
    assert receipt["conservative_controls_capture"] is False
    assert receipt["same_state_history_conditioned_capture"] is True
    assert receipt["access_accounting"]["new_real_data_scores"] == 0
    assert (
        "a real-data fit or independent confirmation"
        in receipt["claim_boundary"]["does_not_establish"]
    )


def test_all_artifacts_are_deterministic_and_report_novelty_boundary() -> None:
    config = capture.load_config()
    first = capture.build_artifacts(config)
    second = capture.build_artifacts(config)
    assert first == second
    assert set(first) == {
        "theory-cards.jsonl",
        "synthetic-fixtures.json",
        "tng-target-blind-preflight.json",
        "mechanism-ranking.csv",
        "counterexamples.json",
        "report.md",
    }
    assert b"The potentially publishable claim is not the sum of exponentials" in first["report.md"]
    assert b"Boylan-Kolchin--Ma--Quataert" in first["report.md"]
    assert len(first["theory-cards.jsonl"].splitlines()) == 8
    assert len(json.loads(first["synthetic-fixtures.json"])) == 7


def test_atomic_packet_round_trip(tmp_path, monkeypatch) -> None:
    output = tmp_path / "receipt.json"
    artifact_directory = tmp_path / "artifacts"
    monkeypatch.setattr(capture, "OUTPUT_PATH", output)
    monkeypatch.setattr(capture, "ARTIFACT_DIRECTORY", artifact_directory)
    assert capture.write_packet() == "CREATED"
    capture.validate_receipt()
    assert capture.write_packet() == "EXISTING_IDENTICAL"
    assert len(list(artifact_directory.iterdir())) == 6


def test_atomic_no_clobber_rejects_different_content(tmp_path) -> None:
    path = tmp_path / "sealed.txt"
    assert capture._atomic_no_clobber(path, b"original") == "CREATED"
    assert capture._atomic_no_clobber(path, b"original") == "EXISTING_IDENTICAL"
    with pytest.raises(capture.DissipativeCaptureError, match="existing artifact differs"):
        capture._atomic_no_clobber(path, b"forgery")


def test_config_mutations_fail_closed() -> None:
    config = capture.load_config()
    for mutate in (
        lambda value: value["access_contract"].__setitem__("new_real_data_scores", 1),
        lambda value: value["real_data_preflight"].__setitem__("response_status", "OPENED"),
        lambda value: value["ranking_policy"].__setitem__("retain_all_failures", False),
        lambda value: value["mechanisms"].pop(),
        lambda value: value.__setitem__("output_path", "elsewhere.json"),
    ):
        forged = copy.deepcopy(config)
        mutate(forged)
        with pytest.raises(capture.DissipativeCaptureError):
            capture.validate_config(forged)


def test_published_source_contract_has_primary_controls_and_data_release() -> None:
    sources = {row["id"]: row for row in capture.load_config()["published_sources"]}
    assert sources["SRC_PETERS_1964"]["doi"] == "10.1103/PhysRev.136.B1224"
    assert sources["SRC_TNG_RELEASE_2019"]["doi"] == "10.1186/s40668-019-0028-x"
    assert sources["SRC_YOUNG_2018"]["doi"] == "10.1088/1475-7516/2018/02/033"
    assert sources["SRC_CALDEIRA_LEGGETT_1983"]["role"].startswith("closest general")
    assert sources["SRC_BOYLAN_KOLCHIN_2008"]["doi"] == "10.1111/j.1365-2966.2007.12530.x"
    assert sources["SRC_FAN_2013_DDDM"]["doi"] == "10.1103/PhysRevLett.110.211302"
    assert sources["SRC_HUO_2020"]["doi"] == "10.1088/1475-7516/2020/06/051"
