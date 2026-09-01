from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_quantum_entity_wave_atlas_v1 as atlas


def _copy_package(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    for relative in (atlas.CONFIG_PATH, atlas.MODULE_PATH, atlas.TEST_PATH):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    return tmp_path


def test_config_has_fourteen_fully_typed_cards_and_independent_scores() -> None:
    config = atlas.load_config()
    assert tuple(card["id"] for card in config["theory_cards"]) == atlas.CARD_IDS
    assert len(config["theory_cards"]) == 14
    assert config["score_contract"]["no_composite_score"] is True
    assert config["access_contract"]["observational_rows_read"] == 0
    assert config["claim_boundary"]["any_branch_empirically_supported"] is False
    for card in config["theory_cards"]:
        assert set(card["observables"]) == {"matter", "photon", "tensor"}
        assert card["dimensional_closure"]
        assert card["classical_limit"]
        assert card["equivalence_relations"]
        assert card["nearest_primary_literature"]["url"].startswith("https://")
        assert card["falsifier"]["unchanged_test"]


def test_gaussian_classical_record_equivalence_is_exact() -> None:
    result = atlas.gaussian_pushforward_fixture()
    assert result["output_mean"] == [-1, -1]
    assert result["output_covariance"] == [[19, 7], [7, 5]]
    assert result["classical_record_discriminator_exists"] is False
    assert result["proof_status"] == "EXACT_RATIONAL_PUSHFORWARD_EQUALITY"


def test_finite_poisson_occupation_escapes_matched_gaussian_two_point() -> None:
    result = atlas.poisson_cumulant_fixture(25)
    assert result["mean"] == result["variance"] == 25
    assert result["connected_kappa3"] == result["connected_kappa4"] == 25
    assert result["matched_gaussian_kappa3"] == result["matched_gaussian_kappa4"] == 0
    assert result["normalized_skewness"] == pytest.approx(0.2)
    assert result["excess_kurtosis"] == pytest.approx(0.04)


def test_entangling_channel_and_dispersion_fixtures() -> None:
    entanglement = atlas.entangling_channel_fixture(math.pi / 12.0)
    assert entanglement["concurrence"] == pytest.approx(0.5)
    assert entanglement["entangled_output"] is True
    dispersion = atlas.dispersion_fixture()
    delays = [row["delay_over_light_time_unit"] for row in dispersion["rows"]]
    assert delays == sorted(delays, reverse=True)
    assert dispersion["low_frequency_arrives_later"] is True


def test_rank_memory_capture_and_classical_limit_fixtures() -> None:
    rank = atlas.polarization_rank_fixture()
    assert rank["two_detector_rank"] == 2
    assert rank["ideal_six_detector_rank"] == 6
    assert rank["two_detector_identifiable"] is False
    memory = atlas.memory_hysteresis_fixture()
    assert memory["response_lag_radians"] > 0
    assert 0 < memory["source_off_envelope_at_t5"] < 1
    capture = atlas.capture_stationary_fixture()
    assert capture["stationary_mean"] == capture["stationary_variance"] == 12
    assert capture["connected_kappa3"] == capture["connected_kappa4"] == 12
    limit = atlas.classical_limit_fixture()
    assert limit["relative_poisson_noise"][-1] == pytest.approx(0.01)


def test_pairwise_atlas_retains_collisions_and_counterexamples() -> None:
    config = atlas.load_config()
    rows = atlas.pairwise_discriminators(config)
    assert len(rows) == math.comb(14, 2)
    assert all("warning" in row for row in rows)
    assert len(atlas.counterexamples()) == 8
    assert any(row["id"] == "CEX_TWO_POINT_QUANTUM_PROOF" for row in atlas.counterexamples())


def test_deterministic_build_check_replay_and_tamper_gate(tmp_path: Path) -> None:
    base = _copy_package(tmp_path)
    assert atlas.build(base) == "CREATED"
    assert atlas.check(base) == "VALID"
    assert atlas.build(base) == "EXISTING_IDENTICAL"
    receipt = json.loads((base / atlas.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert receipt["counts"]["theory_cards"] == 14
    assert receipt["counts"]["pairwise_comparisons"] == 91
    assert receipt["counts"]["real_observational_rows"] == 0
    assert receipt["lead_triage"]["strongest_synthesis_candidate"] == (
        "Q12_QUANTIZED_TIMEWELL_MEMORY_MODE"
    )
    artifact = base / atlas.ARTIFACT_DIR / "report.md"
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(atlas.QuantumAtlasError, match="artifact differs"):
        atlas.check(base)


def test_receipt_claim_boundary_and_access_ledger_are_zero() -> None:
    config = atlas.load_config()
    receipt, payloads = atlas.build_receipt(config, Path(__file__).resolve().parents[1])
    assert receipt["access_ledger"] == config["access_contract"]
    assert set(receipt["access_ledger"].values()) == {0}
    assert receipt["claim_boundary"]["real_observational_rows_scored"] is False
    assert receipt["claim_boundary"]["historical_novelty_established"] is False
    assert "public-data-preflights.json" in payloads
    assert "counterexamples.json" in payloads
