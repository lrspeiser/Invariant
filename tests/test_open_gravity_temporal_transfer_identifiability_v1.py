from __future__ import annotations

import copy

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_temporal_transfer_identifiability_v1 as temporal


def test_exact_80_concept_coverage() -> None:
    coverage = temporal.build_receipt()["coverage"]
    assert coverage["concepts"] == 80
    assert set(coverage["by_architecture"].values()) == {20}
    assert len({row["concept_id"] for row in coverage["rows"]}) == 80
    assert all(
        row["empirical_status"] == "UNTESTED_SOURCE_HISTORY_REQUIRED" for row in coverage["rows"]
    )


def test_static_data_are_exactly_degenerate() -> None:
    receipt = temporal.build_receipt()
    for rows in receipt["fingerprints"]["transfer_rows"].values():
        assert rows[0] == {"omega": 0.0, "magnitude": 1.0, "phase": 0.0}
    assert receipt["checks"]["STATIC_ZERO_FREQUENCY_DEGENERACY"]


def test_retarded_and_memory_have_distinct_nonzero_frequency_signatures() -> None:
    omega = np.array([0.25, 0.5, 1.0, 2.0])
    retarded = temporal.transfer_retarded(omega, delay=0.5)
    memory = temporal.transfer_memory(omega, tau=0.5)
    assert np.allclose(np.abs(retarded), 1.0, rtol=0.0, atol=1.0e-15)
    assert np.all(np.abs(memory) < 1.0)
    assert not np.array_equal(retarded, memory)


def test_stochastic_mean_needs_variance_channel() -> None:
    receipt = temporal.build_receipt()
    fingerprints = receipt["fingerprints"]
    assert fingerprints["memory_and_stochastic_mean_exactly_equal"]
    assert fingerprints["stochastic_stationary_variance"] == pytest.approx(0.000625)
    assert fingerprints["pairwise_fingerprint_distances"]["A16_MEMORY__A18_STOCHASTIC"] > 0.0


def test_step_signatures_are_causal_and_distinct() -> None:
    rows = temporal.build_receipt()["step_signatures"]
    assert rows["retarded_first_positive_time"] >= 1.5
    assert rows["memory_monotone"]
    assert rows["resonance_peak"] > 1.1
    assert rows["no_advanced_response"]


def test_telegraph_retains_instantaneous_source_caveat() -> None:
    row = temporal.build_receipt()["telegraph"]
    assert row["necessary_speed_over_c"] == pytest.approx(0.5)
    assert row["instantaneous_baryonic_source_still_unresolved"]
    assert row["static_spatial_attenuation"][0] == 1.0


def test_config_mutation_fails_closed() -> None:
    config = temporal.load_config()
    mutated = copy.deepcopy(config)
    mutated["driver_ids"].pop()
    with pytest.raises(temporal.TemporalIdentifiabilityError):
        temporal.validate_config(mutated)


def test_no_scientific_access() -> None:
    receipt = temporal.build_receipt()
    assert set(receipt["access_accounting"].values()) == {0}
    assert (
        "that any temporal architecture fits a galaxy or cluster"
        in receipt["claim_boundary"]["does_not_establish"]
    )


def test_receipt_round_trip(tmp_path, monkeypatch) -> None:
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(temporal, "OUTPUT_PATH", output)
    assert temporal.write_receipt() == "CREATED"
    temporal.validate_receipt()
    assert temporal.write_receipt() == "EXISTING_IDENTICAL"
