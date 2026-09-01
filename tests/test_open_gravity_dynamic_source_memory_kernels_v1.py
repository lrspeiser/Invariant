from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_dynamic_source_memory_kernels_v1 as memory

ROOT = Path(__file__).resolve().parents[1]


def _config_unsealed() -> dict[str, object]:
    return json.loads((ROOT / memory.CONFIG_PATH).read_text(encoding="utf-8"))


def test_config_maps_all_predecessor_and_new_dynamic_concepts() -> None:
    config = _config_unsealed()
    memory.validate_config(config)
    assert [row["id"] for row in config["kernels"]] == list(memory.KERNEL_IDS)
    assert config["concept_mapping"] == {
        "predecessor_architectures": [
            "K01_RETARDED",
            "K02_EXPONENTIAL",
            "K04_DAMPED_RESONANCE",
            "K06_STOCHASTIC_OU",
        ],
        "new_architectures": ["K03_BIEXPONENTIAL", "K05_HYSTERETIC"],
        "candidate_architectures": 6,
        "drivers_per_architecture": 20,
        "predecessor_concepts": 80,
        "new_concepts": 40,
        "total_executable_concepts": 120,
    }


def test_static_and_boundary_equivalences_are_exact() -> None:
    config = _config_unsealed()
    kernels = memory._kernel_map(config)
    frequencies = np.asarray([0.0, 0.2, 0.9, 3.0])
    for kernel_id in memory.KERNEL_IDS:
        if kernel_id == "K05_HYSTERETIC":
            continue
        response = memory.transfer_response(
            kernel_id, np.asarray([0.0]), kernels[kernel_id]["parameters"]
        )
        assert response[0] == pytest.approx(1.0 + 0.0j, abs=2.0e-16)
    exp_response = memory.transfer_response("K02_EXPONENTIAL", frequencies, {"tau": 0.7})
    ou_mean = memory.transfer_response("K06_STOCHASTIC_OU", frequencies, {"tau": 0.7})
    mixture_collapse = memory.transfer_response(
        "K03_BIEXPONENTIAL",
        frequencies,
        {"tau1": 0.7, "tau2": 0.7, "weight": 0.23},
    )
    assert np.array_equal(exp_response, ou_mean)
    assert np.allclose(exp_response, mixture_collapse, rtol=0.0, atol=2.0e-16)
    assert np.array_equal(
        memory.transfer_response("K01_RETARDED", frequencies, {"delay": 0.0}),
        np.ones(frequencies.size, dtype=complex),
    )


def test_step_fixture_proves_causality_overshoot_and_persistence() -> None:
    config = _config_unsealed()
    rows = memory._step_signatures(config)
    assert set(rows) == set(memory.KERNEL_IDS)
    assert all(row["pre_source_max_abs"] <= 1.0e-14 for row in rows.values())
    assert rows["K04_DAMPED_RESONANCE"]["overshoot_above_unit"] > 0.4
    assert rows["K02_EXPONENTIAL"]["overshoot_above_unit"] == 0.0
    assert rows["K05_HYSTERETIC"]["final_response"] == pytest.approx(0.15, abs=1.0e-12)


def test_rate_sweep_separates_persistent_hysteresis_from_lti_lag() -> None:
    config = _config_unsealed()
    rows, summary = memory._rate_sweep(config)
    assert len(rows) == 28
    assert abs(summary["K05_HYSTERETIC"]["low_rate_log_slope"]) < 0.15
    assert summary["K05_HYSTERETIC"]["linear_zero_rate_intercept"] > 0.14
    for kernel_id in (
        "K01_RETARDED",
        "K02_EXPONENTIAL",
        "K03_BIEXPONENTIAL",
        "K04_DAMPED_RESONANCE",
        "K06_STOCHASTIC_OU",
    ):
        assert summary[kernel_id]["low_rate_log_slope"] > 0.8


def test_stochastic_mean_equivalence_requires_variance_channel() -> None:
    signature = memory._ou_signature(_config_unsealed())
    assert signature["conditional_mean_equals_K02_exactly"] is True
    assert signature["analytic_stationary_variance"] == pytest.approx(0.0004375)
    assert 0.85 <= signature["ensemble_to_analytic_variance_ratio"] <= 1.15
    assert signature["ensemble_mean_rmse_from_K02"] < 0.003
    assert signature["noise_bath_injection_power"] == pytest.approx(
        signature["stationary_fluctuation_dissipation"]
    )


def test_receiver_state_energy_ledgers_close_and_missing_baths_remain_labeled() -> None:
    config = _config_unsealed()
    _, rate_summary = memory._rate_sweep(config)
    ledger = memory._energy_ledger(config, rate_summary)
    assert ledger["K02_EXPONENTIAL"]["periodic_average_input_minus_dissipation"] < 1e-15
    assert ledger["K03_BIEXPONENTIAL"]["periodic_average_input_minus_dissipation"] < 1e-15
    assert ledger["K04_DAMPED_RESONANCE"]["periodic_average_input_minus_dissipation"] < 1e-15
    assert ledger["K05_HYSTERETIC"]["relative_residual"] < 0.08
    assert ledger["K01_RETARDED"]["completion_status"] == (
        "PROPAGATING_MEDIATOR_STRESS_ENERGY_NOT_DERIVED"
    )
    assert ledger["K05_HYSTERETIC"]["completion_status"] == "BRANCH_RESERVOIR_NOT_DERIVED"


def test_public_nr_benchmark_is_exact_and_not_an_observational_response() -> None:
    config = _config_unsealed()
    times, source, metadata = memory._load_benchmark(ROOT, config)
    assert times.shape == source.shape == (2769,)
    assert metadata["sha256"] == (
        "ed49c3e83f90e70ac85386f183031b7de3d3d6aa78e75a7d284e5a53a5cc0b76"
    )
    assert metadata["sample_rate_hz"] == pytest.approx(4096.0, abs=1e-8)
    assert metadata["observational_response"] is False
    assert np.max(np.abs(source)) == 1.0


def test_nr_waveform_exposes_delay_degeneracy_and_nontrivial_other_shapes() -> None:
    config = _config_unsealed()
    rows, _ = memory._waveform_rows(ROOT, config)
    by_id = {row["kernel_id"]: row for row in rows}
    assert by_id["K01_RETARDED"]["raw_amplitude_projected_mismatch"] > 0.1
    assert by_id["K01_RETARDED"]["amplitude_time_projected_mismatch"] < 1.0e-12
    assert by_id["K01_RETARDED"]["best_time_shift_seconds"] == pytest.approx(
        0.00244140625, abs=1.0e-12
    )
    assert by_id["K04_DAMPED_RESONANCE"]["amplitude_time_projected_mismatch"] > 1e-4
    assert by_id["K06_STOCHASTIC_OU"]["ou_stationary_excess_variance"] > 0.0


def test_equivalence_map_and_ranking_do_not_overstate_novelty() -> None:
    equivalences = memory._equivalence_rows()
    assert len(equivalences) == 8
    assert sum(row["status"].startswith("TRUE_DISCRIMINATOR") for row in equivalences) == 3
    assert (
        next(row for row in equivalences if row["id"] == "E02_COMMON_DELAY_TIME_ORIGIN")["status"]
        == "EXACT_NONIDENTIFIABILITY"
    )
    config = _config_unsealed()
    _, rate_summary = memory._rate_sweep(config)
    ranked = memory._ranked_leads(config, rate_summary, memory._step_signatures(config))
    assert ranked[0]["kernel_id"] == "K05_HYSTERETIC"
    assert "not derived" in ranked[0]["blocker"]
    assert ranked[1]["kernel_id"] == "K04_DAMPED_RESONANCE"
    assert "quasinormal" in ranked[1]["blocker"]


def test_receipt_and_artifacts_are_deterministic() -> None:
    first, first_artifacts = memory.derive_package(ROOT)
    second, second_artifacts = memory.derive_package(ROOT)
    assert first == second
    assert first_artifacts == second_artifacts
    assert first["checks_passed"] == first["checks_total"] == 18
    assert all(first["checks"].values())
    assert first["strongest_counterexample"]["selection_aware_permutation_p"] == 0.75
    assert first["claim_boundary"]["historical_novelty_established"] is False
    assert first["claim_boundary"]["empirical_gravity_discovery"] is False
    assert first["content_sha256"] == memory._self_hash(first)
    assert set(first_artifacts) == {
        "kernel-frequency-signatures.csv",
        "low-rate-loop-discriminator.csv",
        "gw150914-nr-response-blind-predictions.csv",
        "equivalence-and-discriminator-map.csv",
        "ranked-leads.md",
        "low-rate-loop-discriminator.svg",
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("claim_boundary", "observational_response_opened", True),
        ("claim_boundary", "real_data_fit_performed", True),
        ("claim_boundary", "historical_novelty_established", True),
        ("claim_boundary", "empirical_gravity_discovery", True),
        ("access_ledger", "observational_response_files_opened", 1),
        ("access_ledger", "post_response_formula_changes", 1),
    ],
)
def test_claim_or_response_access_mutation_fails_closed(
    section: str, key: str, value: object
) -> None:
    config = copy.deepcopy(_config_unsealed())
    config[section][key] = value
    with pytest.raises(memory.DynamicMemoryError):
        memory.validate_config(config)


def test_rehashed_receipt_overclaim_fails_closed() -> None:
    config = memory.load_config(ROOT)
    receipt, _ = memory.derive_package(ROOT)
    receipt["claim_boundary"]["historical_novelty_established"] = True
    receipt["content_sha256"] = memory._self_hash(receipt)
    with pytest.raises(memory.DynamicMemoryError, match="claims changed"):
        memory.validate_receipt(receipt, config)


def test_atomic_no_clobber_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    assert memory._atomic_bytes_no_clobber(path, b"one") == "CREATED"
    assert memory._atomic_bytes_no_clobber(path, b"one") == "EXISTING_IDENTICAL"
    with pytest.raises(memory.DynamicMemoryError, match="refusing to replace"):
        memory._atomic_bytes_no_clobber(path, b"two")


def test_source_acquisition_receipt_is_bound_and_response_free() -> None:
    receipt = json.loads((ROOT / memory.SOURCE_RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["sha256"] == memory._sha256_file(ROOT / memory.SOURCE_PATH)
    assert receipt["network_calls"] == 1
    assert receipt["observational_response"] is False
    assert receipt["response_values_opened"] == 0
    assert receipt["content_sha256"] == memory._self_hash(receipt)


def test_frozen_response_preflight_has_exact_products_and_no_per_event_scale() -> None:
    preflight = _config_unsealed()["observational_response_preflight"]
    assert preflight["state"] == "DIRECT_URLS_FROZEN_RESPONSES_NOT_OPENED"
    assert [row["expected_http_bytes"] for row in preflight["products"]] == [
        1040592,
        1007420,
    ]
    assert all(row["url"].endswith(".hdf5") for row in preflight["products"])
    assert "no per-event kernel timescale or threshold" in preflight["frozen_nuisance_projection"]
    assert len(preflight["required_acquisition_receipt"]) == 5


def test_one_frequency_is_not_mistaken_for_identification() -> None:
    amplitude = 1.7
    phase = -0.43
    nuisance = amplitude * np.exp(1j * phase)
    for transfer in (
        0.4 + 0.1j,
        -0.2 + 0.7j,
        1.0 - 0.8j,
    ):
        fitted_amplitude = abs(transfer / nuisance)
        fitted_phase = np.angle(transfer / nuisance)
        reconstructed = nuisance * fitted_amplitude * np.exp(1j * fitted_phase)
        assert reconstructed == pytest.approx(transfer, abs=3.0e-16)
    assert math.isfinite(fitted_phase)
