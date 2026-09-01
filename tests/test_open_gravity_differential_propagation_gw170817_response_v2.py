from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_response_v2 as response,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return response.load_config()


def _copy_freeze_package(tmp_path: Path) -> Path:
    config = json.loads((ROOT / response.CONFIG_PATH).read_text(encoding="utf-8"))
    relatives = [response.CONFIG_PATH, response.MODULE_PATH, response.TEST_PATH]
    for predecessor in config["predecessors"].values():
        relatives.extend(
            Path(predecessor[f"{role}_path"]) for role in ("config", "module", "test", "receipt")
        )
    for relative in relatives:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_strict_audit_preserves_theorem_and_exact_preflight_products() -> None:
    config = _config()
    assert config["official_metadata_audit"]["status"] == (
        "PASS_STRICT_PREDECESSOR_AND_OFFICIAL_METADATA_AUDIT"
    )
    assert config["official_metadata_audit"]["observational_payload_bytes_opened_during_audit"] == 0
    assert config["claim_boundary"]["theorem_modified"] is False
    assert [row["detector"] for row in config["products"]] == ["H1", "L1", "V1"]
    assert [row["expected_bytes"] for row in config["products"]] == [
        125217658,
        124266501,
        129470892,
    ]
    assert [row["published_md5"] for row in config["products"]] == [
        "1a1cca3fb28686d5798539468a99dbae",
        "dbbde824db6df6a9f653db374fc5c88c",
        "8ea80f93257a292d82f0af497e2a4cff",
    ]
    assert all("LOSC_C00_4_V1" in row["filename"] for row in config["products"])


def test_preprocessing_and_l1_official_glitch_gate_are_exact() -> None:
    config = _config()
    preprocessing = config["preprocessing"]
    assert preprocessing["analysis_gps_start"] == 1187008756
    assert preprocessing["analysis_duration_seconds"] == 128
    assert preprocessing["psd_intervals_gps"] == [
        [1187007090, 1187007602],
        [1187010162, 1187010674],
    ]
    gate = preprocessing["l1_glitch_gate"]
    times = np.asarray(
        [
            gate["center_gps"] - 0.61,
            gate["center_gps"] - 0.60,
            gate["center_gps"] - 0.10,
            gate["center_gps"],
            gate["center_gps"] + 0.10,
            gate["center_gps"] + 0.60,
            gate["center_gps"] + 0.61,
        ]
    )
    window = response._glitch_gate(times, gate)
    assert window[[0, 1, 5, 6]].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert window[[2, 3, 4]].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_target_free_response_controls_recover_injections_and_freeze_null_thresholds() -> None:
    controls = response.target_free_controls(_config())
    assert all(row["passed"] for row in controls["zero_noise"])
    assert [row["family"] for row in controls["zero_noise"]] == [
        "GR",
        "PHASE_INVERSE_FREQUENCY",
        "PHASE_CUBIC_FREQUENCY",
        "ATTENUATION_LINEAR_FREQUENCY",
    ]
    assert controls["null_realizations"] == 64
    assert all(
        math.isfinite(value) and value >= 0.0
        for value in controls["family_delta_2_log_likelihood_thresholds"].values()
    )
    assert len(controls["control_sha256"]) == 64
    assert controls == response.target_free_controls(_config())


def test_phase_bases_remove_constant_and_linear_components() -> None:
    frequencies = np.linspace(30.0, 300.0, 541)
    design = np.column_stack((np.ones(frequencies.size), frequencies / 100.0))
    weights = frequencies ** (-7.0 / 3.0)
    for family in ("PHASE_INVERSE_FREQUENCY", "PHASE_CUBIC_FREQUENCY"):
        basis = response._phase_basis(frequencies, family, 100.0)
        moments = design.T @ (weights * basis)
        assert np.max(np.abs(moments)) < 1e-10
        assert np.max(np.abs(basis)) == pytest.approx(1.0)


def test_synthetic_gr_control_recovers_frozen_mass_and_time() -> None:
    config = _config()
    frequencies = np.linspace(30.0, 300.0, 136)
    chirp_mass = 1.1975
    eta = 0.245
    base_time = (
        config["official_metadata"]["event_gps"] - config["preprocessing"]["analysis_gps_start"]
    )
    template = response._taylorf2(frequencies, chirp_mass, eta, 1.0) * np.exp(
        -2j * np.pi * frequencies * base_time
    )
    delta_f = frequencies[1] - frequencies[0]
    weight = np.full(frequencies.size, 4.0 * delta_f)
    amplitude = 20.0 / math.sqrt(float(np.sum(np.abs(template) ** 2 * weight)))
    processed = [
        {
            "detector": detector,
            "frequencies": frequencies,
            "spectrum": amplitude * template,
            "psd": np.ones(frequencies.size),
            "delta_f": delta_f,
        }
        for detector in ("H1", "L1", "V1")
    ]
    recovered = response._gr_recovery(config, processed)
    assert recovered["passed"] is True
    assert recovered["best"]["chirp_mass_solar"] == pytest.approx(chirp_mass)
    assert recovered["best"]["symmetric_mass_ratio"] == pytest.approx(eta)
    assert recovered["best"]["fourier_phase_sign"] == 1
    assert all(
        row["coalescence_offset_seconds"] == pytest.approx(0.0)
        for row in recovered["best"]["detectors"]
    )


def test_config_mutations_fail_closed_before_payload_access() -> None:
    config = _config()
    wrong_product = copy.deepcopy(config)
    wrong_product["products"][2]["filename"] = "substitute.hdf5"
    with pytest.raises(response.GW170817ResponseError):
        response.validate_config(wrong_product)

    wrong_gate = copy.deepcopy(config)
    wrong_gate["preprocessing"]["l1_glitch_gate"]["center_gps"] += 0.1
    with pytest.raises(response.GW170817ResponseError, match="glitch gate"):
        response.validate_config(wrong_gate)

    access = copy.deepcopy(config)
    access["access_contract"]["observational_payload_bytes_before_freeze"] = 1
    with pytest.raises(response.GW170817ResponseError, match="access contract"):
        response.validate_config(access)


def test_prediction_receipt_freezes_code_grids_and_controls_before_data(tmp_path: Path) -> None:
    base = _copy_freeze_package(tmp_path)
    assert response.freeze(base) == "CREATED"
    assert response.freeze(base) == "EXISTING_IDENTICAL"
    config = response.load_config(base)
    prediction = response.validate_prediction(config, base)
    assert prediction["status"] == "FROZEN_BEFORE_OBSERVATIONAL_PAYLOAD_ACCESS"
    assert prediction["access_at_freeze"]["observational_payload_bytes_before_freeze"] == 0
    assert all(row["passed"] for row in prediction["target_free_controls"]["zero_noise"])
    assert len(prediction["preprocessing_sha256"]) == 64
    assert len(prediction["response_likelihood_sha256"]) == 64


def test_results_are_required_to_remain_separate_and_narrow() -> None:
    config = _config()
    boundary = config["claim_boundary"]
    assert boundary["source_metadata_result_separate"] is True
    assert boundary["real_data_result_separate"] is True
    assert boundary["approximate_GR_control_only"] is True
    assert boundary["published_GR_parameter_estimation_reproduced"] is False
    assert boundary["fundamental_parameter_posterior"] is False
    assert boundary["publication_ready"] is False
    assert (
        "not an exact Gamma-only"
        in config["response_likelihood"]["families"][2]["theorem_relation"]
    )
