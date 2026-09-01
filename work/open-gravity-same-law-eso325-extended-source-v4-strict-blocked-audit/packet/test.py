from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_same_law_eso325_extended_source_v4 as subject


def test_contract_was_frozen_before_scientific_array_decode() -> None:
    config = subject.load_config()
    source = config["source_freeze"]
    assert config["status"] == "FROZEN_BEFORE_SCIENTIFIC_ARRAY_DECODE"
    assert source["target"]["role"] == "DEVELOPMENT_ONLY"
    assert source["scientific_arrays_decoded_at_freeze"] == 0
    assert source["scientific_response_rows_opened_at_freeze"] == 0
    assert config["slacs_confirmation_seal"]["reserved_confirmation"] == 12
    assert config["slacs_confirmation_seal"]["confirmation_response_values_read_by_v4"] == 0


def test_exact_local_sources_and_append_only_predecessor_are_hash_bound() -> None:
    result = subject.verify_frozen_sources(subject.load_config())
    assert result["all_exact_sources_pass"] is True
    assert len(result["source_rows"]) == 6
    assert sum(row["bytes"] for row in result["source_rows"]) == 8_119_967_556
    assert result["sealed_response_manifest_deserialized"] is False
    assert len(result["sealed_manifest_hashes_only"]) == 3


def test_fits_preflight_reads_structure_only_and_skips_arrays() -> None:
    result = subject.structural_source_preflight(subject.load_config())
    assert result["status"] == "PASS_EXACT_BYTES_AND_FITS_STRUCTURE_ONLY"
    assert result["scientific_arrays_decoded"] == 0
    assert result["scientific_response_rows_opened"] == 0
    assert len(result["structures"]) == 4
    for structure in result["structures"]:
        assert structure["block_aligned"] is True
        assert structure["hdu_count"] >= 1
        assert structure["scientific_array_elements_decoded"] == 0
        assert sum(row["data_bytes_skipped"] for row in structure["hdus"]) > 0


def test_paper_corrects_voronoi_assumption_and_contract_fails_closed() -> None:
    reduction = subject.load_config()["reduction_contract"]
    assert "not Voronoi" in reduction["muse"]["correction_to_prior_plan"]
    assert reduction["muse"]["spatial_bins"].startswith("Non-overlapping 0.6x0.6")
    assert "Do not fill" in reduction["fail_closed_rule"]
    assert "positive-semidefinite" in reduction["muse"]["noise_and_covariance"]


def test_one_state_feeds_matter_and_photon_routes_without_photon_knob() -> None:
    coordinates, density, cell = subject.synthetic_density(25, 8.0)
    state = subject.solve_extended_state(
        density, coordinates, cell, g=0.18, range_kpc=3.0
    )
    matter = subject.matter_observable(state, np.array([[1.0, 0.0, 0.0]]))
    lens = subject.photon_observable(state, np.array([[1.0, 0.0]]))
    assert matter.shape == (1, 3)
    assert lens.shape == (1, 2)
    assert np.all(np.isfinite(matter))
    assert np.all(np.isfinite(lens))
    assert list(inspect.signature(subject.matter_observable).parameters) == ["state", "points"]
    assert list(inspect.signature(subject.photon_observable).parameters) == ["state", "points_xy"]
    assert list(inspect.signature(subject.extended_source_image).parameters) == [
        "state",
        "image_coordinates",
    ]


def test_gr_limit_uses_identical_density_and_routes() -> None:
    coordinates, density, cell = subject.synthetic_density(25, 8.0)
    state = subject.solve_extended_state(density, coordinates, cell, g=0.0, range_kpc=3.0)
    assert np.array_equal(state.Phi, state.U)
    assert np.array_equal(state.Psi, state.U)


def test_target_free_injection_limits_and_countermodels_pass() -> None:
    result = subject.target_free_gate(subject.load_config())
    metrics = result["metrics"]
    assert result["status"] == "PASS_TARGET_FREE_SHARED_STATE"
    assert result["pass"] is True
    assert metrics["relative_mass_error"] <= 1e-12
    assert metrics["gr_state_max_absolute_error"] == 0.0
    assert metrics["short_range_relative_state_error"] <= 1e-6
    assert metrics["reflection_relative_error"] <= 5e-10
    assert metrics["axis_permutation_relative_error"] <= 5e-10
    assert metrics["grid_convergence_relative_rms"] <= 0.08
    assert metrics["recovered_g_at_frozen_range"] == pytest.approx(0.18, abs=1e-10)
    assert metrics["matter_gr_relative_residual"] > 0
    assert metrics["lensing_gr_relative_residual"] > 0
    assert metrics["extended_image_gr_relative_residual"] > 0
    assert metrics["common_mass_rescale_joint_relative_residual"] > 0
    assert result["retained_countermodels"]["SPLIT_STATE"]["status"].startswith("FORBIDDEN")


def test_source_readiness_blocks_before_array_decode_without_inventing_covariance() -> None:
    result = subject.source_readiness(subject.load_config())
    assert result["status"] == "BLOCK_BEFORE_SCIENTIFIC_ARRAY_DECODE"
    assert result["scientific_arrays_decoded"] == 0
    assert result["eso_development_score_computed"] is False
    assert result["slacs_confirmation_opened"] is False
    assert "revision-bound MILES template payloads" in result["missing_or_unverified"]


def test_adversarial_scope_mutations_fail_closed() -> None:
    mutations = (
        lambda value: value.__setitem__("status", "READY"),
        lambda value: value["source_freeze"]["target"].__setitem__("role", "CONFIRMATION"),
        lambda value: value["shared_physical_state"]["forbidden"].remove("photon_multiplier"),
        lambda value: value["slacs_confirmation_seal"].__setitem__(
            "confirmation_response_values_read_by_v4", 1
        ),
        lambda value: value["access_contract"].__setitem__("confirmation_targets_opened", 1),
    )
    for mutation in mutations:
        forged = copy.deepcopy(subject.load_config())
        mutation(forged)
        with pytest.raises(subject.SameLawESO325V4Error):
            subject.validate_config(forged)


def test_receipt_is_target_free_pass_but_real_data_block() -> None:
    receipt = subject.build_receipt()
    assert receipt["status"] == "SOURCE_BLOCKED_ESO_REDUCTION_INPUTS_AFTER_TARGET_FREE_PASS"
    assert receipt["decision"].startswith("KEEP_ESO_SCIENTIFIC_ARRAYS")
    assert receipt["access_accounting"]["scientific_fits_array_elements_decoded"] == 0
    assert receipt["access_accounting"]["eso_scores_computed"] == 0
    assert receipt["access_accounting"]["slacs_confirmation_response_values_opened"] == 0
    assert receipt["target_free_gate"]["pass"] is True
    assert receipt["source_readiness"]["status"].startswith("BLOCK_")


def test_artifacts_are_deterministic_and_counterexamples_are_retained() -> None:
    config = subject.load_config()
    first = subject.build_artifacts(config)
    second = subject.build_artifacts(config)
    assert first == second
    assert set(first) == {
        "exact-source-receipt.json",
        "fits-structure-only.json",
        "target-free-shared-state-gate.json",
        "source-readiness.json",
        "frozen-reduction-and-analysis-contract.json",
        "report.md",
    }
    gate = json.loads(first["target-free-shared-state-gate.json"])
    assert set(gate["retained_countermodels"]) == {"GR", "COMMON_MASS_RESCALE", "SPLIT_STATE"}
    assert b"0.6-arcsec square pixels" in first["report.md"]
    assert b"refuses to invent covariance" in first["report.md"]


def test_packet_matches_deterministic_rebuild_if_present() -> None:
    if not subject.OUTPUT_PATH.exists():
        pytest.skip("packet is built after unit tests")
    observed = json.loads(subject.OUTPUT_PATH.read_text(encoding="utf-8"))
    assert observed == subject.build_receipt()
    for name, payload in subject.build_artifacts(subject.load_config()).items():
        assert (subject.ARTIFACT_DIRECTORY / name).read_bytes() == payload


def test_source_anomaly_is_retained_not_silently_substituted() -> None:
    anomaly = subject.load_config()["source_freeze"]["source_anomalies_retained"][0]
    assert anomaly["same_bytes"] == 1_108_800
    assert anomaly["alternate_sha256"] != "8cf81fdc7f93e285444f7a83bce57cfcd974f165e6e9aa9b25093eb09a35f6e6"
    assert Path(subject.REPOSITORY_ROOT / anomaly["alternate_local_path"]).is_file()
