from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.gravity_g4_cluster_lensing_exploration as cluster_lensing
from sigma_theory_compiler.gravity_g4_first_principles_mechanism_search import mechanism_specs
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / cluster_lensing.OUTPUT_PATH


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


def test_source_is_hash_bound_and_assembles_20_monotone_cluster_profiles() -> None:
    config = cluster_lensing.load_config(ROOT)
    packets = cluster_lensing.prepare_packets(ROOT, config)
    assert len(packets) == 20
    assert sum(len(packet["gbar"]) for packet in packets) == 84
    assert len({record for packet in packets for record in packet["records"]}) == 84
    for packet in packets:
        radius = np.asarray(packet["arrays"]["radius"])
        assert np.all(np.diff(radius) > 0)
        assert np.all(np.asarray(packet["gbar"]) > 0)
        assert np.all(np.asarray(packet["sigma_log_gtot"]) > 0)


def test_transfer_excludes_stellar_density_but_retains_nine_mechanism_lanes() -> None:
    eligible = [spec for spec in mechanism_specs() if cluster_lensing._eligible_spec(spec)]
    creative = [spec for spec in eligible if spec["role"] == "mechanism"]
    assert len(eligible) == 185
    assert len(creative) == 184
    assert len({spec["lane"] for spec in creative}) == 9
    assert all(spec.get("source") != "stellar_surface_density" for spec in eligible)
    assert all(spec["lane"] != "geometry_directed_gravity" for spec in eligible)
    assert [spec["candidate_id"] for spec in eligible].count(
        "known-control:exact-empirical-rar-rewrite"
    ) == 1


def test_receipt_rebuilds_exactly_and_keeps_direct_lensing_gate_closed() -> None:
    stored = _load(OUTPUT)
    rebuilt = cluster_lensing.build_receipt(ROOT)
    assert rebuilt == stored
    cluster_lensing.validate_receipt(stored, root=ROOT)
    assert stored["decision"] == "BLOCK_CROSS_SCALE_ACTION_CLUSTER_LENSING_EXPLORATION"
    assert stored["counts"] == {
        "clusters": 20,
        "coefficient_cells_per_selection": 2392,
        "creative_mechanism_specs": 184,
        "direct_image_or_shear_likelihood_evaluations": 0,
        "eligible_mechanism_lanes": 9,
        "known_control_specs": 1,
        "outer_folds": 5,
        "radial_points": 84,
        "scoring_point_evaluations": 1205568,
        "source_compatible_specs_total": 185,
    }
    assert stored["claims"]["fixed_D3_projection_diagnostically_tested"] is True
    assert stored["claims"]["whole_cluster_transfer_explored"] is True
    assert stored["claims"]["direct_lensing_test_completed"] is False
    assert stored["gate_checks"]["direct_lensing_observable_likelihood"] is False
    assert stored["gate_checks"]["known_RAR_control_recovered_exactly"] is True
    assert stored["gate_checks"]["same_structure_as_fixed_galaxy_parent_in_all_outer_folds"] is True
    assert stored["gate_checks"]["same_coefficient_as_fixed_galaxy_parent"] is False
    assert (
        float(stored["mechanism_transfer"]["cluster_selected_to_galaxy_parent_beta_ratio"]) == 4.0
    )


def test_whole_cluster_folds_are_disjoint_and_every_cluster_is_held_out_once() -> None:
    receipt = _load(OUTPUT)
    ledger = receipt["mechanism_transfer"]["outer_ledger"]
    heldout = [cluster for fold in ledger for cluster in fold["heldout_clusters"]]
    assert len(heldout) == 20
    assert len(set(heldout)) == 20
    assert sum(fold["heldout_score"]["points"] for fold in ledger) == 84
    assert all(fold["training_clusters"] + len(fold["heldout_clusters"]) == 20 for fold in ledger)


def test_fixed_action_has_zero_cluster_fitted_parameters_and_unique_predictions() -> None:
    receipt = _load(OUTPUT)
    fixed = receipt["fixed_D3_action"]
    assert fixed["parameters_fit_to_cluster_data"] == 0
    assert fixed["support_dimension"] == 3
    assert len(fixed["prediction_manifest_sha256"]) == 64
    assert (
        fixed["prediction_manifest_sha256"]
        != receipt["mechanism_transfer"]["nested_oof_prediction_manifest_sha256"]
    )
    parent = receipt["mechanism_transfer"]["fixed_galaxy_parent_cluster_diagnostic"]
    assert parent["parameters_fit_to_cluster_data"] == 0
    assert float(parent["beta"]) == 0.5
    assert parent["candidate_id"] == ("cross-scale:y:q0p1:ell0p25:permittivity_plus_auxiliary")


@pytest.mark.parametrize(
    "claim",
    [
        "alternative_to_gr_confirmed",
        "covariant_lensing_equation_derived",
        "direct_cluster_thermodynamic_test_completed",
        "direct_lensing_test_completed",
        "historical_novelty_established",
        "sequential_G6_G7_G8_advanced",
    ],
)
def test_resealed_overclaim_is_rejected(claim: str) -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["claims"][claim] = True
    with pytest.raises(cluster_lensing.GravityG4ClusterLensingError):
        cluster_lensing.validate_receipt(_reseal(receipt), root=ROOT)


def test_resealed_direct_lensing_open_or_source_rebinding_is_rejected() -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["gate_checks"]["direct_lensing_observable_likelihood"] = True
    with pytest.raises(cluster_lensing.GravityG4ClusterLensingError):
        cluster_lensing.validate_receipt(_reseal(receipt), root=ROOT)
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["source_bindings"]["expected_radial_points"] = 83
    with pytest.raises(cluster_lensing.GravityG4ClusterLensingError):
        cluster_lensing.validate_receipt(_reseal(receipt), root=ROOT)


def test_lensing_target_never_enters_baryonic_packet_features() -> None:
    config = cluster_lensing.load_config(ROOT)
    packets = cluster_lensing.prepare_packets(ROOT, config)
    for packet in packets:
        assert set(packet["features"]) == {"log1p_sb_total", "log_y"}
        assert "log_gtot" not in packet["features"]
        assert "sigma_log_gtot" not in packet["features"]
