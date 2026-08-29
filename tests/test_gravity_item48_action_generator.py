from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item48_action_generator import (
    action_bank_from_arrays,
    action_catalog,
    admissible_candidates,
    build_action_features,
    build_candidate_manifest,
    build_derivation_receipt,
    build_exposure_manifest,
    build_aggregate_result,
    build_evaluation_result,
    decode_candidate,
    load_config,
    malformed_action_controls,
    symbolic_action_derivation,
)


ROOT = Path(__file__).resolve().parents[1]


def test_item48_config_and_equal_action_capacity() -> None:
    config = load_config(ROOT)
    catalog = action_catalog(config)
    assert len(catalog) == 96
    assert len({row["action_class"] for row in catalog}) == 6
    assert all(
        sum(row["action_class"] == name for row in catalog) == 16
        for name in config["action_generator"]["action_classes"]
    )
    admitted, audit = admissible_candidates(config)
    assert audit["raw_candidates"] == 393216
    assert audit["admitted_candidates"] == len(admitted["candidate_id"])
    assert len(set(audit["admitted_by_action_class"].values())) == 1
    assert audit["admitted_per_action_recipe"] > 0


def test_symbolic_action_variation_and_malformed_controls() -> None:
    derivation = symbolic_action_derivation()
    assert derivation["all_exact_euler_residuals_zero"]
    assert derivation["all_source_free_shift_identities_zero"]
    assert len(derivation["action_classes"]) == 6
    controls = malformed_action_controls()
    assert controls["all_malformed_controls_rejected"]
    assert all(not row["admitted"] for row in controls["controls"])


def test_discrete_action_solvers_are_convex_and_exact() -> None:
    config = load_config(ROOT)
    radius = np.logspace(-2.0, 1.0, 24)
    mass = np.square(radius) / (1.0 + np.square(radius))
    arrays = {
        "population": np.asarray(["S4TM"]),
        "object": np.asarray(["synthetic-source-only"]),
        "fold": np.asarray([0]),
        "radius": np.asarray([1.0]),
        "size": np.asarray([1.0]),
        "redshift": np.asarray([0.1]),
        "u": np.asarray([0.2]),
    }
    # Replace the profile helper's analytic profile contract with a direct source-only
    # check by testing the same profile at all nodes under one synthetic object.
    arrays = {
        **arrays,
        "population": np.repeat("CLASH", len(radius)),
        "object": np.repeat("synthetic-source-only", len(radius)),
        "fold": np.zeros(len(radius), dtype=int),
        "radius": radius,
        "size": np.ones(len(radius)),
        "redshift": np.repeat(0.1, len(radius)),
        "u": mass / np.square(radius),
    }
    raw, bank, audit = action_bank_from_arrays(arrays, config)
    assert raw.shape == (len(radius), 96)
    assert bank.shape == (len(radius), 96)
    assert np.all((bank > 0.0) & (bank < 1.0))
    assert audit["maximum_relative_discrete_euler_residual"] <= 1e-8
    assert audit["minimum_reduced_hessian_eigenvalue"] > 0.0


def test_response_blind_features_and_recorded_freeze_receipts() -> None:
    features = build_action_features(ROOT)
    assert features["response_fields_read_by_feature_builder"] == []
    assert features["response_values_used"] == 0
    assert features["counts"]["action_recipes"] == 96
    assert features["dataset_behavior"]["unique_action_coordinate_hashes"] == 96
    assert features["solver_audit"]["maximum_relative_discrete_euler_residual"] <= 1e-8
    config = load_config(ROOT)
    source = ROOT / config["paths"]["source_dir"]
    assert json.loads((source / config["paths"]["candidate_manifest"]).read_text()) == build_candidate_manifest(ROOT)
    assert json.loads((source / config["paths"]["derivation_receipt"]).read_text()) == build_derivation_receipt(ROOT)
    assert json.loads((source / config["paths"]["exposure_manifest"]).read_text()) == build_exposure_manifest(ROOT)


def test_candidate_decoding_preserves_action_provenance() -> None:
    config = load_config(ROOT)
    decoded = decode_candidate(5 * 65536, config)
    assert decoded["action_class"] == "adaptive_gradient_auxiliary"
    assert decoded["creativity_label"] == "potentially_new_observational_synthesis"
    assert decoded["historical_novelty_claimed"] is False
    assert decoded["derived_flux_equation"] == "epsilon_c*g=g_bar"


def test_recorded_result_retains_action_without_promotion() -> None:
    config = load_config(ROOT)
    source = ROOT / config["paths"]["source_dir"]
    evaluation = json.loads(
        (source / config["paths"]["evaluation_result"]).read_text(encoding="utf-8")
    )
    aggregate = json.loads(
        (ROOT / config["paths"]["aggregate_result"]).read_text(encoding="utf-8")
    )
    assert evaluation == build_evaluation_result(ROOT)
    assert aggregate == build_aggregate_result(ROOT)
    assert evaluation["selected_candidate"]["candidate_id"] == 245512
    assert evaluation["selected_candidate"]["action_class"] == "mixed_two_field"
    assert {
        row["selected_action"]["candidate_id"] for row in evaluation["fold_ledger"]
    } == {245512}
    assert evaluation["scores"]["action_generator"]["balanced_loss"] < evaluation[
        "scores"
    ]["ordinary_ridge"]["balanced_loss"]
    assert evaluation["scores"]["action_generator"]["balanced_loss"] > evaluation[
        "scores"
    ]["item45_universal_interaction"]["balanced_loss"]
    assert aggregate["decision"] == "NONPROMOTED_ITEM48_ACTION_RESULT_RETAINED"
    assert aggregate["claims"]["formula_family_pruned"] is False
    assert aggregate["formal_scope"]["covariant_completion"] is False
