"""Derive and test the fold-stable cross-scale parent from one radial action."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from .gravity_g0_experiment import score_predictions
from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .gravity_g4_first_principles_mechanism_search import (
    validate_receipt as validate_predecessor_receipt,
)
from .gravity_g4_nonlocal_profile_law_construction import (
    _kernel_matrix,
    _log_radius_cell_widths,
    prepare_nonlocal_packets,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-g4-auxiliary-action-derivation-receipt-6.0"
CONFIG_SCHEMA = "invariant-gravity-g4-auxiliary-action-derivation-config-6.0"
CONFIG_PATH = "configs/gravity_g4_auxiliary_action_derivation.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g4_auxiliary_action_derivation.py"
TEST_PATH = "tests/test_gravity_g4_auxiliary_action_derivation.py"
OUTPUT_PATH = "runs/gravity/g4/auxiliary-action-derivation-v6.json"


class GravityG4AuxiliaryActionError(ValueError):
    """The auxiliary-action contract, derivation, or evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load and validate the frozen post-v5 action-derivation contract."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG4AuxiliaryActionError("auxiliary-action config changed")
    binding = config.get("predecessor_binding", {})
    path = root / str(binding.get("path"))
    if _file_sha256(path) != binding.get("file_sha256"):
        raise GravityG4AuxiliaryActionError("auxiliary-action predecessor file changed")
    predecessor = _load_json(path)
    validate_predecessor_receipt(predecessor, root=root)
    if predecessor.get("content_sha256") != binding.get("content_sha256") or predecessor.get(
        "decision"
    ) != binding.get("required_decision"):
        raise GravityG4AuxiliaryActionError("auxiliary-action predecessor content changed")
    best = predecessor.get("lane_results", [])[0].get("final_all_exploration_selection", {})
    if (
        best.get("candidate_id") != binding.get("required_parent_candidate_id")
        or float(best.get("beta")) != float(binding.get("required_parent_beta"))
        or abs(float(best.get("chi_square")) - float(binding.get("required_parent_chi_square")))
        > 1e-7
    ):
        raise GravityG4AuxiliaryActionError("fold-stable parent changed")
    closure = config.get("dimensional_closure_hypothesis", {})
    dimension = int(closure.get("effective_baryonic_support_dimension", 0))
    if (
        dimension != 2
        or float(closure.get("predicted_beta")) != 1.0 / dimension
        or float(closure.get("predicted_log_radius_scale")) != 1.0 / (2.0 * dimension)
        or closure.get("velocity_data_used_to_choose_D") is not False
        or closure.get("post_v5_hypothesis") is not True
        or closure.get("historical_novelty_claimed") is not False
    ):
        raise GravityG4AuxiliaryActionError("dimensional closure changed")
    inherited = config.get("inherited_unresolved_quantity", {})
    if (
        inherited.get("id") != "y_transition"
        or float(inherited.get("value")) != 0.1
        or inherited.get("may_count_as_first_principles") is not False
    ):
        raise GravityG4AuxiliaryActionError("transition inheritance changed")
    if tuple(int(value) for value in config.get("counterfactual_dimensions", ())) != (
        1,
        2,
        3,
    ):
        raise GravityG4AuxiliaryActionError("dimension counterfactuals changed")
    population = config.get("population", {})
    if any(
        population.get(key) != 0
        for key in (
            "confirmation_evaluator_accesses_allowed",
            "cluster_evaluator_accesses_allowed",
            "lensing_evaluator_accesses_allowed",
        )
    ):
        raise GravityG4AuxiliaryActionError("auxiliary-action opens downstream data")
    if any(
        config.get("claim_boundaries", {}).get(key) is not False
        for key in (
            "effective_radial_action_is_covariant_theory",
            "directed_log_radius_constraint_is_causal_time_evolution",
            "thin_disk_dimension_rule_is_derived_fundamental_law",
            "parent_selection_is_independent_confirmation",
            "historical_novelty_claimed",
        )
    ):
        raise GravityG4AuxiliaryActionError("auxiliary-action claim boundary changed")
    return config


def symbolic_derivation() -> dict[str, Any]:
    """Verify the two auxiliary equations and integrated flux relation exactly."""

    x = sp.symbols("x", real=True)
    dimension, ell, a0, gbar = sp.symbols("D ell a0 g_bar", positive=True)
    psi = sp.Function("psi")(x)
    chi = sp.Function("chi")(x)
    eta = sp.Function("eta")(x)
    q = sp.Function("q")(x)

    screened_lagrangian = psi**2 / 2 + ell**2 * sp.diff(psi, x) ** 2 / 2 - q * psi
    screened_euler = sp.simplify(
        sp.diff(screened_lagrangian, psi)
        - sp.diff(sp.diff(screened_lagrangian, sp.diff(psi, x)), x)
    )
    screened_expected = psi - q - ell**2 * sp.diff(psi, x, 2)
    screened_residual = sp.simplify(screened_euler - screened_expected)

    directed_lagrangian = eta * (chi + ell * sp.diff(chi, x) - q)
    directed_euler_eta = sp.simplify(sp.diff(directed_lagrangian, eta))
    directed_expected = chi + ell * sp.diff(chi, x) - q
    directed_residual = sp.simplify(directed_euler_eta - directed_expected)

    psi_symbol, chi_symbol = sp.symbols("psi chi", real=True)
    permittivity = 1 / (1 + chi_symbol / dimension)
    auxiliary_flux = a0 * psi_symbol / (dimension * (1 + chi_symbol / dimension))
    derived_acceleration = gbar * (1 + chi_symbol / dimension) + (a0 * psi_symbol / dimension)
    flux_residual = sp.factor(permittivity * derived_acceleration - auxiliary_flux - gbar)

    dimension_value = sp.Integer(2)
    beta = sp.simplify(1 / dimension_value)
    scale = sp.simplify(1 / (2 * dimension_value))
    quadratic_hessian = sp.hessian(
        sp.Symbol("psi0") ** 2 / 2 + ell**2 * sp.Symbol("psi1") ** 2 / 2,
        (sp.Symbol("psi0"), sp.Symbol("psi1")),
    )
    return {
        "derived_acceleration": str(derived_acceleration),
        "dimension_closure": {
            "D": 2,
            "beta": str(beta),
            "ell": str(scale),
        },
        "directed_constraint_euler_equation": str(directed_euler_eta),
        "directed_constraint_residual": str(directed_residual),
        "flux_identity_residual": str(flux_residual),
        "permittivity": str(permittivity),
        "quadratic_auxiliary_hessian": [
            [str(quadratic_hessian[row, column]) for column in range(2)] for row in range(2)
        ],
        "screened_euler_equation": str(screened_euler),
        "screened_euler_residual": str(screened_residual),
        "all_exact_residuals_zero": all(
            value == 0 for value in (screened_residual, directed_residual, flux_residual)
        ),
    }


def action_prediction2(packet: Mapping[str, Any], *, support_dimension: int) -> dict[str, Any]:
    """Evaluate the boundary-normalized discrete action solution from baryons only."""

    if support_dimension <= 0:
        raise GravityG4AuxiliaryActionError("support dimension must be positive")
    dimension = float(support_dimension)
    beta = 1.0 / dimension
    scale = 1.0 / (2.0 * dimension)
    radius = np.asarray(packet["arrays"]["radius"], dtype=np.float64)
    vbar2 = np.asarray(packet["arrays"]["vbar2"], dtype=np.float64)
    y = vbar2 / (radius * float(packet["a0"]))
    q = y / (y + 0.1)
    log_radius = np.log(radius)
    widths = _log_radius_cell_widths(log_radius)
    interior = _kernel_matrix(log_radius, widths, "interior_exponential", scale) @ q
    symmetric = _kernel_matrix(log_radius, widths, "symmetric_exponential", scale) @ q
    prediction2 = vbar2 + beta * (vbar2 * interior + radius * float(packet["a0"]) * symmetric)
    if np.any(~np.isfinite(prediction2)):
        raise GravityG4AuxiliaryActionError("action prediction is non-finite")
    return {
        "beta": beta,
        "chi": interior,
        "ell": scale,
        "prediction2": prediction2,
        "psi": symmetric,
        "q": q,
    }


def _score_dimension(
    packets: Sequence[Mapping[str, Any]], support_dimension: int
) -> dict[str, Any]:
    chi_square = 0.0
    invalid = 0
    galaxy_rows = []
    prediction_hash_rows = []
    for packet in packets:
        action = action_prediction2(packet, support_dimension=support_dimension)
        prediction2 = np.asarray(action["prediction2"])
        invalid += int(np.sum(prediction2 <= 0))
        prediction = np.sqrt(np.maximum(prediction2, np.finfo(np.float64).tiny))
        score = score_predictions(prediction, packet["arrays"]["vobs"], packet["arrays"]["sigma"])
        chi_square += float(score["chi_square"])
        prediction_hash = canonical_sha256([format(float(value), ".15e") for value in prediction])
        prediction_hash_rows.append(
            {"galaxy": packet["galaxy"].name, "prediction_sha256": prediction_hash}
        )
        galaxy_rows.append(
            {
                "action_score": score,
                "galaxy": packet["galaxy"].name,
                "point_count": packet["galaxy"].count,
                "rar_score": score_predictions(
                    np.sqrt(packet["rar2"]),
                    packet["arrays"]["vobs"],
                    packet["arrays"]["sigma"],
                ),
            }
        )
    return {
        "beta": _metric(1.0 / support_dimension),
        "chi_square": _metric(chi_square),
        "ell": _metric(1.0 / (2.0 * support_dimension)),
        "galaxies": galaxy_rows,
        "invalid_prediction2": invalid,
        "prediction_manifest_sha256": canonical_sha256(prediction_hash_rows),
        "support_dimension": support_dimension,
    }


def build_receipt(root: Path) -> dict[str, Any]:
    """Derive the action and evaluate its frozen dimension counterfactuals."""

    root = root.resolve()
    config = load_config(root)
    packets = sorted(prepare_nonlocal_packets(root), key=lambda packet: packet["galaxy"].name)
    population = config["population"]
    if len(packets) != int(population["exploration_galaxies"]) or sum(
        packet["galaxy"].count for packet in packets
    ) != int(population["exploration_points"]):
        raise GravityG4AuxiliaryActionError("auxiliary-action population changed")
    symbolic = symbolic_derivation()
    counterfactuals = [
        _score_dimension(packets, dimension) for dimension in config["counterfactual_dimensions"]
    ]
    by_dimension = {int(row["support_dimension"]): row for row in counterfactuals}
    selected = by_dimension[2]
    predecessor = _load_json(root / str(config["predecessor_binding"]["path"]))
    parent_score = float(config["predecessor_binding"]["required_parent_chi_square"])
    parent_reproduction_error = abs(float(selected["chi_square"]) - parent_score)
    rar_chi = float(predecessor["controls"]["empirical_rar"]["chi_square"])
    nfw_chi = float(predecessor["scores"]["nfw_ceiling_chi_square"])
    point_count = sum(packet["galaxy"].count for packet in packets)
    nfw_limit = (
        nfw_chi + float(config["admission"]["nfw_ceiling_slack_chi_square_per_point"]) * point_count
    )
    admission = config["admission"]
    radial_positive = int(selected["invalid_prediction2"]) == 0 and all(
        np.all(action_prediction2(packet, support_dimension=2)["chi"] >= 0)
        and np.all(action_prediction2(packet, support_dimension=2)["psi"] >= 0)
        for packet in packets
    )
    obligations = {
        "boundary_normalized_auxiliary_solutions": "PASS",
        "causal_time_evolution": "PENDING_DIRECTED_COORDINATE_IS_NOT_TIME",
        "complete_positive_energy_and_stability": "PENDING",
        "conservation_identity": "PENDING_COVARIANT_COMPLETION",
        "covariant_action": "PENDING",
        "dimensionally_typed_radial_action": "PASS",
        "inherited_transition_scale_derived": "FAIL_INHERITED_FROM_V5",
        "radial_auxiliary_quadratic_positive": "PASS",
        "same_field_cluster_forward_model": "LOCKED_G4_BLOCKED",
        "same_field_lensing_equation": "LOCKED_G4_BLOCKED",
        "solar_system_and_gravitational_wave_limits": "PENDING",
    }
    gate_checks = {
        "all_predictions_positive_and_finite": radial_positive,
        "complete_covariant_conservation_stability_lensing_obligations": False,
        "D2_beats_D1_and_D3": (
            float(selected["chi_square"]) < float(by_dimension[1]["chi_square"])
            and float(selected["chi_square"]) < float(by_dimension[3]["chi_square"])
        ),
        "dimension_rule_reproduces_parent_beta_and_scale": (
            float(selected["beta"])
            == float(config["dimensional_closure_hypothesis"]["predicted_beta"])
            and float(selected["ell"])
            == float(config["dimensional_closure_hypothesis"]["predicted_log_radius_scale"])
        ),
        "inherited_transition_scale_is_first_principles": False,
        "numerically_reproduces_fold_stable_parent": (
            parent_reproduction_error <= float(admission["numerical_parent_chi_square_tolerance"])
        ),
        "per_galaxy_fitted_gravitational_constants_zero": True,
        "symbolic_auxiliary_euler_residuals_zero": bool(symbolic["all_exact_residuals_zero"]),
        "symbolic_integrated_flux_residual_zero": (symbolic["flux_identity_residual"] == "0"),
        "beats_rar_by_minimum_fraction": (
            1.0 - float(selected["chi_square"]) / rar_chi
            >= float(admission["minimum_fractional_gain_over_rar"])
        ),
        "within_nfw_performance_ceiling": (float(selected["chi_square"]) <= nfw_limit),
    }
    passed = all(gate_checks.values())
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G4_AUXILIARY_ACTION_DERIVATION",
        "decision": (
            "PASS_G4_AUXILIARY_ACTION_DERIVATION"
            if passed
            else "BLOCK_G4_AUXILIARY_ACTION_DERIVATION"
        ),
        "action": config["effective_action"],
        "claims": {
            "alternative_to_gr_discovered": False,
            "complete_first_principles_derivation": False,
            "confirmation_authorized": passed,
            "covariant_action_derived": False,
            "historical_novelty_established": False,
            "parent_terms_derived_from_one_effective_radial_action": bool(
                symbolic["all_exact_residuals_zero"]
            ),
            "parent_selection_independently_confirmed": False,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "confirmation_evaluator_accesses": 0,
            "counterfactual_dimensions": len(counterfactuals),
            "cross_scale_cluster_evaluator_accesses": 0,
            "cross_scale_lensing_evaluator_accesses": 0,
            "exploration_galaxies": len(packets),
            "exploration_points": point_count,
            "per_galaxy_fitted_gravitational_constants": 0,
            "scoring_point_evaluations": len(counterfactuals) * point_count,
        },
        "counterfactual_dimensions": [
            {key: value for key, value in row.items() if key != "galaxies"}
            for row in counterfactuals
        ],
        "dimensional_closure_hypothesis": config["dimensional_closure_hypothesis"],
        "first_principles_obligations": obligations,
        "gate_checks": gate_checks,
        "galaxies": selected["galaxies"],
        "inherited_unresolved_quantity": config["inherited_unresolved_quantity"],
        "parent_reproduction": {
            "absolute_chi_square_error": _metric(parent_reproduction_error),
            "expected_parent_chi_square": _metric(parent_score),
            "prediction_manifest_sha256": selected["prediction_manifest_sha256"],
            "reproduced_chi_square": selected["chi_square"],
        },
        "scores": {
            "action_chi_square": selected["chi_square"],
            "empirical_rar_chi_square": _metric(rar_chi),
            "fractional_gain_over_rar": _metric(1.0 - float(selected["chi_square"]) / rar_chi),
            "nfw_ceiling_chi_square": _metric(nfw_chi),
            "nfw_ceiling_excess": _metric(float(selected["chi_square"]) - nfw_limit),
            "nfw_ceiling_limit_with_slack": _metric(nfw_limit),
        },
        "symbolic_derivation": symbolic,
        "limitations": [
            "The action is an effective radial variational construction in x=log(r), not a four-dimensional covariant action.",
            "The D=2 closure was proposed after v5 exposed the 0.5 coefficient and 0.25 scale; its counterfactual is model-development evidence.",
            "The occupancy transition y=0.1 remains inherited and underived.",
            "The directed chi constraint is radial ordering, not causal time evolution.",
            "Positive radial auxiliary Hessians do not establish complete field-theory stability.",
            "No confirmation, cluster, or lensing data were opened.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "predecessor": _binding(root, str(config["predecessor_binding"]["path"])),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG4AuxiliaryActionError("auxiliary-action receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG4AuxiliaryActionError("auxiliary-action receipt seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG4AuxiliaryActionError("auxiliary-action config binding changed")
    expected = {
        "config": CONFIG_PATH,
        "predecessor": str(config["predecessor_binding"]["path"]),
        "source": SOURCE_PATH,
        "test": TEST_PATH,
    }
    for key, path in expected.items():
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG4AuxiliaryActionError(f"auxiliary-action {key} binding changed")
    counts = receipt.get("counts", {})
    if any(
        counts.get(key) != 0
        for key in (
            "confirmation_evaluator_accesses",
            "cross_scale_cluster_evaluator_accesses",
            "cross_scale_lensing_evaluator_accesses",
            "per_galaxy_fitted_gravitational_constants",
        )
    ):
        raise GravityG4AuxiliaryActionError("auxiliary-action violates data or fit lock")
    claims = receipt.get("claims", {})
    if claims.get("historical_novelty_established") is not False:
        raise GravityG4AuxiliaryActionError("auxiliary-action overstates novelty")
    if claims.get("covariant_action_derived") is not False:
        raise GravityG4AuxiliaryActionError("auxiliary-action overstates covariance")
    if claims.get("complete_first_principles_derivation") is not False:
        raise GravityG4AuxiliaryActionError("auxiliary-action overstates derivation")
    passed = receipt.get("decision") == "PASS_G4_AUXILIARY_ACTION_DERIVATION"
    if passed and (
        not all(receipt.get("gate_checks", {}).values())
        or claims.get("confirmation_authorized") is not True
    ):
        raise GravityG4AuxiliaryActionError("auxiliary-action PASS is unsupported")
    if not passed and claims.get("confirmation_authorized") is not False:
        raise GravityG4AuxiliaryActionError("blocked auxiliary action authorizes confirmation")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG4AuxiliaryActionError(
                f"refusing to overwrite immutable auxiliary-action receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_checked:
        validate_receipt(_load_json(root / OUTPUT_PATH), root=root)
        return 0
    receipt = build_receipt(root)
    _write_immutable(root / OUTPUT_PATH, receipt)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "counterfactual_dimensions": receipt["counterfactual_dimensions"],
                "decision": receipt["decision"],
                "gate_checks": receipt["gate_checks"],
                "scores": receipt["scores"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OUTPUT_PATH",
    "GravityG4AuxiliaryActionError",
    "action_prediction2",
    "build_receipt",
    "load_config",
    "symbolic_derivation",
    "validate_receipt",
]
